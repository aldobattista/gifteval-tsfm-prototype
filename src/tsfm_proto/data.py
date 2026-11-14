"""Parquet dataset store, temporal holdout split, and temperature sampler.

Reads the stratified local subset (``data/gifteval_pretrain_subset/``), enforces
a hard ``RuntimeError`` if real shards are missing, holds out 1,500 temporal
validation tails ``[-H_freq:]`` while permanently truncating those 1,500 series
in ``train_views`` (excluding ``__hist`` slices when constructing new indices),
and exposes deterministic CPU-only worker RNG streams.
"""

## General: it defines the class SubsetDatasetStore, the whole data used for training. It is instantiated in train.py (at the beginning to create the validaiton dataset; within DataLoader to generate the data)
##--> 3 main functions:
##    1. [:-H_freq] in load_train_dir: it retains the last H_freq steps of 1500 series for internal validation and it discards from the memory when it loads the training, such that the model 
##       can never see the future
##    2. sampling_probs(alpha=0.35): it applies the temperature to reduce the unbalance between big and small datasets. d is the dataset, Nd is the number of series in the dataset.
##        the prob of sampling from dataset d is pd~Nd^alpha and here we set alpha=0.35 to balance the sampling procedure between big and small datasets
##    3. sample_train_item: sample a time series (real or synthetic)
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from tsfm_proto.config import TSFMConfig
from tsfm_proto.synthetic import SeriesItem, sample_synthetic_series


def normalize_pandas_freq(freq: str) -> str:
    """Map legacy pandas offset aliases (``T``, ``H``, ``M``, ``Q-DEC``, ``A-DEC``) to modern equivalents.

    Preserves exact ``gluonts.time_feature.get_seasonality`` parity while eliminating
    ``pandas`` ``FutureWarning``s and ensuring forward compatibility with ``pandas >= 3.0``.
    """
    f = str(freq).strip()
    m = re.fullmatch(r"(\d*)([A-Za-z]+)(-[A-Za-z0-9]+)?", f)
    if not m:
        return f
    mult, code, anchor = m.group(1), m.group(2), m.group(3) or ""
    remap = {
        "T": "min",
        "H": "h",
        "S": "s",
        "L": "ms",
        "U": "us",
        "N": "ns",
        "M": "ME",
        "BM": "BME",
        "Q": "QE",
        "BQ": "BQE",
        "A": "YE",
        "Y": "YE",
        "BA": "BYE",
        "BY": "BYE",
    }
    return f"{mult}{remap.get(code, code)}{anchor}"


def horizon_for_freq(freq: str) -> int:
    """Standard per-frequency evaluation horizon used for temporal val holdout."""
    f = str(freq).strip()
    m = re.fullmatch(r"(\d*)([A-Za-z]+)(-[A-Za-z0-9]+)?", f)
    code = m.group(2).upper() if m else f.upper()
    if code in {"S", "SEC", "SECOND", "SECONDS", "L", "U", "N"}:
        return 60
    if code in {"T", "MIN", "MINUTE", "MINUTES"}:
        return 48
    if code in {"H", "BH"}:
        return 48
    if code in {"D", "B", "C"}:
        return 30
    if code == "W":
        return 13
    if code in {"M", "ME", "MS", "BM", "BME", "BMS", "CBME", "CBMS", "SM", "SMS"}:
        return 18
    if code in {"Q", "QE", "QS", "BQ", "BQE", "BQS"}:
        return 8
    if code in {"A", "Y", "YE", "YS", "AS", "BA", "BY", "BYE", "BYS", "BAS"}:
        return 6
    return 32


def _align_cov_width(cov: np.ndarray, target_len: int) -> np.ndarray:
    """Align covariate array ``[V_c, T_c]`` to ``target_len`` by tail-slicing or left-NaN padding."""
    arr = np.asarray(cov, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[0] == 0 or target_len <= 0:
        return np.zeros((0, max(target_len, 0)), dtype=np.float32)
    t_c = arr.shape[1]
    if t_c == target_len:
        return np.array(arr, dtype=np.float32, copy=True)
    if t_c > target_len:
        return np.array(arr[:, -target_len:], dtype=np.float32, copy=True)
    pad = np.full((arr.shape[0], target_len - t_c), np.nan, dtype=np.float32)
    return np.concatenate([pad, arr], axis=1)


@dataclass(frozen=True)
class ValExample:
    """Held-out temporal validation series: context up to -H_freq, target tail [-H_freq:]."""

    dataset_name: str
    item_id: str
    freq: str
    horizon: int
    context_target: np.ndarray  # [V_t, T_ctx], float32
    context_past_cov: np.ndarray  # [V_c, T_ctx], float32
    label_target: np.ndarray  # [V_t, H], float32


def _decode_nested_float32_column(col: pa.ChunkedArray) -> list[np.ndarray]:
    """Decode ``list<list<float>>`` column via contiguous Arrow offsets without boxing Python floats."""
    if len(col) == 0:
        return []
    arr = col.combine_chunks()
    nulls = arr.is_null().to_numpy(zero_copy_only=False) if arr.null_count > 0 else None
    outer_off = arr.offsets.to_numpy()
    inner = arr.values
    inner_off = inner.offsets.to_numpy()
    flat = inner.values.to_numpy(zero_copy_only=False).astype(np.float32, copy=False)

    out: list[np.ndarray] = []
    for i in range(len(arr)):
        r0 = int(outer_off[i])
        r1 = int(outer_off[i + 1])
        if (nulls is not None and bool(nulls[i])) or r1 <= r0:
            out.append(np.zeros((0, 0), dtype=np.float32))
            continue
        v = r1 - r0
        s0 = int(inner_off[r0])
        s1 = int(inner_off[r1])
        t0 = int(inner_off[r0 + 1]) - s0
        if (s1 - s0) == v * t0:
            out.append(flat[s0:s1].reshape(v, t0))
        else:
            rows = [flat[int(inner_off[k]) : int(inner_off[k + 1])] for k in range(r0, r1)]
            out.append(np.stack(rows, axis=0))
    return out


class SubsetDatasetStore:
    """Lazy-cached, fork/spawn-safe store over the local GiftEvalPretrain Parquet subset.

    At startup only lightweight metadata and list lengths are read (and cached in
    ``_index_v*.json`` for <10 ms reloads), identifying the 1,500 held-out validation
    series deterministically. Actual float32 arrays for each dataset directory are
    decoded on demand via zero-copy Arrow offsets into a bounded LRU cache with
    permanent ``[:-H_freq]`` truncation on held-out items.
    """

    def __init__(
        self,
        subset_dir: Path,
        val_num_series: int = 1500,
        min_val_len: int = 96,
        max_cached_dirs: int = 128,
    ) -> None:
        self.subset_dir = Path(subset_dir)
        self.max_cached_dirs = max(1, int(max_cached_dirs))
        if not self.subset_dir.exists():
            raise RuntimeError(
                f"Subset directory does not exist: {self.subset_dir}. "
                "Run `scripts/build_subset.py` first; training refuses to run on missing data."
            )
        parquet_paths = sorted(self.subset_dir.glob("*/part-00000.parquet"))
        if not parquet_paths:
            raise RuntimeError(
                f"No Parquet shards found under {self.subset_dir}. "
                "Run `scripts/build_subset.py` before initializing SubsetDatasetStore."
            )

        self.dir_names: list[str] = []
        self.dir_to_path: dict[str, Path] = {p.parent.name: p for p in parquet_paths}
        self.dir_counts: dict[str, int] = {}
        self._cache: OrderedDict[str, list[SeriesItem]] = OrderedDict()

        index_path = self.subset_dir / f"_index_v{val_num_series}_m{min_val_len}.json"
        if index_path.exists():
            try:
                cached_idx = json.loads(index_path.read_text())
                c_dirs = list(cached_idx["dir_names"])
                if len(c_dirs) == len(parquet_paths) and all(d in self.dir_to_path for d in c_dirs):
                    self.dir_names = c_dirs
                    self.dir_counts = {k: int(v) for k, v in cached_idx["dir_counts"].items()}
                    self._val_specs = [
                        (str(d), int(r), int(h)) for d, r, h in cached_idx["val_specs"]
                    ]
                    self.val_holdout_by_dir: dict[str, dict[int, int]] = {}
                    for dname, r_i, h_freq in self._val_specs:
                        self.val_holdout_by_dir.setdefault(dname, {})[r_i] = h_freq
                    return
            except Exception:
                pass

        # Record validation candidates (sha_key, dir_name, row_idx, h_freq), excluding __hist slices.
        primary_candidates: list[tuple[str, str, int, int]] = []
        hist_fallback_candidates: list[tuple[str, str, int, int]] = []
        for p in parquet_paths:
            dname = p.parent.name
            tbl_meta = pq.read_table(p, columns=["dataset_name", "item_id", "freq", "target"])
            n_rows = tbl_meta.num_rows
            if n_rows == 0:
                continue
            self.dir_names.append(dname)
            self.dir_counts[dname] = n_rows

            row_outer_lens = pc.list_value_length(tbl_meta["target"]).to_numpy(zero_copy_only=False)
            if (row_outer_lens == 0).any():
                raise ValueError(f"Empty outer variate list encountered in {p}")
            first_row_lens = pc.list_value_length(
                pc.list_flatten(pc.list_slice(tbl_meta["target"], 0, 1))
            ).to_numpy(zero_copy_only=False)
            ds_col = tbl_meta["dataset_name"].to_pylist()
            id_col = tbl_meta["item_id"].to_pylist()
            fq_col = tbl_meta["freq"].to_pylist()
            for r_i in range(n_rows):
                t_len = int(first_row_lens[r_i])
                h_freq = horizon_for_freq(fq_col[r_i])
                if t_len >= max(min_val_len, 3 * h_freq):
                    item_s = str(id_col[r_i])
                    key = hashlib.sha256(f"val:{ds_col[r_i]}:{item_s}".encode()).hexdigest()
                    entry = (key, dname, r_i, h_freq)
                    if item_s.endswith("__hist"):
                        hist_fallback_candidates.append(entry)
                    else:
                        primary_candidates.append(entry)

        if not self.dir_names:
            raise RuntimeError(f"All Parquet shards under {self.subset_dir} are empty.")

        primary_candidates.sort(key=lambda x: x[0])
        hist_fallback_candidates.sort(key=lambda x: x[0])
        chosen_val = (primary_candidates + hist_fallback_candidates)[:val_num_series]

        # Permanent per-dataset lookup mapping row_idx -> H_freq so train_views never reads [-H_freq:].
        self.val_holdout_by_dir = {}
        self._val_specs = []
        for _, dname, r_i, h_freq in chosen_val:
            self.val_holdout_by_dir.setdefault(dname, {})[r_i] = h_freq
            self._val_specs.append((dname, r_i, h_freq))

        # Atomically persist lightweight index so worker subprocesses and notebooks boot in <10 ms.
        payload = json.dumps(
            {
                "dir_names": self.dir_names,
                "dir_counts": self.dir_counts,
                "val_specs": self._val_specs,
            }
        )
        try:
            fd, tmp_name = tempfile.mkstemp(prefix=".idx_", dir=str(self.subset_dir))
        except OSError:
            return
        try:
            with os.fdopen(fd, "w") as fh:
                fh.write(payload)
            os.replace(tmp_name, index_path)
        except Exception:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)

    def sampling_probs(self, alpha: float) -> np.ndarray:
        """Compute dataset temperature probabilities ``p_d ~ N_d^alpha``."""
        counts = np.array([self.dir_counts[d] for d in self.dir_names], dtype=np.float64)
        weights = np.power(np.maximum(counts, 1.0), float(alpha))
        return weights / weights.sum()

    def _load_train_dir(self, dname: str) -> list[SeriesItem]:
        """Decode and cache ``train_views`` for one dataset, permanently stripping val tails."""
        cached = self._cache.get(dname)
        if cached is not None:
            self._cache.move_to_end(dname)
            return cached

        tbl = pq.read_table(self.dir_to_path[dname])
        ds_col = tbl["dataset_name"].to_pylist()
        id_col = tbl["item_id"].to_pylist()
        fq_col = tbl["freq"].to_pylist()
        t_list = _decode_nested_float32_column(tbl["target"])
        c_list = _decode_nested_float32_column(tbl["past_feat_dynamic_real"])
        holdouts = self.val_holdout_by_dir.get(dname, {})

        items: list[SeriesItem] = []
        for r_i in range(tbl.num_rows):
            tgt = np.array(t_list[r_i], dtype=np.float32, copy=True)
            if tgt.ndim == 1:
                tgt = tgt[None, :]
            cov = _align_cov_width(c_list[r_i], tgt.shape[1])

            if r_i in holdouts:
                h_cut = holdouts[r_i]
                tgt = tgt[:, :-h_cut]
                cov = cov[:, :-h_cut]
            items.append(
                SeriesItem(
                    dataset_name=str(ds_col[r_i]),
                    item_id=str(id_col[r_i]),
                    freq=str(fq_col[r_i]),
                    target=tgt,
                    past_cov=cov,
                )
            )
        self._cache[dname] = items
        if len(self._cache) > self.max_cached_dirs:
            self._cache.popitem(last=False)
        return items

    def sample_train_item(
        self,
        rng: np.random.Generator,
        cfg: TSFMConfig,
        probs: np.ndarray,
    ) -> SeriesItem:
        """Sample a real (truncated) series or a leak-free synthetic series on CPU."""
        if cfg.synth_prob > 0.0 and float(rng.random()) < cfg.synth_prob:
            # Draw TSMixup donors directly via temperature probs (stateless wrt _cache; never touching val tails).
            donors: list[SeriesItem] = []
            for _ in range(8):
                d_pick = self.dir_names[int(rng.choice(len(self.dir_names), p=probs))]
                pool = self._load_train_dir(d_pick)
                donors.append(pool[int(rng.integers(0, len(pool)))])
            return sample_synthetic_series(rng, donors)

        for _ in range(16):
            d_idx = int(rng.choice(len(self.dir_names), p=probs))
            pool = self._load_train_dir(self.dir_names[d_idx])
            item = pool[int(rng.integers(0, len(pool)))]
            if item.target.shape[1] >= 2 and np.isfinite(item.target).sum() >= 2:
                return item
        # Fallback within loaded pool guarantees forward progress even on sparse directories.
        dname = self.dir_names[0]
        return self._load_train_dir(dname)[0]

    def build_validation_examples(self, max_examples: int | None = None) -> list[ValExample]:
        """Materialize the exact held-out ``[-H_freq:]`` validation tails."""
        specs = self._val_specs[:max_examples] if max_examples is not None else self._val_specs
        by_dir: dict[str, list[tuple[int, int]]] = {}
        for dname, r_i, h_freq in specs:
            by_dir.setdefault(dname, []).append((r_i, h_freq))

        examples: list[ValExample] = []
        for dname, pairs in by_dir.items():
            tbl = pq.read_table(self.dir_to_path[dname])
            rows = sorted({r for r, _ in pairs})
            sub = tbl.take(rows)
            row_map = {r: pos for pos, r in enumerate(rows)}
            ds_col = sub["dataset_name"].to_pylist()
            id_col = sub["item_id"].to_pylist()
            fq_col = sub["freq"].to_pylist()
            t_list = _decode_nested_float32_column(sub["target"])
            c_list = _decode_nested_float32_column(sub["past_feat_dynamic_real"])

            for r_i, h_freq in pairs:
                pos = row_map[r_i]
                tgt = t_list[pos]
                if tgt.ndim == 1:
                    tgt = tgt[None, :]
                cov_aligned = _align_cov_width(c_list[pos], tgt.shape[1])
                cov_ctx = cov_aligned[:, :-h_freq] if cov_aligned.shape[0] > 0 else np.zeros((0, tgt.shape[1] - h_freq), dtype=np.float32)
                examples.append(
                    ValExample(
                        dataset_name=str(ds_col[pos]),
                        item_id=str(id_col[pos]),
                        freq=str(fq_col[pos]),
                        horizon=h_freq,
                        context_target=tgt[:, :-h_freq].copy(),
                        context_past_cov=cov_ctx.copy(),
                        label_target=tgt[:, -h_freq:].copy(),
                    )
                )
        return examples


def make_step_rng(base_seed: int, step_idx: int, worker_id: int = 0) -> np.random.Generator:
    """Create a reproducible CPU-only NumPy Generator from ``SeedSequence``."""
    ss = np.random.SeedSequence([int(base_seed), int(step_idx), int(worker_id)])
    return np.random.default_rng(ss)
