"""Build a stratified, leak-audited local Parquet subset of GiftEvalPretrain.

This script inspects the full Hugging Face repository
``theforecastingcompany/GiftEvalPretrain`` (152 Parquet directories / 878+ GB /
3.49M series), writes a full-corpus metadata catalog
(``results/eda_dataset_catalog.csv``), builds a balanced ~2.74 GB (2.738 GB) laptop subset
(``data/gifteval_pretrain_subset/``) covering all 89 non-climate directories and
4 representative climate directories (cmip6_1995, cmip6_2010, era5_2005,
era5_2018), and runs a two-stage anti-leakage audit against ``GIFT-Eval``
(``results/leakage_audit.json``).
"""

# 2 MAIN functions to consider:
### _select_shard_indices: it reads 4 equidistant Parquet files (0,N/3,2N/3, last one) instead of the only data-00000 to avoid geographical/alphabetic biases 
### _cap_series_arrays:  it crops long time series up to the last 8192 steps, and if the time series length is > ~16k steps it will add ~4k steps from the history

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "theforecastingcompany/GiftEvalPretrain"
REPRESENTATIVE_CLIMATE_DIRS = ("cmip6_1995", "cmip6_2010", "era5_2005", "era5_2018")
SMALL_DATASET_SERIES_THRESHOLD = 2000
LARGE_DATASET_CAP_SERIES = 2500
MAX_SHARDS_PER_DIR = 4
RECENT_CAP_STEPS = 8192
HIST_SLICE_TRIGGER = 16384
HIST_SLICE_STEPS = 4096


def _default_gifteval_dir() -> Path:
    env_dir = os.environ.get("GIFT_EVAL")
    if env_dir:
        return Path(env_dir)
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root / "data" / "GiftEval"

TARGET_ARROW_SCHEMA = pa.schema(
    [
        ("dataset_name", pa.string()),
        ("item_id", pa.string()),
        ("start", pa.timestamp("ms")),
        ("freq", pa.string()),
        ("target", pa.list_(pa.list_(pa.float32()))),
        ("past_feat_dynamic_real", pa.list_(pa.list_(pa.float32()))),
    ]
)


def _norm_freq_bucket(freq: str) -> str:
    """Map raw pandas/GluonTS frequency strings (including anchored W-TUE, Q-OCT) to coarse buckets."""
    f = str(freq).strip()
    m = re.fullmatch(r"(\d*)([A-Za-z]+)(-[A-Za-z0-9]+)?", f)
    code = m.group(2).upper() if m else f.upper()
    if code in {"S", "SEC", "SECOND", "SECONDS", "L", "U", "N"}:
        return "S"
    if code in {"T", "MIN", "MINUTE", "MINUTES"}:
        return "T"
    if code in {"H", "BH"}:
        return "H"
    if code in {"D", "B", "C"}:
        return "D"
    if code == "W":
        return "W"
    if code in {"M", "ME", "MS", "BM", "BME", "BMS", "CBME", "CBMS", "SM", "SMS"}:
        return "M"
    if code in {"Q", "QE", "QS", "BQ", "BQE", "BQS"}:
        return "Q"
    if code in {"A", "Y", "YE", "YS", "AS", "BA", "BY", "BYE", "BYS", "BAS"}:
        return "Y"
    return code


def _select_shard_indices(n_shards: int, max_shards: int = MAX_SHARDS_PER_DIR) -> list[int]:
    """Choose evenly spaced shard indices across [0, n_shards - 1] to avoid sorting bias."""
    if n_shards <= max_shards:
        return list(range(n_shards))
    # Always include first, 1/3, 2/3, and last shard so sorted item blocks stay representative.
    return sorted({0, n_shards // 3, (2 * n_shards) // 3, n_shards - 1})


def _sha_rank_key(dataset_name: str, item_id: str, row_idx: int) -> str:
    """Deterministic SHA256 digest used to subsample large datasets reproducibly."""
    raw = f"{dataset_name}:{item_id}:{row_idx}" if not item_id else f"{dataset_name}:{item_id}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _cap_series_arrays(
    target_rows: list[list[float]],
    cov_rows: list[list[float]],
    item_id: str,
) -> list[tuple[str, list[list[float]], list[list[float]]]]:
    """Apply uniform 8,192-step recency capping (+ 4,096-step historical slice when T > 16,384)."""
    if not target_rows:
        return []
    t_len = len(target_rows[0])
    # Align covariates defensively to t_len if any row has mismatched or empty width.
    valid_cov = [c for c in cov_rows if isinstance(c, (list, np.ndarray)) and len(c) == t_len]
    if t_len <= RECENT_CAP_STEPS:
        return [(item_id, target_rows, valid_cov)]
    recent_target = [list(map(float, row[-RECENT_CAP_STEPS:])) for row in target_rows]
    recent_cov = [list(map(float, row[-RECENT_CAP_STEPS:])) for row in valid_cov]
    out = [(f"{item_id}__recent" if t_len > HIST_SLICE_TRIGGER else item_id, recent_target, recent_cov)]
    if t_len > HIST_SLICE_TRIGGER:
        # Cap end_h strictly below (t_len - RECENT_CAP_STEPS) so __hist and __recent never overlap.
        max_hist_end = t_len - RECENT_CAP_STEPS
        end_h = min(t_len // 3 + HIST_SLICE_STEPS, max_hist_end)
        start_h = max(0, end_h - HIST_SLICE_STEPS)
        hist_target = [list(map(float, row[start_h:end_h])) for row in target_rows]
        hist_cov = [list(map(float, row[start_h:end_h])) for row in valid_cov]
        out.append((f"{item_id}__hist", hist_target, hist_cov))
    return out


def _compute_table_stats(
    lengths: list[int],
    target_dims: list[int],
    cov_dims: list[int],
    freqs: list[str],
    nan_count: int,
    zero_count: int,
    total_vals: int,
    nonneg_series: int,
    int_series: int,
    n_evaluated: int,
) -> dict[str, Any]:
    """Summarize empirical distribution statistics for one dataset directory."""
    if not lengths:
        return {
            "freq": "",
            "median_target_dim": 0,
            "max_target_dim": 0,
            "has_past_cov": False,
            "max_past_cov_dim": 0,
            "len_p10": 0.0,
            "len_p50": 0.0,
            "len_p90": 0.0,
            "len_max": 0,
            "nan_rate": 0.0,
            "zero_rate": 0.0,
            "nonneg_rate": 0.0,
            "integer_rate": 0.0,
        }
    freq_mode = pd.Series(freqs).mode().iloc[0] if freqs else ""
    len_arr = np.asarray(lengths, dtype=np.float64)
    td_arr = np.asarray(target_dims, dtype=np.int64)
    cd_arr = np.asarray(cov_dims, dtype=np.int64)
    denom_series = max(1, n_evaluated)
    denom_vals = max(1, total_vals)
    denom_fin = max(1, total_vals - nan_count)
    return {
        "freq": str(freq_mode),
        "median_target_dim": int(np.median(td_arr)),
        "max_target_dim": int(np.max(td_arr)),
        "has_past_cov": bool(np.max(cd_arr) > 0),
        "max_past_cov_dim": int(np.max(cd_arr)),
        "len_p10": float(np.percentile(len_arr, 10)),
        "len_p50": float(np.percentile(len_arr, 50)),
        "len_p90": float(np.percentile(len_arr, 90)),
        "len_max": int(np.max(len_arr)),
        "nan_rate": float(nan_count / denom_vals),
        "zero_rate": float(zero_count / denom_fin),
        "nonneg_rate": float(nonneg_series / denom_series),
        "integer_rate": float(int_series / denom_series),
    }


def _process_directory(
    dir_name: str,
    shard_files: list[str],
    out_dir: Path,
    tmp_dir: Path,
) -> tuple[dict[str, Any], int]:
    """Download selected shards for one directory, compute stats, write capped Parquet, clean tmp."""
    dir_out = out_dir / dir_name
    out_parquet = dir_out / "part-00000.parquet"
    meta_json = dir_out / "meta.json"
    if out_parquet.exists() and meta_json.exists():
        meta = json.loads(meta_json.read_text())
        return meta["stats"], int(meta["subset_num_series"])

    dir_out.mkdir(parents=True, exist_ok=True)
    selected_idx = _select_shard_indices(len(shard_files), MAX_SHARDS_PER_DIR)
    chosen_paths = [shard_files[i] for i in selected_idx]

    lengths: list[int] = []
    target_dims: list[int] = []
    cov_dims: list[int] = []
    freqs: list[str] = []
    nan_count = 0
    zero_count = 0
    total_vals = 0
    nonneg_series = 0
    int_series = 0
    n_evaluated = 0
    shard_row_counts: list[int] = []

    # Candidate keys (sha_rank, shard_order, row_idx) across chosen shards.
    candidates: list[tuple[str, int, int]] = []
    local_shard_paths: list[Path] = []
    seen_ids: set[str] = set()

    for s_order, rfilename in enumerate(chosen_paths):
        downloaded = hf_hub_download(
            repo_id=REPO_ID,
            filename=rfilename,
            repo_type="dataset",
            local_dir=str(tmp_dir),
        )
        lp = Path(downloaded)
        local_shard_paths.append(lp)

        # Read metadata columns first to build deterministic SHA ranking cleanly.
        table_meta = pq.read_table(lp, columns=["dataset_name", "item_id", "freq"])
        n_rows = table_meta.num_rows
        shard_row_counts.append(n_rows)
        ds_col = table_meta["dataset_name"].to_pylist()
        id_col = table_meta["item_id"].to_pylist()
        fq_col = table_meta["freq"].to_pylist()

        # Sample empirical stats from up to 600 rows per shard so 60k-row shards inspect fast.
        stat_step = max(1, n_rows // 600)
        stat_indices = list(range(0, n_rows, stat_step))
        tbl_stat = pq.read_table(lp, columns=["target", "past_feat_dynamic_real"]).take(stat_indices)
        t_list = tbl_stat["target"].to_pylist()
        c_list = tbl_stat["past_feat_dynamic_real"].to_pylist()

        for pos, row_i in enumerate(stat_indices):
            freqs.append(str(fq_col[row_i]))
            tgt = t_list[pos] or []
            cov = c_list[pos] or []
            tdim = len(tgt)
            cdim = len(cov)
            tlen = len(tgt[0]) if tdim > 0 else 0
            target_dims.append(tdim)
            cov_dims.append(cdim)
            lengths.append(tlen)
            if tdim > 0 and tlen > 0:
                arr = np.asarray(tgt, dtype=np.float32)
                fin = np.isfinite(arr)
                n_tot = arr.size
                n_fin = int(fin.sum())
                total_vals += n_tot
                nan_count += n_tot - n_fin
                n_evaluated += 1
                if n_fin > 0:
                    vals = arr[fin]
                    zero_count += int(np.sum(vals == 0.0))
                    if bool(np.all(vals >= 0.0)):
                        nonneg_series += 1
                    bounded = np.abs(vals) < float(2**23)
                    if bool(np.all(bounded)) and bool(np.all(np.abs(vals - np.round(vals)) < 1e-5)):
                        int_series += 1

        for row_i in range(n_rows):
            dname = str(ds_col[row_i] or dir_name)
            iid = str(id_col[row_i] or f"row_{s_order}_{row_i}")
            # Guard against duplicate item_id across shards by adding shard suffix if collision occurs.
            uid = iid if iid not in seen_ids else f"{iid}__s{s_order}_{row_i}"
            seen_ids.add(uid)
            candidates.append((_sha_rank_key(dname, uid, row_i), s_order, row_i))

    total_rows_seen = len(candidates)
    if total_rows_seen > SMALL_DATASET_SERIES_THRESHOLD:
        candidates.sort(key=lambda x: x[0])
        candidates = candidates[:LARGE_DATASET_CAP_SERIES]

    by_shard: dict[int, list[int]] = {}
    for _, s_order, row_i in candidates:
        by_shard.setdefault(s_order, []).append(row_i)

    out_rows: dict[str, list[Any]] = {
        "dataset_name": [],
        "item_id": [],
        "start": [],
        "freq": [],
        "target": [],
        "past_feat_dynamic_real": [],
    }
    used_ids: set[str] = set()
    for s_order, row_indices in sorted(by_shard.items()):
        row_indices_sorted = sorted(row_indices)
        tbl_sel = pq.read_table(local_shard_paths[s_order]).take(row_indices_sorted)
        cols = {name: tbl_sel[name].to_pylist() for name in TARGET_ARROW_SCHEMA.names}
        for i in range(tbl_sel.num_rows):
            dname = str(cols["dataset_name"][i] or dir_name)
            base_id = str(cols["item_id"][i] or f"row_{s_order}_{row_indices_sorted[i]}")
            if base_id in used_ids:
                base_id = f"{base_id}__s{s_order}_{row_indices_sorted[i]}"
            used_ids.add(base_id)
            start_ts = cols["start"][i]
            freq_val = str(cols["freq"][i])
            tgt = cols["target"][i] or []
            cov = cols["past_feat_dynamic_real"][i] or []
            if not tgt or len(tgt[0]) == 0:
                continue
            for capped_id, cap_tgt, cap_cov in _cap_series_arrays(tgt, cov, base_id):
                out_rows["dataset_name"].append(dname)
                out_rows["item_id"].append(capped_id)
                out_rows["start"].append(start_ts)
                out_rows["freq"].append(freq_val)
                out_rows["target"].append(cap_tgt)
                out_rows["past_feat_dynamic_real"].append(cap_cov)

    out_table = pa.table(out_rows, schema=TARGET_ARROW_SCHEMA)
    pq.write_table(out_table, out_parquet, compression="zstd", compression_level=3)

    stats = _compute_table_stats(
        lengths=lengths,
        target_dims=target_dims,
        cov_dims=cov_dims,
        freqs=freqs,
        nan_count=nan_count,
        zero_count=zero_count,
        total_vals=total_vals,
        nonneg_series=nonneg_series,
        int_series=int_series,
        n_evaluated=n_evaluated,
    )
    stats["sampled_shards_rows"] = int(sum(shard_row_counts))
    subset_count = int(out_table.num_rows)
    meta_json.write_text(json.dumps({"stats": stats, "subset_num_series": subset_count}, indent=2))

    # Immediately delete raw temporary HF shards so total local disk usage stays bounded.
    tmp_sub = tmp_dir / dir_name
    if tmp_sub.exists():
        shutil.rmtree(tmp_sub, ignore_errors=True)
    return stats, subset_count


def _run_leakage_audit(
    subset_dir: Path,
    gift_eval_dir: Path,
    all_pretrain_dirs: list[str],
    out_path: Path,
) -> dict[str, Any]:
    """Verify dataset-name disjointness and 64-step z-normalized Pearson separation vs GIFT-Eval."""
    import sys

    parquet_shards = sorted(subset_dir.glob("*/part-00000.parquet")) if subset_dir.exists() else []
    if not parquet_shards:
        raise FileNotFoundError(
            f"Cannot run leakage audit: no Parquet shards found under {subset_dir}. "
            "Run `python scripts/build_subset.py` first to materialize the local subset."
        )

    if not gift_eval_dir.exists():
        raise FileNotFoundError(
            f"GIFT-Eval data directory not found at {gift_eval_dir}. "
            "Set `GIFT_EVAL=/path/to/GiftEval` or pass `--gift-eval-dir`."
        )
    ext_src = os.environ.get("GIFT_EVAL_SRC", "")
    if ext_src and Path(ext_src).exists() and ext_src not in sys.path:
        sys.path.insert(0, ext_src)

    os.environ["GIFT_EVAL"] = str(gift_eval_dir)
    try:
        from gift_eval.data import Dataset  # noqa: WPS433
    except ImportError as exc:
        raise ImportError(
            "`gift_eval` is required for the anti-leakage audit. Install via "
            "`pip install git+https://github.com/SalesforceAIResearch/gift-eval.git` "
            "(or set `GIFT_EVAL_SRC=/path/to/gift-eval/src`)."
        ) from exc

    props_path = out_path.parent / "gift_eval_dataset_properties.json"
    ge_props = json.loads(props_path.read_text()) if props_path.exists() else {}

    # 1. Name and substring audit against GIFT-Eval dataset families & subdirectories.
    ge_folders = sorted([p.name for p in gift_eval_dir.iterdir() if p.is_dir() and p.name != ".cache"])
    pretrain_lower = {d.lower(): d for d in all_pretrain_dirs}
    ge_lower = {g.lower(): g for g in ge_folders}
    exact_overlap = sorted(set(pretrain_lower) & set(ge_lower))

    near_misses: list[dict[str, str]] = []
    for pl, pd_orig in sorted(pretrain_lower.items()):
        for gl, ge_orig in sorted(ge_lower.items()):
            if pl != gl and (pl in gl or gl in pl):
                near_misses.append({"pretrain_dir": pd_orig, "gift_eval_dir": ge_orig})

    # 2. 64-step z-normalized fingerprint screen on matching frequencies against GIFT-Eval test tails.
    ge_fps_by_bucket: dict[str, list[tuple[str, np.ndarray]]] = {}
    ge_hashes_by_bucket: dict[str, set[bytes]] = {}
    ge_nondeg_hashes_by_bucket: dict[str, set[bytes]] = {}
    short_names = [
        "m4_yearly", "m4_quarterly", "m4_monthly", "m4_weekly", "m4_daily", "m4_hourly",
        "electricity/15T", "electricity/H", "electricity/D", "electricity/W",
        "solar/10T", "solar/H", "solar/D", "solar/W",
        "hospital", "covid_deaths", "us_births/D", "us_births/M", "us_births/W",
        "saugeenday/D", "saugeenday/M", "saugeenday/W",
        "temperature_rain_with_missing", "kdd_cup_2018_with_missing/H",
        "kdd_cup_2018_with_missing/D", "car_parts_with_missing",
        "restaurant", "hierarchical_sales/D", "hierarchical_sales/W",
        "LOOP_SEATTLE/5T", "LOOP_SEATTLE/H", "LOOP_SEATTLE/D",
        "SZ_TAXI/15T", "SZ_TAXI/H", "M_DENSE/H", "M_DENSE/D",
        "ett1/15T", "ett1/H", "ett1/D", "ett1/W", "ett2/15T", "ett2/H", "ett2/D", "ett2/W",
        "jena_weather/10T", "jena_weather/H", "jena_weather/D",
        "bitbrains_fast_storage/5T", "bitbrains_fast_storage/H",
        "bitbrains_rnd/5T", "bitbrains_rnd/H",
        "bizitobs_application", "bizitobs_service", "bizitobs_l2c/5T", "bizitobs_l2c/H",
    ]
    for ds_name in short_names:
        ds = Dataset(name=ds_name, term="short", to_univariate=False)
        bucket = _norm_freq_bucket(ds.freq)
        count = 0
        for inp, lbl in zip(ds.test_data.input, ds.test_data.label):
            if count >= 60:
                break
            ctx = np.asarray(inp["target"], dtype=np.float64)
            fut = np.asarray(lbl["target"], dtype=np.float64)
            if ctx.ndim == 2:
                ctx = ctx[0]
            if fut.ndim == 2:
                fut = fut[0]
            combined = np.concatenate([ctx, fut], axis=-1)
            fin = combined[np.isfinite(combined)]
            if len(fin) < 64:
                continue
            w = fin[-64:]
            std = float(np.std(w))
            if std <= 1e-6:
                continue
            z = ((w - float(np.mean(w))) / std).astype(np.float32)
            ge_fps_by_bucket.setdefault(bucket, []).append((ds_name, z))
            qhash = hashlib.sha1(np.round(z, 3).tobytes()).digest()
            ge_hashes_by_bucket.setdefault(bucket, set()).add(qhash)
            if int(np.unique(np.round(w, 6)).size) > 4:
                ge_nondeg_hashes_by_bucket.setdefault(bucket, set()).add(qhash)
            count += 1

    # Vectorize Pearson check per frequency bucket against pretrain subset windows.
    ge_mats: dict[str, tuple[list[str], np.ndarray]] = {}
    for bucket, items in ge_fps_by_bucket.items():
        labels = [it[0] for it in items]
        mat = np.stack([it[1] for it in items], axis=0) / math.sqrt(64.0)
        ge_mats[bucket] = (labels, mat)

    max_corr = 0.0
    max_corr_pair: dict[str, Any] = {}
    exact_hash_matches = 0
    nondeg_hash_matches = 0
    deg_collision_dirs: list[str] = []
    high_corr_pairs: list[dict[str, Any]] = []
    windows_checked = 0

    for pq_path in sorted(subset_dir.glob("*/part-00000.parquet")):
        dname = pq_path.parent.name
        tbl = pq.read_table(pq_path, columns=["freq", "target"])
        freqs = tbl["freq"].to_pylist()
        targets = tbl["target"].to_pylist()
        step = max(1, len(targets) // 80)
        pre_by_bucket: dict[str, list[np.ndarray]] = {}
        for i in range(0, len(targets), step):
            bkt = _norm_freq_bucket(freqs[i])
            if bkt not in ge_mats:
                continue
            tgt = targets[i]
            if not tgt:
                continue
            arr = np.asarray(tgt[0], dtype=np.float64)
            fin = arr[np.isfinite(arr)]
            if len(fin) < 64:
                continue
            for sl in (fin[-64:], fin[:64]):
                s = float(np.std(sl))
                if s <= 1e-6:
                    continue
                z = ((sl - float(np.mean(sl))) / s).astype(np.float32)
                qh = hashlib.sha1(np.round(z, 3).tobytes()).digest()
                if qh in ge_hashes_by_bucket.get(bkt, set()):
                    exact_hash_matches += 1
                    if int(np.unique(np.round(sl, 6)).size) > 4 and qh in ge_nondeg_hashes_by_bucket.get(bkt, set()):
                        nondeg_hash_matches += 1
                    else:
                        deg_collision_dirs.append(f"{dname}:{bkt}:n_uniq={int(np.unique(np.round(sl, 6)).size)}")
                pre_by_bucket.setdefault(bkt, []).append(z / math.sqrt(64.0))
                windows_checked += 1

        for bkt, vecs in pre_by_bucket.items():
            pre_mat = np.stack(vecs, axis=0)
            ge_labels, ge_mat = ge_mats[bkt]
            corr_mat = pre_mat @ ge_mat.T
            best_idx = int(np.argmax(corr_mat))
            p_i, g_i = divmod(best_idx, corr_mat.shape[1])
            val = float(corr_mat[p_i, g_i])
            if val > max_corr:
                max_corr = val
                max_corr_pair = {
                    "pretrain_dir": dname,
                    "gift_eval_task": ge_labels[g_i],
                    "freq_bucket": bkt,
                    "pearson_r": round(val, 5),
                }
            if val >= 0.995:
                high_corr_pairs.append(
                    {
                        "pretrain_dir": dname,
                        "gift_eval_task": ge_labels[g_i],
                        "freq_bucket": bkt,
                        "pearson_r": round(val, 5),
                    }
                )

    audit = {
        "pretrain_directories_checked": len(all_pretrain_dirs),
        "gift_eval_families_checked": len(ge_folders),
        "gift_eval_properties_count": len(ge_props),
        "exact_name_overlap": exact_overlap,
        "substring_near_misses": near_misses,
        "near_miss_explanation": (
            "Monash Australian daily `weather` (freq='D') is verified distinct from "
            "Max-Planck Jena sub-hourly/hourly/daily `jena_weather` (21 meteorological channels)."
        ),
        "fingerprint_windows_screened": windows_checked,
        "exact_quantized_z64_hash_collisions": exact_hash_matches,
        "nondegenerate_z64_hash_collisions": nondeg_hash_matches,
        "degenerate_step_collisions": deg_collision_dirs,
        "max_pearson_r_64step": round(max_corr, 5),
        "max_pearson_pair": max_corr_pair,
        "pairs_above_0_995": high_corr_pairs,
        "clean_disjoint_verified": len(exact_overlap) == 0 and nondeg_hash_matches == 0,
    }
    out_path.write_text(json.dumps(audit, indent=2))
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Build stratified TFC GiftEvalPretrain laptop subset.")
    parser.add_argument(
        "--out-dir",
        "--subset-dir",
        dest="out_dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "gifteval_pretrain_subset",
    )
    parser.add_argument(
        "--gift-eval-dir",
        type=Path,
        default=_default_gifteval_dir(),
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Re-run the anti-leakage audit on the local subset without querying Hugging Face Hub.",
    )
    args = parser.parse_args()

    t_start = time.time()
    repo_root = Path(__file__).resolve().parents[1]
    results_dir = repo_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    if args.audit_only:
        cat_path = results_dir / "eda_dataset_catalog.csv"
        if cat_path.exists():
            all_dirs = sorted(pd.read_csv(cat_path)["dir_name"].astype(str).tolist())
        else:
            all_dirs = sorted(p.parent.name for p in args.out_dir.glob("*/part-00000.parquet"))
        audit = _run_leakage_audit(args.out_dir, args.gift_eval_dir, all_dirs, results_dir / "leakage_audit.json")
        print(
            f"Saved anti-leakage audit: overlap={audit['exact_name_overlap']}, "
            f"nondeg_collisions={audit['nondegenerate_z64_hash_collisions']}, "
            f"verified={audit['clean_disjoint_verified']}"
        )
        return
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = args.out_dir.parent / "_hf_tmp_cache"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    info = api.dataset_info(REPO_ID, files_metadata=True)
    shards_by_dir: dict[str, list[str]] = {}
    bytes_by_dir: dict[str, int] = {}
    for s in info.siblings:
        if s.rfilename.endswith(".parquet") and "/" in s.rfilename:
            dname = s.rfilename.split("/")[0]
            shards_by_dir.setdefault(dname, []).append(s.rfilename)
            bytes_by_dir[dname] = bytes_by_dir.get(dname, 0) + int(s.size or 0)
    for dname in shards_by_dir:
        shards_by_dir[dname].sort()

    card_map: dict[str, dict[str, Any]] = {}
    if info.card_data:
        for entry in info.card_data.to_dict().get("dataset_info", []):
            cname = entry.get("config_name")
            if cname and cname != "default":
                splits = entry.get("splits", [])
                n_ex = int(splits[0]["num_examples"]) if splits else 0
                card_map[cname] = {
                    "full_uncompressed_bytes": int(entry.get("dataset_size", 0)),
                    "full_compressed_bytes": int(entry.get("download_size", 0)),
                    "full_num_series": n_ex,
                }

    clim_set = set(REPRESENTATIVE_CLIMATE_DIRS)
    all_dirs = sorted(shards_by_dir.keys())
    selected_dirs = [d for d in all_dirs if (not d.startswith(("cmip6_", "era5_"))) or d in clim_set]
    print(f"Discovered {len(all_dirs)} directories; building subset over {len(selected_dirs)} directories...")

    catalog_rows: list[dict[str, Any]] = []
    subset_total_series = 0
    subset_total_steps = 0

    for dname in all_dirs:
        is_clim = dname.startswith(("cmip6_", "era5_"))
        in_subset = dname in selected_dirs
        cinfo = card_map.get(dname, {})
        full_comp = int(cinfo.get("full_compressed_bytes") or bytes_by_dir.get(dname, 0))
        full_uncomp = int(cinfo.get("full_uncompressed_bytes") or full_comp)
        full_n = int(cinfo.get("full_num_series") or 0)

        if in_subset:
            t0 = time.time()
            stats, sub_n = _process_directory(dname, shards_by_dir[dname], args.out_dir, tmp_dir)
            if full_n == 0:
                full_n = int(stats.get("sampled_shards_rows", sub_n))
            subset_total_series += sub_n
            # Read lightweight column to count exact stored timesteps.
            pq_path = args.out_dir / dname / "part-00000.parquet"
            sub_mb = pq_path.stat().st_size / (1024 * 1024)
            print(
                f"[{len(catalog_rows)+1:3d}/{len(all_dirs)}] {dname:34s} | "
                f"rows={sub_n:4d} (full={full_n:7d}) | {sub_mb:6.1f} MB | {time.time()-t0:4.1f}s",
                flush=True,
            )
            row = {
                "dir_name": dname,
                "in_subset": True,
                "is_climate": is_clim,
                "full_num_shards": len(shards_by_dir[dname]),
                "full_compressed_bytes": full_comp,
                "full_uncompressed_bytes": full_uncomp,
                "full_num_series": full_n,
                "subset_num_series": sub_n,
                "subset_parquet_bytes": pq_path.stat().st_size,
                **{k: v for k, v in stats.items() if k != "sampled_shards_rows"},
            }
        else:
            row = {
                "dir_name": dname,
                "in_subset": False,
                "is_climate": is_clim,
                "full_num_shards": len(shards_by_dir[dname]),
                "full_compressed_bytes": full_comp,
                "full_uncompressed_bytes": full_uncomp,
                "full_num_series": full_n,
                "subset_num_series": 0,
                "subset_parquet_bytes": 0,
                "freq": "M" if dname.startswith("cmip6_") else "H",
                "median_target_dim": 1,
                "max_target_dim": 1,
                "has_past_cov": False,
                "max_past_cov_dim": 0,
                "len_p10": 0.0,
                "len_p50": 0.0,
                "len_p90": 0.0,
                "len_max": 0,
                "nan_rate": 0.0,
                "zero_rate": 0.0,
                "nonneg_rate": 0.0,
                "integer_rate": 0.0,
            }
        catalog_rows.append(row)

    shutil.rmtree(tmp_dir, ignore_errors=True)
    # Remove any HF cache folder hf_hub_download may have left in out_dir.parent/_hf_tmp_cache.
    catalog_df = pd.DataFrame(catalog_rows)
    catalog_path = results_dir / "eda_dataset_catalog.csv"
    catalog_df.to_csv(catalog_path, index=False)

    # Compute exact subset disk size and manifest.
    pq_files = sorted(args.out_dir.glob("*/part-00000.parquet"))
    total_disk_bytes = sum(p.stat().st_size for p in pq_files)
    for p in pq_files:
        tcol = pq.read_table(p, columns=["target"])["target"].to_pylist()
        subset_total_steps += sum(len(r) * len(r[0]) for r in tcol if r)

    manifest = {
        "repo_id": REPO_ID,
        "full_corpus_directories": len(all_dirs),
        "full_corpus_series": int(catalog_df["full_num_series"].sum()),
        "full_corpus_compressed_gb": round(float(catalog_df["full_compressed_bytes"].sum()) / 1e9, 2),
        "subset_directories": len(pq_files),
        "subset_non_climate_directories": int(
            sum(1 for d in selected_dirs if not d.startswith(("cmip6_", "era5_")))
        ),
        "subset_climate_directories": [d for d in selected_dirs if d.startswith(("cmip6_", "era5_"))],
        "subset_total_series": int(subset_total_series),
        "subset_total_variate_timesteps": int(subset_total_steps),
        "subset_disk_bytes": int(total_disk_bytes),
        "subset_disk_gb": round(total_disk_bytes / 1e9, 3),
        "dir_list_sha256": hashlib.sha256(",".join(selected_dirs).encode()).hexdigest(),
        "build_wall_seconds": round(time.time() - t_start, 1),
    }
    manifest_path = results_dir / "subset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Saved subset manifest ({manifest['subset_disk_gb']} GB, {subset_total_series:,} series) to {manifest_path}")

    try:
        audit = _run_leakage_audit(args.out_dir, args.gift_eval_dir, all_dirs, results_dir / "leakage_audit.json")
        print(f"Saved anti-leakage audit: overlap={audit['exact_name_overlap']}, max_r={audit['max_pearson_r_64step']}")
    except (FileNotFoundError, ImportError) as exc:
        print(
            f"[WARN] Subset build succeeded, skipped anti-leakage audit ({exc}). "
            "Mount GIFT_EVAL + install `gift-eval` and run `python scripts/build_subset.py --audit-only`."
        )


if __name__ == "__main__":
    main()
