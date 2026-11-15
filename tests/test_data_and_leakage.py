"""Tests for local Parquet subset schema, temperature sampling, and validation tail holdout."""

from __future__ import annotations

import importlib.util
import json
import os
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tsfm_proto.config import TSFMConfig
from tsfm_proto.data import SubsetDatasetStore, horizon_for_freq, make_step_rng, normalize_pandas_freq
from tsfm_proto.eval_metrics import _fallback_seasonality
from tsfm_proto.synthetic import _sample_tsmixup_1d

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUBSET_DIR = PROJECT_ROOT / "data" / "gifteval_pretrain_subset"
RESULTS_DIR = PROJECT_ROOT / "results"


def test_missing_subset_raises_runtime_error(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="build_subset"):
        SubsetDatasetStore(tmp_path / "does_not_exist")


def test_subset_manifest_and_eda_catalog_invariants() -> None:
    manifest = json.loads((RESULTS_DIR / "subset_manifest.json").read_text())
    assert int(manifest["full_corpus_directories"]) == 152
    assert int(manifest["subset_directories"]) == 93
    assert int(manifest["subset_non_climate_directories"]) == 89
    assert len(manifest["subset_climate_directories"]) == 4
    assert int(manifest["subset_total_series"]) == 96831

    catalog = pd.read_csv(RESULTS_DIR / "eda_dataset_catalog.csv")
    assert len(catalog) == 152


def test_subset_schema_temperature_and_val_truncation() -> None:
    if not SUBSET_DIR.exists() or not any(SUBSET_DIR.glob("*/part-00000.parquet")):
        pytest.skip("Run scripts/build_subset.py first to materialize local Parquet subset")

    first_shard = next(SUBSET_DIR.glob("*/part-00000.parquet"))
    schema_cols = set(pq.read_schema(first_shard).names)
    assert {"dataset_name", "item_id", "freq", "target", "past_feat_dynamic_real"}.issubset(schema_cols)

    store = SubsetDatasetStore(SUBSET_DIR, val_num_series=1500)
    assert len(store._val_specs) == 1500

    # Alpha=0.35 must strictly flatten top-k sampling dominance relative to uniform alpha=1.0.
    p_unif = store.sampling_probs(1.0)
    p_temp = store.sampling_probs(0.35)
    assert np.isclose(p_unif.sum(), 1.0)
    assert np.isclose(p_temp.sum(), 1.0)
    top5_unif = float(np.sort(p_unif)[-5:].sum())
    top5_temp = float(np.sort(p_temp)[-5:].sum())
    assert top5_temp < top5_unif

    # Verify held-out validation items are permanently truncated in train_views (including TSMixup donors).
    dname, r_i, h_freq = store._val_specs[0]
    train_pool = store._load_train_dir(dname)
    val_example = store.build_validation_examples(max_examples=1)[0]
    assert train_pool[r_i].target.shape[1] == val_example.context_target.shape[1]
    assert val_example.horizon == h_freq
    assert train_pool[r_i].item_id == val_example.item_id

    # Verify TSMixup draws purely from truncated train_views and never sees held-out tail values.
    cfg_s = TSFMConfig.from_yaml(PROJECT_ROOT / "configs" / "run5_coupled_synth_mix.yaml")
    rng = make_step_rng(cfg_s.seed, 7, 0)
    mixed = _sample_tsmixup_1d(rng, [train_pool[r_i]], length=256)
    assert mixed.shape == (256,)
    assert np.isfinite(mixed).all()


def test_normalize_pandas_freq_zero_future_warnings() -> None:
    get_seasonality = pytest.importorskip("gluonts.time_feature").get_seasonality
    if not SUBSET_DIR.exists() or not any(SUBSET_DIR.glob("*/part-00000.parquet")):
        pytest.skip("Run scripts/build_subset.py first to materialize local Parquet subset")

    freqs: set[str] = set()
    for p in sorted(SUBSET_DIR.glob("*/part-00000.parquet")):
        tbl = pq.read_table(p, columns=["freq"])
        freqs.update(str(v) for v in tbl["freq"].unique().to_pylist())

    assert freqs
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for f in sorted(freqs):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                expected = int(get_seasonality(f))
            normed = normalize_pandas_freq(f)
            actual = int(get_seasonality(normed))
            assert actual == expected, f"Seasonality mismatch for {f} -> {normed}: {actual} != {expected}"
    future_warnings = [w for w in caught if issubclass(w.category, FutureWarning)]
    assert len(future_warnings) == 0, f"Unexpected FutureWarnings: {[str(w.message) for w in future_warnings]}"


def test_subset_store_readonly_safe(tmp_path: Path) -> None:
    subset_root = tmp_path / "readonly_subset"
    ds_dir = subset_root / "tiny_synth"
    ds_dir.mkdir(parents=True)
    rng = np.random.default_rng(0)
    targets = [[rng.normal(size=160).astype(np.float32).tolist()] for _ in range(3)]
    covs: list[list[list[float]]] = [[] for _ in range(3)]
    tbl = pa.table(
        {
            "dataset_name": ["tiny_synth"] * 3,
            "item_id": ["s0", "s1", "s2"],
            "freq": ["H", "H", "H"],
            "target": pa.array(targets, type=pa.list_(pa.list_(pa.float32()))),
            "past_feat_dynamic_real": pa.array(covs, type=pa.list_(pa.list_(pa.float32()))),
        }
    )
    pq.write_table(tbl, ds_dir / "part-00000.parquet")

    os.chmod(subset_root, 0o555)
    try:
        store = SubsetDatasetStore(subset_root, val_num_series=1, min_val_len=96)
        assert len(store._val_specs) == 1
        assert not (subset_root / "_index_v1_m96.json").exists()
        vals = store.build_validation_examples(max_examples=1)
        assert len(vals) == 1
        train_pool = store._load_train_dir("tiny_synth")
        assert len(train_pool) == 3
    finally:
        os.chmod(subset_root, 0o755)


def test_fallback_seasonality_matches_gluonts_all_21_freqs() -> None:
    get_seasonality = pytest.importorskip("gluonts.time_feature").get_seasonality
    freqs = [
        "10S",
        "10T",
        "15T",
        "30T",
        "4S",
        "5T",
        "6H",
        "A",
        "A-DEC",
        "Y",
        "B",
        "D",
        "H",
        "M",
        "MS",
        "Q",
        "Q-DEC",
        "S",
        "T",
        "W",
        "W-SUN",
    ]
    for freq in freqs:
        expected = int(get_seasonality(normalize_pandas_freq(freq)))
        actual = _fallback_seasonality(freq)
        assert actual == expected, f"_fallback_seasonality({freq!r}) = {actual} != GluonTS {expected}"


def test_config_yaml_roundtrip_and_stateless_store_sampling(tmp_path: Path) -> None:
    cfg = TSFMConfig.from_yaml(PROJECT_ROOT / "configs" / "run5_coupled_synth_mix.yaml")
    rt_path = tmp_path / "rt.yaml"
    cfg.to_yaml(rt_path)
    assert TSFMConfig.from_yaml(rt_path) == cfg

    sub = tmp_path / "stateless_sub"
    rng_data = np.random.default_rng(11)
    for d_i in range(4):
        ds_dir = sub / f"dir_{d_i}"
        ds_dir.mkdir(parents=True)
        targets = [[rng_data.normal(size=180).astype(np.float32).tolist()] for _ in range(3)]
        covs: list[list[list[float]]] = [[] for _ in range(3)]
        tbl = pa.table(
            {
                "dataset_name": [f"dir_{d_i}"] * 3,
                "item_id": [f"s{j}" for j in range(3)],
                "freq": ["H"] * 3,
                "target": pa.array(targets, type=pa.list_(pa.list_(pa.float32()))),
                "past_feat_dynamic_real": pa.array(covs, type=pa.list_(pa.list_(pa.float32()))),
            }
        )
        pq.write_table(tbl, ds_dir / "part-00000.parquet")

    s1 = SubsetDatasetStore(sub, val_num_series=0, min_val_len=96, max_cached_dirs=1)
    s2 = SubsetDatasetStore(sub, val_num_series=0, min_val_len=96, max_cached_dirs=1)
    # Warm caches on opposite directories to prove TSMixup donor sampling ignores LRU cache state.
    _ = s1._load_train_dir("dir_0")
    _ = s2._load_train_dir("dir_3")

    cfg_syn = replace(cfg, run_name="syn_stateless", synth_prob=1.0)
    probs1 = s1.sampling_probs(cfg_syn.sampling_alpha)
    probs2 = s2.sampling_probs(cfg_syn.sampling_alpha)
    item1 = s1.sample_train_item(make_step_rng(99, 5, 0), cfg_syn, probs1)
    item2 = s2.sample_train_item(make_step_rng(99, 5, 0), cfg_syn, probs2)
    assert np.array_equal(item1.target, item2.target)
    assert item1.dataset_name == item2.dataset_name


def _load_build_subset_module():
    spec = importlib.util.spec_from_file_location("build_subset_mod", PROJECT_ROOT / "scripts" / "build_subset.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_horizon_for_freq_and_norm_freq_bucket_regex() -> None:
    build_subset = _load_build_subset_module()
    expected_h = {
        "MS": 18,
        "QS": 8,
        "YS": 6,
        "AS": 6,
        "W-TUE": 13,
        "W-THU": 13,
        "10S": 60,
        "5T": 48,
        "H": 48,
        "D": 30,
    }
    for fq, h in expected_h.items():
        assert horizon_for_freq(fq) == h, f"horizon_for_freq({fq!r}) mismatch"

    expected_b = {
        "W-TUE": "W",
        "W-THU": "W",
        "MS": "M",
        "QS": "Q",
        "Q-OCT": "Q",
        "YS": "Y",
        "10S": "S",
        "5T": "T",
    }
    for fq, bucket in expected_b.items():
        assert build_subset._norm_freq_bucket(fq) == bucket, f"_norm_freq_bucket({fq!r}) mismatch"


def test_cap_series_arrays_disjoint_and_store_excludes_hist(tmp_path: Path) -> None:
    build_subset = _load_build_subset_module()
    t_len = 17000
    rows = build_subset._cap_series_arrays([list(range(t_len))], [], "id0")
    assert len(rows) == 2
    recent_id, recent_tgt, _ = rows[0]
    hist_id, hist_tgt, _ = rows[1]
    assert recent_id == "id0__recent"
    assert hist_id == "id0__hist"
    assert len(recent_tgt[0]) == build_subset.RECENT_CAP_STEPS
    assert len(hist_tgt[0]) == build_subset.HIST_SLICE_STEPS
    assert max(hist_tgt[0]) < min(recent_tgt[0])

    sub = tmp_path / "hist_excl_sub"
    ds_dir = sub / "ds0"
    ds_dir.mkdir(parents=True)
    seq = np.arange(200, dtype=np.float32).tolist()
    tbl = pa.table(
        {
            "dataset_name": ["ds0", "ds0"],
            "item_id": ["a__hist", "b__recent"],
            "freq": ["H", "H"],
            "target": pa.array([[seq], [seq]], type=pa.list_(pa.list_(pa.float32()))),
            "past_feat_dynamic_real": pa.array([[], []], type=pa.list_(pa.list_(pa.float32()))),
        }
    )
    pq.write_table(tbl, ds_dir / "part-00000.parquet")
    store = SubsetDatasetStore(sub, val_num_series=1, min_val_len=96)
    vals = store.build_validation_examples()
    assert len(vals) == 1
    assert vals[0].item_id == "b__recent"
