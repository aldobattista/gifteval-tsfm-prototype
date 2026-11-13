# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
# ---

# %% [markdown]
# # 01 — Data Pipeline, Exploratory Analysis, Anti-Leakage Audit & Aligned Strip Packing
#
# **Goal:** Empirically verify every data handling, normalization, seasonal lag, and batch-packing decision built into `gifteval-tsfm-prototype` against real shards from [`theforecastingcompany/GiftEvalPretrain`](https://huggingface.co/datasets/theforecastingcompany/GiftEvalPretrain).
#
# Each section poses a concrete engineering question, walks through the math and implementation in `src/tsfm_proto/`, and visualizes the exact empirical evidence.

# %%
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path.cwd() if (Path.cwd() / "src" / "tsfm_proto").exists() else Path.cwd().parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from tsfm_proto.config import TSFMConfig
from tsfm_proto.data import SubsetDatasetStore, make_step_rng, normalize_pandas_freq
from tsfm_proto.packing import pack_strip_batch
from tsfm_proto.scaling import _scale_equivariant_stats_1d, build_group_patch_features

plt.rcParams.update(
    {
        "figure.dpi": 115,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
    }
)
print(f"Repository: {REPO_ROOT.name} (src/tsfm_proto present={(REPO_ROOT / 'src' / 'tsfm_proto').exists()})")

# %% [markdown]
# ## 1. Full Corpus Pareto Skew & Stratified Subset Construction
#
# **Question:** Why sample a `2.74 GB` compressed multi-shard subset across `93` directories instead of streaming `100%` of the `878.26 GB` full corpus?
#
# Loading `results/eda_dataset_catalog.csv` (`152` Parquet directories, `3,490,516` series) shows extreme storage and count skew:
# - Gridded climate outputs (`63` `cmip6_*` and `era5_*` directories) account for `90.2%` of total compressed bytes.
# - Simulated building energy traces (`buildings_900k`) alone account for `51.4%` of all series in the full corpus.
# - Conversely, all `67` small/medium datasets ($\le 2{,}000$ series each) require just `2.60 GB` in total—`0.3%` of the corpus bytes.
#
# Keeping `100%` of small/medium datasets while sampling large datasets across `4` evenly spaced shards (`0, N//3, 2N//3, N-1`) preserves every domain and low-resource frequency (`A`, `Q`, `M`, `W`) cleanly on a single laptop.

# %%
catalog = pd.read_csv(REPO_ROOT / "results" / "eda_dataset_catalog.csv")
manifest = json.loads((REPO_ROOT / "results" / "subset_manifest.json").read_text())

print("Subset Manifest Summary:")
for k in (
    "full_corpus_directories",
    "full_corpus_series",
    "full_corpus_compressed_gb",
    "subset_directories",
    "subset_non_climate_directories",
    "subset_total_series",
    "subset_total_variate_timesteps",
    "subset_disk_gb",
):
    print(f"  {k:32s}: {manifest[k]:,}" if isinstance(manifest[k], int) else f"  {k:32s}: {manifest[k]}")

top_bytes = catalog.sort_values("full_compressed_bytes", ascending=False).head(8)
print("\nTop 8 Full-Corpus Directories by Compressed Size:")
print(
    top_bytes[
        ["dir_name", "is_climate", "in_subset", "full_num_series", "subset_num_series", "full_compressed_bytes"]
    ].to_string(index=False)
)

# %%
fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.2), constrained_layout=True)

sorted_b = np.sort(catalog["full_compressed_bytes"].to_numpy(dtype=np.float64))[::-1]
cum_b = np.cumsum(sorted_b) / sorted_b.sum()
axes[0].plot(np.arange(1, len(cum_b) + 1), cum_b * 100.0, lw=2.3, color="#1f4e79")
axes[0].axvline(64, color="#b03a2e", ls="--", lw=1.4, label="Top-64 byte dirs = 97.2% (63 climate = 90.2%)")
axes[0].set_xlabel("Datasets ranked by compressed bytes")
axes[0].set_ylabel("Cumulative % of 878.26 GB corpus")
axes[0].set_title("Storage Skew Across 152 Corpus Directories")
axes[0].legend(loc="lower right", framealpha=0.95, facecolor="white")

sub_df = catalog[catalog["in_subset"]].copy()
sub_df["freq_norm"] = sub_df["freq"].astype(str).map(normalize_pandas_freq)
freq_counts = sub_df.groupby("freq_norm")["subset_num_series"].sum().sort_values(ascending=False).head(10)
axes[1].bar(freq_counts.index, freq_counts.values, color="#2e86c1")
axes[1].set_title("Top-10 Frequencies (93,300 of 96,831 Subset Segments; +3,531 Rare)")
axes[1].set_xlabel("Canonicalized Time-Series Cadence (Pandas Offset)")
axes[1].set_ylabel("Segment Count in Subset")
axes[1].tick_params(axis="x", rotation=30)
for lbl in axes[1].get_xticklabels():
    lbl.set_ha("right")
plt.show()

# %% [markdown]
# ## 2. Zero-Leakage Protocol (`GIFT-Eval` Disjointness & Validation Tail Truncation)
#
# **Audit Takeaways (`results/leakage_audit.json`):**
# 1. `exact_name_overlap == []` across all `152` pretraining directories vs. the `28` `GIFT-Eval` families (`weather` [`3,220` daily Monash Australia `'D'` + `16` hourly `'H'` `oikolab_weather` rows, modal `'D'` in `leakage_audit.json`] $\neq$ Max-Planck 21-channel `jena_weather`).
# 2. `nondegenerate_z64_hash_collisions == 0`. The `4` exact quantized 64-step z-normalized hash collisions reported in `pairs_above_0_995` (`uber_tlc_hourly` / `bdg-2_fox` vs. `bitbrains_fast_storage/H`) arise solely from degenerate binary two-level step windows (`n_uniq == 2`) whose z-normalized vectors algebraically collapse regardless of domain—highlighting why first-difference ($\Delta y$) and non-degenerate support screens belong in `docs/04_with_more_time_roadmap.md`.
# 3. Internally, `SubsetDatasetStore` holds out `1,500` temporal validation tails `[-H_freq:]` across `66` datasets (`710` plain + `790` items paired with a same-`base_id` `__recent`/`__hist` mate in the local subset) and **permanently truncates** those tails in `train_views` and `TSMixup` (`1,425 / 1,500` free of exact `[-H_freq:]` mate-tail sequence collisions in cached `_index_v1500_m96.json`—`0.5662 -> 0.5301` `Run 5` vs. `0.5289` `Run 4` `Point-SN` soup `val_nmwql`, and `1,336 / 1,500` [`1,326` finite] = `0.5631 -> 0.5281` `Run 5` vs. `0.5267` `Run 4` after also excluding `89` non-constant cross-directory upstream duplicates, vs. `0.5696 -> 0.5289` across all `1,500`; newly built indices exclude `__hist` slices when primary series suffice).

# %%
_TMP_SUBSET_CTX = None


def _open_subset_store() -> tuple[SubsetDatasetStore, bool]:
    """Open local 93-directory subset if present, else build a tiny deterministic fixture for bare clones."""
    global _TMP_SUBSET_CTX
    real_dir = REPO_ROOT / "data" / "gifteval_pretrain_subset"
    if real_dir.exists() and any(real_dir.glob("*/part-00000.parquet")):
        return SubsetDatasetStore(real_dir, val_num_series=1500), False

    _TMP_SUBSET_CTX = tempfile.TemporaryDirectory(prefix="tsfm_nb01_")
    tmp_root = Path(_TMP_SUBSET_CTX.name)
    rng_fix = np.random.default_rng(7)
    specs = [
        ("demo_large_cov", 40, 2, 160),
        ("demo_medium", 16, 0, 144),
        ("demo_small", 6, 0, 128),
        ("demo_rare", 2, 0, 112),
    ]
    for dname, n_ser, n_cov, t_len in specs:
        ds_dir = tmp_root / dname
        ds_dir.mkdir(parents=True, exist_ok=True)
        t_ax = np.arange(t_len, dtype=np.float32)
        targets = [
            [(np.sin(2.0 * np.pi * t_ax / 24.0 + 0.3 * s_i) + 0.1 * rng_fix.normal(size=t_len)).astype(np.float32).tolist()]
            for s_i in range(n_ser)
        ]
        covs = [
            [np.cos(2.0 * np.pi * t_ax / 12.0 + c_i).astype(np.float32).tolist() for c_i in range(n_cov)] if n_cov > 0 else []
            for _ in range(n_ser)
        ]
        tbl = pa.table(
            {
                "dataset_name": [dname] * n_ser,
                "item_id": [f"s{j}" for j in range(n_ser)],
                "freq": ["H"] * n_ser,
                "target": pa.array(targets, type=pa.list_(pa.list_(pa.float32()))),
                "past_feat_dynamic_real": pa.array(covs, type=pa.list_(pa.list_(pa.float32()))),
            }
        )
        pq.write_table(tbl, ds_dir / "part-00000.parquet")
    return SubsetDatasetStore(tmp_root, val_num_series=2, min_val_len=96), True


audit = json.loads((REPO_ROOT / "results" / "leakage_audit.json").read_text())
print(f"Pretrain dirs checked: {audit['pretrain_directories_checked']}")
print(f"GIFT-Eval families checked: {audit['gift_eval_families_checked']}")
print(f"Exact name overlap: {audit['exact_name_overlap']}")
print(f"Non-degenerate z64 hash collisions (n_uniq > 4): {audit['nondegenerate_z64_hash_collisions']}")
print(f"Degenerate two-level step collisions (n_uniq == 2): {audit['exact_quantized_z64_hash_collisions']}")
print(f"Clean disjoint verified: {audit['clean_disjoint_verified']}")
print(f"Near-miss explanation: {audit['near_miss_explanation']}")

store, is_demo_fixture = _open_subset_store()
if is_demo_fixture:
    print("\nNote: local data/gifteval_pretrain_subset/ not mounted; using deterministic temporary Parquet fixture.")
val_examples = store.build_validation_examples(max_examples=25)
val_example0 = val_examples[0]

# Confirm exact boundary alignment and disjointness in train_views vs val_examples.
first_dname, first_row_idx, first_h = store._val_specs[0]
train_pool_first = store._load_train_dir(first_dname)
train_item_first = train_pool_first[first_row_idx]
assert train_item_first.target.shape[1] == val_example0.context_target.shape[1]
assert np.allclose(
    np.nan_to_num(train_item_first.target, nan=0.0),
    np.nan_to_num(val_example0.context_target, nan=0.0),
)
print(
    f"\nVerified holdout boundary for `{first_dname}` row {first_row_idx}: "
    f"train_views T={train_item_first.target.shape[1]}, held-out H_freq={val_example0.horizon} stripped permanently."
)

# %% [markdown]
# ## 3. Dataset Temperature Balancing ($\alpha = 1.0 \to \alpha = 0.35$)
#
# **Question:** Why change $\alpha=1.0$ (`Run 0`) to $\alpha=0.35$ (`Runs 1..5`) when sampling `93` directories?
#
# Under proportional sampling ($p_d \propto N_d^{1.0}$), `2,500–5,000`-segment capped directories dominate gradient updates while rare `2–50`-series healthcare/economic datasets receive negligible weight. Conversely, pure uniform sampling ($\alpha = 0.0$) risks memorizing tiny datasets in a single pass. $\alpha = 0.35$ compresses the max/min sampling ratio across the `93` subset directories by `162x` (`2,500.0x` down to `15.5x`) while preserving monotonic volume priority.

# %%
counts = np.array([store.dir_counts[d] for d in store.dir_names], dtype=np.float64)
p10 = store.sampling_probs(1.0)
p05 = store.sampling_probs(0.5)
p035 = store.sampling_probs(0.35)
p00 = store.sampling_probs(0.0)

def _norm_entropy(p: np.ndarray) -> float:
    return float(-(p * np.log(p + 1e-12)).sum() / np.log(len(p)))

print(
    f"Max/Min sampling ratio & normalized entropy across {len(counts)} dirs:\n"
    f"  alpha=1.00 : max/min = {p10.max()/p10.min():6.1f}x | H_norm = {_norm_entropy(p10):.3f}\n"
    f"  alpha=0.50 : max/min = {p05.max()/p05.min():6.1f}x | H_norm = {_norm_entropy(p05):.3f}\n"
    f"  alpha=0.35 : max/min = {p035.max()/p035.min():6.1f}x | H_norm = {_norm_entropy(p035):.3f}\n"
    f"  alpha=0.00 : max/min = {p00.max()/p00.min():6.1f}x | H_norm = {_norm_entropy(p00):.3f}"
)

fig, ax = plt.subplots(figsize=(9.5, 3.8), constrained_layout=True)
order = np.argsort(counts)[::-1]
x = np.arange(len(counts))
ax.plot(x, p10[order] * 100.0, label=r"$\alpha = 1.00$ (Run 0)", lw=2.0, color="#7f8c8d")
ax.plot(x, p035[order] * 100.0, label=r"$\alpha = 0.35$ (Runs 1..5)", lw=2.2, color="#1f4e79")
ax.plot(x, p00[order] * 100.0, label=r"$\alpha = 0.00$ (Uniform)", lw=1.6, ls="--", color="#b03a2e")
ax.set_xlabel("Subset Directory Rank (sorted by series count)")
ax.set_ylabel("Sampling Probability (%)")
ax.set_title(r"Effect of Temperature $\alpha$ on Dataset Sampling Exposure")
ax.legend()
plt.show()

# %% [markdown]
# ## 4. Unit-Equivariant `arcsinh` Scaling, `W = 64` Local Scale & 4-Cycle Blackout Lags
#
# Three subtle invariants in `src/tsfm_proto/scaling.py` prevent numerical failure in zero-shot forecasting:
# 1. **Exact Scale Equivariance on Micro-SI Series:** `_scale_equivariant_stats_1d` checks relative variation `s > max(1e-6 * |mu|, 1e-12)` instead of adding a fixed `+0.01` floor in variance space. Thus rescaling a micro-SI series in $[10^{-6}, 10^{-5}]$ up to $[1, 10]$ (both strictly above the `1e-12` underflow floor) preserves `1.50e-15` (`< 2e-15` below; $\mathcal{O}((|\mu|/\sigma)\epsilon_{\text{f64}})$ generally) invariance in `float64` scalar `arcsinh` normalization, and `< 5e-5` across tested decimal SI rescalings (`0.0` on exact powers of two, `~9e-7` on this low-offset signal, `1.32e-5` in `test_causality_and_masks.py`) inside `float32`-cast `build_group_patch_features`.
# 2. **Observed-Only Moments (`Runs 2..5`):** Missing values do not pull $\mu$ toward zero or shrink/distort $\sigma$, while all-missing patches stay zeroed so blanked-key masking works cleanly.
# 3. **`W = 64` Trailing Local Scale + 4-Cycle Observed Lags (`Runs 3..5`):** Even if a contiguous `64`-step pre-origin blackout blanks two full `P=32` patches right before $T_{\text{origin}}$, observed lags step backward across $m \in \{0, 1, 2, 3\}$ multiples of $S \in \{7, 12, 24, 48, 52, 96\}$ to retrieve real seasonal anchors.

# %%
# 1. Scale-equivariance demonstration across 6 orders of magnitude.
rng = np.random.default_rng(7)
base = 2.5e-6 + 4.0e-7 * np.sin(np.linspace(0, 8 * np.pi, 128))
mu_micro, sig_micro = _scale_equivariant_stats_1d(base)
mu_macro, sig_macro = _scale_equivariant_stats_1d(base * 1e6)
norm_micro = np.arcsinh((base - mu_micro) / sig_micro)
norm_macro = np.arcsinh((base * 1e6 - mu_macro) / sig_macro)
print(
    f"Max abs diff between 1e-6 SI series and 1e+0 rescaled series after arcsinh normalization: "
    f"{float(np.max(np.abs(norm_micro - norm_macro))):.2e}"
)

# 2. 64-step pre-origin blackout demonstration on Run 3+ stem features (C_in = 17).
t = np.arange(160, dtype=np.float32)
signal = (100.0 + 0.25 * t + 12.0 * np.sin(2.0 * np.pi * t / 24.0))[None, :]
signal_blk = signal.copy()
signal_blk[:, -48:] = np.nan  # Blank the trailing 48 steps right before origin.

patched, obs_any, mu_obs, sig_obs = build_group_patch_features(
    y_ctx_real=signal_blk,
    left_pad=(-160) % 32,
    n_ctx_patches=5,
    n_hor_patches=2,
    patch_size=32,
    scaling_mode="observed_frozen_origin",
    use_local_scale_and_lags=True,
)
# Channel indices: 7 = lag S=24 value, 13 = lag_ok[S=24].
lag24_val_hor0 = patched[0, 5, :, 7]
lag24_ok_hor0 = patched[0, 5, :, 13]
print(
    f"Horizon patch +1 (under 48-step pre-origin blackout): "
    f"lag_24 resolved on {int(lag24_ok_hor0.sum())}/32 steps via 4-cycle fallback; "
    f"lag_24 range = [{lag24_val_hor0.min():.2f}, {lag24_val_hor0.max():.2f}]"
)

# %% [markdown]
# ## 5. Universal Modulo-32 Origin Alignment & `[1, 32, 128]` Strip Packing
#
# In `src/tsfm_proto/packing.py`, *every* context length $T_{\text{ctx}} \ge 1$ receives `left_pad = (-T_ctx) % 32` slack steps so that the true forecast origin $T_{\text{origin}}$ always sits at **timestep `31` of patch `rel_patch_idx == 0`**. Multiple heterogeneous variable-length windows pack cleanly side-by-side onto a static `[1, 32, 128]` canvas with `~99%` occupancy and zero cross-segment contamination.

# %%
cfg5 = TSFMConfig.from_yaml(REPO_ROOT / "configs" / "run5_coupled_synth_mix.yaml")
probs5 = store.sampling_probs(cfg5.sampling_alpha)
batch_rng = make_step_rng(cfg5.seed, step_idx=1, worker_id=0)
packed = pack_strip_batch(lambda r: store.sample_train_item(r, cfg5, probs5), batch_rng, cfg5)

seg_np = packed.segment_id[0].numpy()  # [32, 128]
rel_np = packed.rel_patch_idx[0].numpy()
cov_np = packed.patch_is_cov[0].numpy()
obs_np = packed.patch_obs_any[0].numpy()
lmask_np = packed.loss_mask[0].numpy()

occupancy = float((seg_np >= 0).mean()) * 100.0
origin_cols = rel_np == 0
n_segments = int(seg_np.max()) + 1
n_groups = int(packed.patch_group_id[0].max().item()) + 1
n_cov_patches = int(((seg_np >= 0) & cov_np).sum())
print(
    f"Packed batch shape: {tuple(packed.patch_inputs.shape)}\n"
    f"Strip occupancy   : {occupancy:.2f}% ({int((seg_np >= 0).sum())}/{seg_np.size} patch slots)\n"
    f"Packed segments   : {n_segments} across {n_groups} groups | cov_patches={n_cov_patches} | active loss steps={int(lmask_np.sum()):,}"
)

fig, ax = plt.subplots(figsize=(11.5, 4.4), constrained_layout=True)
# Discrete 7-class token encoding separating observed past covariates from unobserved future covariate horizon slots.
from matplotlib.colors import BoundaryNorm, ListedColormap

vis = np.zeros(seg_np.shape, dtype=np.int32)
active = seg_np >= 0
vis[active & (~cov_np) & (rel_np < 0) & obs_np] = 1
vis[active & (~cov_np) & origin_cols & obs_np] = 2
vis[active & (~cov_np) & (rel_np > 0)] = 3
vis[active & cov_np & (rel_np <= 0) & obs_np] = 4
vis[active & cov_np & (rel_np > 0)] = 5
vis[active & (rel_np <= 0) & (~obs_np)] = 6
cmap = ListedColormap(["#EAEDED", "#5DADE2", "#1B4F72", "#E67E22", "#27AE60", "#94A3B8", "#DC2626"])
norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5], cmap.N)
im = ax.imshow(vis, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
ax.grid(False)
ax.set_xlabel("Patch Column l (0 .. 127; P=32 steps per patch -> 4,096 steps/row)")
ax.set_ylabel("Canvas Row r (0 .. 31)")
ax.set_title(
    f"[1, 32, 128] Aligned Strip Packing Layout ({occupancy:.2f}% occupancy, {n_segments} segments, {n_cov_patches} cov patches)"
)
cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2, 3, 4, 5, 6])
cbar.ax.set_yticklabels([
    "Slack (-1)",
    "Target Ctx (<0)",
    "Target Origin (0)",
    "Target Horizon (>0)",
    "Obs. Past Cov (<=0)",
    "Cov Horizon (off)",
    "Masked Ctx (obs=0)",
])
plt.show()

# %% [markdown]
# ## Summary
# - The `2.74 GB` (`93`-directory, `96,831`-segment) stratified subset eliminates `99.7%` of raw storage bloat from gridded climate archives while preserving `100%` of small/medium datasets and all sampling frequencies.
# - `SubsetDatasetStore` enforces zero name overlap (`0 / 28` families) and permanently strips validation tails `[-H_freq:]` from `train_views` (`790 / 1,500` paired mates; `1,425 / 1,500` free of exact `[-H_freq:]` mate-tail collisions in the cached holdout index, and `1,336 / 1,500` [`1,326` finite] after excluding `89` non-constant cross-directory upstream duplicates; `Run 4` slightly edges `Run 5` on `1,425`/`1,336` `Point-SN` `val_nmwql` [`0.5289`/`0.5267` vs. `0.5301`/`0.5281`]; newly built indices exclude `__hist` slices when primary candidates suffice).
# - Modulo-32 left padding aligns `T_origin` cleanly onto patch `rel_patch_idx == 0`, packing variable-length target and covariate windows to `>99%` density on `[1, 32, 128]`.
