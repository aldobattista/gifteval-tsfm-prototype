# 01 — Data Pipeline, Corpus EDA & Strip Packing

## 1. Unified Schema & Full-Corpus Taxonomy (`theforecastingcompany/GiftEvalPretrain`)

[`theforecastingcompany/GiftEvalPretrain`](https://huggingface.co/datasets/theforecastingcompany/GiftEvalPretrain) unifies heterogeneous public time-series archives into a standardized Parquet schema (`152` Parquet directories, `3,490,516` series, `878.26 GB` compressed):
- `dataset_name` (`string`): Source dataset identifier.
- `item_id` (`string`): Series identifier within the dataset.
- `freq` (`string`): Native sampling frequency string (e.g., `S`, `5T`, `10T`, `15T`, `H`, `D`, `W`, `M`, `Q-DEC`, `A-DEC`).
- `target` (`list<list<float32>>`): Target channels of shape `[V_t, T]`.
- `past_feat_dynamic_real` (`list<list<float32>>`): Historical exogenous covariates aligned over `[V_c, T]` (present in `13 / 93` subset directories = `8,238 / 96,831` (`8.5%`) series, with `91.0%` concentrated in `alibaba_cluster_trace_2018`, `azure_vm_traces_2017`, and `borg_cluster_data_2011`, and empty in `80 / 93` directories).

### Extreme Storage & Series Count Skew

Full-corpus profiling ([eda_dataset_catalog.csv](../results/eda_dataset_catalog.csv)) reveals severe concentration along two orthogonal axes:
1. **Byte Skew:** Gridded Earth-system climate simulations (`63` `cmip6_*` and `era5_*` directories) occupy **`90.2%`** of total compressed storage (`878.26 GB`), despite sharing high spatial/temporal redundancy across grid cells.
2. **Series-Count Skew:** Simulated building load traces (`buildings_900k`) represent over half (`51.4%`) of all series in the full corpus, whereas all `67` small-to-medium datasets ($\le 2{,}000$ series each) consume just `2.60 GB` (`0.3%`) combined.

---

## 2. Deterministic Stratified Subset (`93` Directories, `2.74 GB`, `96,831` Parquet Segments)

To fit cleanly on an Apple Silicon laptop without losing rare frequencies (`A`, `Q`, `M`, `W`) or falling into shard-ordering bias, `scripts/build_subset.py` builds `data/gifteval_pretrain_subset/` (`2,737,606,113` bytes = `2.738 GB` compressed, `96,831` Parquet series segments representing `~2.8%` of full-corpus series and `0.31%` of bytes, `1,022,231,331` variate-timesteps; [subset_manifest.json](../results/subset_manifest.json)):
1. **Domain Coverage:** Retains **100%** of non-climate directories (`89 / 89`) plus **`4` representative climate regimes** spanning decadal physics (`cmip6_1995`, `cmip6_2010`, `era5_2005`, `era5_2018`) = **`93` directories total**. Note that **`0 / 152` directories across the entire `878.26 GB` `GiftEvalPretrain` corpus are `10S`** (only two `1`-series `4S` shards, `solar_power` and `wind_power`, exist—and both are included).
2. **Small & Medium Datasets ($N_d \le 2{,}000$):** Retains `100%` of shards and `100%` of series (`67` directories).
3. **Large Datasets ($N_d > 2{,}000$):** Samples across up to **`4` evenly spaced Parquet shards** (`0, N//3, 2N//3, N-1`), deterministically selects up to `2,500` native series ranked by `SHA256(f"{dataset_name}:{item_id}")`, and retains the trailing `8,192`-step window (`__recent`) plus an additional earlier historical `4,096`-step slice (`__hist`) when native length exceeds `16,384` (up to `5,000` Parquet segments per directory, with `build_subset.py` capping `end_h <= T - 8,192` so newly generated shards have zero `__hist`/`__recent` overlap).
4. **Zero-Copy Arrow Decoding:** `SubsetDatasetStore` in [src/tsfm_proto/data.py](../src/tsfm_proto/data.py) caches lightweight metadata (`_index_v1500_m96.json`, `<10 ms` load) and decodes `list<list<float32>>` columns directly via contiguous Arrow buffer offsets (`_decode_nested_float32_column`), avoiding Python object-boxing overhead.

---

## 3. Internal Validation Holdout (`[:-H_freq]` Truncation)

- **Deterministic Holdout & Frequency Horizons:** `SubsetDatasetStore` deterministically selects `1,500` validation segments (`SHA256(f"val:{dataset_name}:{item_id}")`) across `66` subset datasets in `_index_v1500_m96.json` with length $\ge \max(96, 3 H_{\text{freq}})$. Per-frequency holdout horizons `H_freq` use exact regex `(mult, code, anchor)` parsing in `horizon_for_freq`: `6` for `A/Y/YE/YS/AS/BA/BY/BYE/BYS/BAS`, `8` for `Q/QE/QS/BQ/BQE/BQS`, `13` for `W`, `18` for `M/ME/MS/BM/BME/BMS/CBME/CBMS/SM/SMS`, `30` for `D/B/C`, `48` for `H/BH` and `T/MIN/MINUTE/MINUTES`, `60` for `S/SEC/SECOND/SECONDS/L/U/N`, fallback `32` (newly built indices exclude `__hist` slices whenever primary candidates suffice, and `build_subset.py` caps `end_h <= T - 8,192` so fresh `__hist`/`__recent` slices never overlap).
- **Permanent Truncation & Point `Seasonal_Naive` Denominator:** In `train_views` **and** in `TSMixup` (`Run 5`), every held-out segment is permanently truncated to `[:, :-H_freq]` upon load so training never accesses held-out tails. Validation evaluation (`step_20000` vs. 5-snap soup) runs solely against this internal holdout (normalizing `val_nmwql` by deterministic zero-spread **Point `Seasonal_Naive`** rather than `GIFT-Eval`'s probabilistic `Seasonal_Naive`); `GIFT-Eval` test sets are never used for checkpoint selection or hyperparameter tuning.

---

## 4. Dataset Temperature Balancing ($\alpha = 1.0 \to \alpha = 0.35$)

Under raw series-count sampling (`Run 0`, $\alpha = 1.0$), large `2,500–5,000`-segment capped directories overwhelm `2–50`-segment healthcare, demographic, and macroeconomic directories. Under `SubsetDatasetStore.sampling_probs(alpha)` (`src/tsfm_proto/data.py`, `Runs 1..5`):

$$
p_d(\alpha) = \frac{N_d^{\alpha}}{\sum_{d'=1}^{93} N_{d'}^{\alpha}}, \qquad \alpha = 0.35.
$$

Moving from $\alpha = 1.0$ to $\alpha = 0.35$ compresses the max-to-min dataset exposure ratio across our `93` subset directories by **`162x`** (`2,500.0x` down to `15.5x`, normalized Shannon entropy `0.954`) while avoiding single-pass memorization of tiny datasets ($\alpha = 0.0$). Combined with group-balanced $\phi_{12}$ damped CRPS (`Run 1`), this directly drives a **`-23.6%`** `nMWQL` reduction on `Domain: Healthcare` (`0.8749 -> 0.6686`) and **`-12.6%`** on `Domain: Econ/Fin` (`0.9682 -> 0.8463`).

---

## 5. Unit-Equivariant `arcsinh` Scaling, `W=64` Local Frame & 4-Cycle Modular Lags

All stem transformations in [src/tsfm_proto/scaling.py](../src/tsfm_proto/scaling.py) are strictly causal with respect to the forecast origin $T_{\text{origin}} = T_{\text{ctx\_pad}} - 1$:

1. **Unit-Equivariant Moments (`_scale_equivariant_stats_1d`):**
   Adding a naive constant variance floor $\sqrt{\sigma^2 + 0.01^2}$ destroys resolution on climate/telemetry series measured in SI units like specific humidity ($\sim 10^{-6}\text{ kg/kg}$) or fractional CPU utilization ($\sim 6 \times 10^{-4}$). Instead, over finite observations $\mathcal{O}$ (`Runs 2..5`) or zero-filled context (`Runs 0..1`):
   $$
   \sigma_v = \begin{cases}
   \operatorname{std}(\mathcal{O}_v) & \text{if } \operatorname{std}(\mathcal{O}_v) > \max(10^{-6} |\mu_v|,\, 10^{-12}), \\
   10^{-3} |\mu_v| & \text{elif } |\mu_v| > 10^{-12}, \\
   \epsilon_\sigma & \text{otherwise (near-zero window with } \operatorname{std}(\mathcal{O}_v) \le 10^{-12} \text{ and } |\mu_v| \le 10^{-12}\text{)}.
   \end{cases}
   $$
   This guarantees scaling equivariance $\sigma(c \cdot y) = c \cdot \sigma(y)$ across positive scales whenever both rescaled series remain strictly above the `1e-12` numerical floor (`< 2e-15` in `float64` scalar moments/`arcsinh` on `NB01`'s micro-SI test; `0.0` on powers of two and `< 5e-5` across tested decimal SI rescalings in `float32`-cast `build_group_patch_features`). Target values are transformed via $\tilde{y}_{v,t} = \sinh^{-1}\!\left(\frac{y_{v,t} - \mu_v}{\sigma_v}\right)$ with quantile inverse $\hat{y}_{v,t}^{(q)} = \mu_v + \sigma_v \sinh\!\bigl(\operatorname{clip}(\hat{\tilde{y}}_{v,t}^{(q)}, -15.0, 15.0)\bigr)$ computed in `float64` in `invert_quantiles_np` before casting predictions back to `float32`.
2. **All-`NaN` Window Fallbacks (`Runs 2..5`):**
   Because cropped trailing contexts can occasionally fall entirely inside a missing span (verified in `tests/test_causality_and_masks.py`), `_prepare_chunk_job` freezes pre-crop moments *before* cropping to `128 - K_hor` patches and passes them as `fallback_mu_sigma`.
3. **Stem Channels (`C_in = 4` in `Runs 0..2`, `C_in = 17` in `Runs 3..5`):**
   - `ch 0`: `scaled_val` (zero-filled in `Runs 0..1`; causal forward-fill on observed patches and `0.0` on all-missing / horizon slots in `Runs 2..5`).
   - `ch 1`: `obs_mask` ($1.0$ strictly on truly observed context steps).
   - `ch 2`: `intra_patch_ramp` ($\operatorname{linspace}(-1, +1, 32)$, zeroed on left-pad slack).
   - `ch 3`: `rel_origin_pos` ($\operatorname{clip}((t - T_{\text{origin}})/1024, -2, 1)$), breaking value-vector invariance on 1-patch short series where `RoPE` alone does not rotate `V`.
   - `ch 4` (`Runs 3..5`): `local_scaled_val` using trailing-`64`-observed points (`obs_vals[-64:]` when $\ge 8$ observed points exist) so non-stationary level shifts remain sharp even under pre-origin blackouts.
   - `ch 5..16` (`Runs 3..5`): `6` observed-aware 4-cycle modular seasonal lags $S \in \{7, 12, 24, 48, 52, 96\}$ plus binary `lag_ok` flags computed **frequency-agnostically** across every stream (no frequency embedding is fed, letting temporal attention infer cadence directly from which observed lag + `lag_ok` pairs align with local signal periodicity). For each index $t$, we compute $k_{\text{base}}(t) = 1$ ($t \le T_{\text{origin}}$) or $\lfloor (t - T_{\text{origin}} + S - 1)/S \rfloor$ and pick the smallest cycle $m \in \{0, 1, 2, 3\}$ whose lookback $\tau = t - (k_{\text{base}}(t) + m)S$ falls inside observed context. Because $\gcd(S, 32) < 32$ for $S \in \{7, 12, 24, 48, 52\}$, feeding sample-exact modular lags fixes inter-patch seasonal phase drift across $P = 32$ boundaries.

---

## 6. Universal Modulo-32 Origin Alignment & `[1, 32, 128]` Strip Packing

In [src/tsfm_proto/packing.py](../src/tsfm_proto/packing.py):
- Every series (`T_ctx >= 1`) first trims leading joint all-`NaN` columns while guaranteeing `T_ctx >= 1` (and filters out any sub-window covariate channel with `0` finite observations before scaling), then applies `left_pad = (-T_ctx) % 32` (`0..31` zero-mask slack steps) so that `T_origin` always lands on timestep `31` of `rel_patch_idx = 0`.
- Training packs heterogeneous series onto a static `[1, R=32, L=128]` grid (`4,096` timesteps/row, `~99.6%` occupancy on typical draws).
- Horizon patch budgets follow a term-stratified distribution: `50%` short (`1..2` patches), `25%` medium (`3..12`), `25%` long (`13..16` in `Runs 0..1`, expanded to `13..32` up to $H = 1{,}024$ in `Runs 2..5`).
- In `Runs 2..5`, `30%` of training draws inject a contiguous `20..64`-step historical blackout (`50%` flush right before $T_{\text{origin}}$, `50%` interior); both `pack_strip_batch` (training) and `eval_metrics.py` (blackout stress test) compute moments strictly post-blackout (falling back safely to `(0, 1)` only if zero finite observations survive), while `_prepare_chunk_job` freezes `fallback_mu_sigma` prior to `128 - K_hor` context-length cropping.
- In `Runs 4..5`, multivariate / covariate-bearing draws (`max_g = 6` rows per group: up to `2` covariates + `4` targets) pack compactly alongside `1`-row series onto free rows within `4` columns of each other, preventing strip fragmentation while keeping high-dimensional series aligned cleanly at inference via dynamic `R_eval = max(64, max_group_rows)`.
