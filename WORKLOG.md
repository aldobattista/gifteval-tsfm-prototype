# Engineering & Research Worklog (`gifteval-tsfm-prototype`)

This worklog documents the chronological engineering decisions, live empirical diagnostics, compute accounting, and transparent workflow behind `gifteval-tsfm-prototype`. The public repository ships as a clean standalone single-commit snapshot whose committed `runs/run*/events.out.tfevents.*` event files and `results/train_metrics_history.csv` preserve the full telemetry across `Runs 0..5`.

---

## Timeline & Compute Accounting

- **Target Hardware:** Single Apple M2 Max laptop (`bfloat16` autocast on MPS with `float32` normalizations/heads/loss; CPU `SeedSequence` strip-packing workers).
- **Hands-On Engineering & Analysis Time:** `~7 hours` of interactive design, data auditing, invariance testing, live monitoring, evaluation parity verification, plotting, and documentation.
- **Unattended Overnight Compute:**
  - Subset construction + full-corpus metadata & leakage audit (`scripts/build_subset.py`): `39.3 min` (`2,359.7s`).
  - `6` progressive `20,000`-step training runs (`Runs 0..5`, `209–249 ms/step`): `~72–86 min` wall-clock per run (`7.70 hours` TensorBoard scalar event span across `Runs 0..5`; `~7.76 hours` total sequential training wall-clock including startup, cache load, step-`20,000` validation, and souping).
  - Full `97`-task (`326,490`-window) `GIFT-Eval` 6-soup evaluation + post-processing decompositions + `0/20/64`-step blackout stress suite (`scripts/eval_gifteval.py`): `~30 min` total (`< 1e-6` GluonTS parity verified on `18/18` smoke checks).
  - **Combined Sequential Final Pipeline Compute:** `~8.9 hours` (`39.3m + ~7.76h [7.70h TB span] + ~30m`; `~10.4 hours` total unattended wall-clock including initial diagnostic runs).

---

## Phase 1: Full-Corpus Profiling, Stratified Subset & Zero-Leakage Protocol

1. **Corpus Discovery (`152` Directories / `878.26 GB`):**
   - Querying `theforecastingcompany/GiftEvalPretrain` showed extreme concentration: `63` gridded `cmip6_*` and `era5_*` climate directories comprise **`90.2%`** of compressed bytes, and simulated `buildings_900k` alone contains `51.4%` of all `3.49M` series.
   - Decision: Preserve **100%** of all `89` non-climate directories (`100%` of shards and series for all **`67`** datasets with $\le 2{,}000$ series) and sample `4` representative climate directories (`cmip6_1995`, `cmip6_2010`, `era5_2005`, `era5_2018`) using `4` evenly spaced shards (`0, N//3, 2N//3, N-1`) for large datasets. Final subset size: `93` directories, `96,831` capped Parquet segments (`~2.8%` of full-corpus series / `0.31%` of bytes; `<= 2,500` native series/dir emitting `__recent` + `__hist` when $T > 16{,}384$, with `build_subset.py` capping `end_h <= T - 8,192` so fresh builds have zero `__hist`/`__recent` overlap), `1.02B` variate-timesteps, `2.738 GB` compressed on disk.
2. **Leakage Audit (`results/leakage_audit.json`):**
   - Verified `exact_name_overlap == []` across all `152` pretraining directories vs. `28` `GIFT-Eval` dataset families (`weather` [`3,220` Monash Australian daily `'D'` rows + `16` hourly `'H'` rows matching `oikolab_weather`, recorded under modal `'D'` in `leakage_audit.json`] verified distinct from 21-channel Max-Planck `jena_weather`). Notice also that **`0 / 152`** pretraining directories in the full `GiftEvalPretrain` corpus are `10S` (explaining `Run 5`'s zero-shot failure on high-cadence `10S` `bizitobs_*` rollouts).
   - Screening `11,260` 64-step z-normalized vectors yielded **`nondegenerate_z64_hash_collisions == 0` (`clean_disjoint_verified: true`)**, while isolating `4` exact hash collisions (`uber_tlc_hourly` / `bdg-2_fox` vs. `bitbrains_fast_storage/H`) to degenerate binary two-level step windows (`n_uniq == 2` in `[0, ..., c, ..., 0]`)—informing Pillar 8's non-degenerate support (`n_unique >= 8`) + $\Delta y$ first-difference screen in `docs/04_with_more_time_roadmap.md`.
3. **Permanent Internal Validation Truncation:**
   - Selected `1,500` validation segments (`>= 96` steps and `>= 3 * H_freq`) across `66` datasets (`710` plain + `790` paired items having a same-`base_id` `__recent`/`__hist` mate in the frozen subset index—across `equal_nan=True` `[-H_freq:]` float32 sequence checks against truncated `train_views`, `75` paired-mate validation tails match spans inside their mate [`61` `__hist` + `14` `__recent`], leaving **`1,425`** free of exact mate-tail sequence collisions with **`Point Seasonal_Naive`** soup `val_nmwql` `0.5662 -> 0.5195 -> 0.5324 -> 0.5286 -> 0.5289 -> 0.5301` vs. `0.5696 -> 0.5289` on all `1,500`; additionally excluding **`89`** non-constant cross-directory upstream duplicates across `london_smart_meters_with_missing` [`36`], `smart` [`29`], `largest_2017/18/19/20/21` [`22` = `2+6+4+1+9`, overlapping `largest_*` and `PEMS03/08` train series], `oikolab_weather` [`1`, row `0` matching `weather`'s `'H'` rows], and `traffic_weekly` [`1`] yields **`1,336`** segments [`1,326` with finite labels across `65` datasets; of the `11` zero-finite-label segments in the `1,500`-item holdout, `largest_2019` `3046032__recent` is already excluded inside `mate_75` while `10` zero-finite segments in `largest_2017/18/19`, `extended_web_traffic_with_missing`, and `vehicle_trips_with_missing` remain unscored inside `1,336`], scoring **`0.5631 -> 0.5168 -> 0.5301 -> 0.5256 -> 0.5267 -> 0.5281`**, with `Run 4` slightly edging out `Run 5` internally by `0.0012–0.0014` on both de-duplicated subsets; upstream datasets also carry **`56`** non-mate same-directory `n_uniq >= 4` rolling/sensor overlaps [`81` total, `38` beyond `mate_75 | cross_89`] across `taxi_30min` [`21`], `largest_2017/18/19/21` [`20`], `rideshare_with_missing` [`8`], `wiki-rolling_nips` [`6`], and `weather` [`1`]; newly generated indices exclude `__hist` slices when primary candidates suffice; `0` overlap with disjoint `GIFT-Eval`) and permanently stripped `[:-H_freq]` in both `train_views` and `TSMixup` (`Run 5`; with stateless $p_d \propto N_d^\alpha$ temperature weighting over directories in `SubsetDatasetStore.sample_train_item`). Note also that `evaluate_val_split` normalizes `val_nmwql` against zero-spread **Point `Seasonal_Naive`** (matching blackout stress rather than `GIFT-Eval`'s probabilistic denominator). Cached `_index_v1500_m96.json` + zero-copy Arrow offset decoding reduced worker boot latency from seconds to `<10 ms`.

---

## Phase 2: Architecture, Numerical Invariants & `pytest` Verification

1. **Unit-Equivariant `arcsinh` Normalization (`scaling.py`):**
   - Discovered during unit profiling that a fixed `0.01` additive variance floor distorts small SI-unit series (`era5` specific humidity $\sim 10^{-6}\text{ kg/kg}$ or `bitbrains` CPU utilization $\sim 6 \times 10^{-4}$). Replaced with relative-variation scaling `_scale_equivariant_stats_1d` (`1.50e-15` [`< 2e-15`] on `NB01`'s micro-SI test and `O((|mu|/sigma) * eps_f64)` generally in `float64` scalar moments/`arcsinh`; `0.0` on exact powers of two and `< 5e-5` across tested decimal SI rescalings [`~9e-7` in `NB01`, `1.32e-5` in `test_causality_and_masks.py`] in `float32`-cast `build_group_patch_features`, whenever both rescaled series remain strictly above the `1e-12` underflow floor) plus pre-crop `fallback_mu_sigma` for windows whose trailing context is completely missing.
2. **1-Patch Symmetry Breaking, All-`NaN` `eye_l`, & Probit Initialization (`model.py`, `heads.py`):**
   - Added `rel_origin_pos` (`C_in >= 4`) because RoPE rotates `Q`/`K` but not `V`.
   - Added automatic `eye_l` self-diagonal fallback whenever all causal keys are blanked, preventing `0/0 = NaN` in SDPA across all-`NaN` cropped windows and slack rows.
   - Anchored `MedianAnchoredProbitHead` spacings to $\sinh^{-1}(\Phi^{-1}(q)) + 10^{-5}$ so `fc2 = 0` gives analytical `80.005%` (`~80.0%`) initial `Cov80` (up to the `1e-5` monotonicity gap floor, `max |sinh(pred) - Phi^-1| = 2.54e-4`) and cuts initial real-batch training CRPS `5.3x` (`3.546 -> 0.671` at Step `100`).
3. **Pandas Frequency Normalization:**
   - Mapped legacy aliases (`T`, `H`, `M`, `Q-DEC`, `A-DEC`) cleanly via `normalize_pandas_freq` before `get_seasonality(...)`, eliminating `FutureWarning` noise while preserving exact `GluonTS` seasonal periods.

---

## Phase 3: Live Monitoring & Empirical Findings Across `Runs 0..5`

1. **`Run 0 -> Run 1` (`alpha=0.35` + $1/V_g$ Damped CRPS):**
   - Smooth monotonic validation improvement across training (`Point-SN soup val_nmwql: 0.5696 -> 0.5200`); translates directly to `-3.11%` overall on `GIFT-Eval` (`Healthcare -23.6%`, `Econ/Fin -12.6%`).
2. **`Run 1 -> Run 2` (Observed-Only + Blank-Key Masking + $K_{\text{hor}}\le 32$ + `30%` Blackouts):**
   - Validation and stress evaluation confirmed exact design intent: internal `5`-snap souping outperformed `step_20000` on `Runs 2, 3, 4` (e.g. `Run 2: 0.5333 < 0.5363`, whereas `Runs 0, 1, 5` saw `step_20000` `0.003–0.007` lower because `step_16000` sits at `LR = 2.73e-4` early in cooldown; we kept the uniform 5-snap soup rule frozen across all 6 runs in [results/val_soup_vs_last.csv](results/val_soup_vs_last.csv) to avoid per-run cherry-picking), and `20`-/`64`-step pre-origin blackout error on the `90` `GIFT-Eval` stress tasks dropped **`0.9410 -> 0.7662` (`-18.6%`)** and **`1.0287 -> 0.8265` (`-19.7%`)** in [results/gifteval_blackout_stress.csv](results/gifteval_blackout_stress.csv), with a modest clean-series capacity tax (`+0.42%` `nMWQL` / `+1.32%` `nMASE`) balanced against `Horizon > 512` `Cov80` recovery (`0.664 -> 0.755`).
3. **`Run 2 -> Run 3` (Probit + `W=64` Local Scale + 4-Cycle Modular Lags):**
   - Immediate `5.3x` Step-100 CRPS gain, improved validation `val_nmwql` (`0.5333 -> 0.5277`), recovered `GIFT-Eval` `nMASE` (`0.8778 -> 0.8713`), and lowered **`91`-task non-`10S` `nMWQL` by `-1.30%`** (`0.5740 -> 0.5665`), with `+0.38%` overall `nMWQL` explained entirely by high-variance `10S` tasks (`+29.8%`) prior to `Runs 4–5` stabilization.
4. **`Run 3 -> Run 4 -> Run 5` (Unlocking Cross-Variate & Past-Covariate `GroupAttention`):**
   - Three live-monitored design findings made cross-variate fusion work cleanly:
     - **Observed-Peer Horizon Guard:** Excluding unobserved covariate horizon slots (`key_ok = obs | (is_hor & (~icov))`) inside `GroupAttention` stops masked future covariate placeholders from diluting target representations while still allowing co-targets to share structure.
     - **Compact Subgroup Packing & Two-Timescale LR:** Capping grouped segments at `max_g = 6` within `4` columns of aligned strip cursors preserved strip occupancy (`~99.6%`) without starving single-series throughput, while holding `GroupAttention` adapters at `0` for `15%` of steps and ramping to `0.20x` LR prevented destabilizing the temporal backbone.
     - **Sparsity of Real Covariates & Synthetic Synergy:** Because only **`13 / 93`** real pretraining directories supply `past_feat_dynamic_real` (`8.5%` of series, `91.0%` concentrated in 3 cloud VM dirs), further gated by `0.35` group-packing probability, `Run 4` learned weak covariate utilization (`cov_role_emb` norm `0.130`). Mixing `15%` coupled synthetic priors in `Run 5` tripled `cov_role_emb` norm (`0.378`), cutting **`Past Covariates (24 tasks)` by `-7.98%`** and **`Web/CloudOps` by `-9.24%`**, making `Run 5` `#1` on every overall metric (`0.5946` raw / `0.5890` `+SymAvg+Clamp`, `#15/28` against the `27` curated external baselines [`row 15/35` in combined CSV including `8` own-prototype rows] and insertion rank `#106` against the `141` complete official `GIFT-Eval` baselines [`105` ahead, `36` beaten]).
   - **Signed-Only Reflection Averaging:** Restricting reflection symmetry strictly to signed series (`(fin < 0.0).any()`) avoided skew distortion on bounded non-negative series (`electricity`, `solar`, `m4_hourly`) while improving signed multivariate/residual tasks (`0.5907 -> 0.5890`).

---

## Phase 4: AI-Tooling Transparency & Verification Discipline

AI coding assistance was used to scaffold repetitive Parquet inspection, plotting scripts, notebook boilerplate, and LaTeX/Markdown formatting, **under strict quantitative verification**:
- Every architectural tensor shape and attention mask invariant was verified explicitly via deterministic unit tests (`34/34` in `tests/`), including `bfloat16` MPS execution, unmasked `NaN` safety, small-grid packing, multi-batch `loss.py` ID offsetting, multi-chunk `eval_metrics` rollouts, and `100%`-NaN stress cases.
- Every `GIFT-Eval` benchmark score was cross-checked (`< 1e-6`) directly against `gluonts.model.evaluate_forecasts(..., axis=None)` and official `Salesforce/GIFT-Eval` `Seasonal_Naive` denominators.
- Every number reported in `README.md`, `docs/01..04.md`, and `notebooks/01..03.ipynb` is read directly from committed CSV/JSON outputs and reproducible end-to-end from committed code and scripts.
