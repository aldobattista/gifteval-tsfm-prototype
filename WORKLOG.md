# Engineering & Research Worklog (`gifteval-tsfm-prototype`)

This worklog documents the chronological engineering decisions, live empirical diagnostics, compute accounting, and transparent workflow behind `gifteval-tsfm-prototype`. Across its multi-commit repository history, the committed `runs/run*/events.out.tfevents.*` TensorBoard event logs, [results/train_metrics_history.csv](results/train_metrics_history.csv), and the 6-run [CoreWeave W&B workspace](https://forge.coreweave.com/wandb/aldobattista/gifteval-tsfm-prototype) preserve full training telemetry across `Runs 0..5`.

---

## Timeline & Compute Accounting

- **Target Hardware:** Single Apple M2 Max laptop (`bfloat16` autocast on MPS with `float32` normalizations/heads/loss; CPU `SeedSequence` strip-packing workers).
- **Hands-On Engineering & Analysis Time:** `~7 hours` of interactive design, data auditing, invariance testing, live monitoring, evaluation parity verification, plotting, and documentation.
- **Unattended Overnight Compute:**
  - Subset construction & full-corpus metadata catalog ([scripts/build_subset.py](scripts/build_subset.py)): `39.3 min` (`2,359.7s`).
  - `6` progressive `20,000`-step training runs (`Runs 0..5`, single fixed `seed=42`, `209–249 ms/step`): `~72–86 min` wall-clock per run (`7.70 hours` TensorBoard scalar event span across `Runs 0..5`; `~7.76 hours` total sequential training wall-clock including startup, cache load, step-`20,000` validation, and souping).
  - Full `97`-task (`326,490`-window) `GIFT-Eval` 6-soup evaluation + post-processing decompositions + `0/20/64`-step blackout stress suite ([scripts/eval_gifteval.py](scripts/eval_gifteval.py)): `~30 min` total (`< 1e-6` GluonTS parity verified on `18/18` smoke checks).
  - **Combined Sequential Final Pipeline Compute:** `~8.9 hours` (`39.3m + ~7.76h [7.70h TB span] + ~30m`; `~10.4 hours` total unattended wall-clock including initial diagnostic runs).

---

## Phase 1: Full-Corpus Profiling, Stratified Subset & Internal Validation Holdout

1. **Corpus Discovery (`152` Directories / `878.26 GB`):**
   - **Extreme Concentration:** Profiling `theforecastingcompany/GiftEvalPretrain` showed `63` gridded `cmip6_*` and `era5_*` climate directories comprise **`90.2%`** of compressed bytes, while simulated `buildings_900k` alone contains `51.4%` of all `3.49M` series.
   - **Stratified Sampling Decision:** Preserved **100%** of all `89` non-climate directories (`100%` of shards and series for all **`67`** datasets with $\le 2{,}000$ series) and sampled `4` representative climate directories (`cmip6_1995`, `cmip6_2010`, `era5_2005`, `era5_2018`) using `4` evenly spaced shards (`0, N//3, 2N//3, N-1`) for large datasets.
   - **Final Subset Footprint:** `93` directories, `96,831` capped Parquet segments (`~2.8%` of full-corpus series / `0.31%` of bytes; `<= 2,500` native series/dir emitting `__recent` + `__hist` when $T > 16{,}384$, with `build_subset.py` capping `end_h <= T - 8,192` so fresh builds have zero `__hist`/`__recent` overlap), `1.02B` variate-timesteps, and `2.738 GB` compressed on disk.
   - **Zero `10S` Cadence Coverage:** Profiling all `152` directories in [results/eda_dataset_catalog.csv](results/eda_dataset_catalog.csv) revealed that **`0 / 152`** pretraining directories in the full `GiftEvalPretrain` corpus have `10S` cadence (only two `1`-series `4S` shards exist—`solar_power` and `wind_power`—and both are retained in our subset), directly explaining `Run 3`'s `+29.8%` `10S` `nMWQL` spike and `Run 5`'s residual gap on `10S` `bizitobs_*` rollouts.
2. **Internal Validation Holdout (`[:-H_freq]` Truncation):**
   - **Holdout Selection & Training Truncation:** Deterministically held out `1,500` validation segments ($\ge 96$ steps and $\ge 3 H_{\text{freq}}$) across `66` pretraining datasets and permanently stripped `[:-H_freq]` in both `train_views` and `Run 5` `TSMixup` (with stateless $p_d \propto N_d^\alpha$ directory weighting in `SubsetDatasetStore.sample_train_item`), ensuring training never accesses held-out validation horizons.
   - **Denominator & Boot Latency Disclosures:** `evaluate_val_split` normalizes `val_nmwql` against zero-spread **Point `Seasonal_Naive`** (matching blackout stress rather than `GIFT-Eval`'s probabilistic denominator). Caching `_index_v1500_m96.json` plus zero-copy Arrow offset decoding reduced worker boot latency from seconds to `<10 ms`.

---

## Phase 2: Architecture, Numerical Invariants & `pytest` Verification

1. **Unit-Equivariant `arcsinh` Normalization ([src/tsfm_proto/scaling.py](src/tsfm_proto/scaling.py)):**
   - Discovered during unit profiling that a fixed `0.01` additive variance floor distorts small SI-unit series (`era5` specific humidity $\sim 10^{-6}\text{ kg/kg}$ or `bitbrains` CPU utilization $\sim 6 \times 10^{-4}$). Replaced it with relative-variation scaling in `_scale_equivariant_stats_1d`—achieving `< 2e-15` discrepancy in `float64` scalar moments/`arcsinh` and `< 5e-5` across tested decimal SI rescalings in `float32`-cast `build_group_patch_features` whenever both rescaled series stay strictly above the `1e-12` underflow floor—paired with pre-crop `fallback_mu_sigma` for windows whose trailing context is completely missing.
2. **1-Patch Symmetry Breaking, All-`NaN` `eye_l`, & Probit Initialization ([src/tsfm_proto/model.py](src/tsfm_proto/model.py), [src/tsfm_proto/heads.py](src/tsfm_proto/heads.py)):**
   - Added `rel_origin_pos` (`C_in >= 4`) because RoPE rotates `Q`/`K` but not `V`.
   - Added automatic `eye_l` self-diagonal fallback whenever all causal keys are blanked, preventing `0/0 = NaN` in SDPA across all-`NaN` cropped windows and slack rows.
   - Anchored `MedianAnchoredProbitHead` spacings to $\Delta \sinh^{-1}(\Phi^{-1}(q)) + 10^{-5}$ so `fc2 = 0` gives analytical `80.005%` (`~80.0%`) initial `Cov80` (up to the `1e-5` monotonicity gap floor, `max |sinh(pred) - Phi^-1| = 2.54e-4`) and cuts initial real-batch training CRPS `5.3x` (`3.546 -> 0.671` at Step `100`).
3. **Pandas Frequency Normalization ([src/tsfm_proto/data.py](src/tsfm_proto/data.py)):**
   - Mapped legacy aliases (`T`, `H`, `M`, `Q-DEC`, `A-DEC`) cleanly via `normalize_pandas_freq` before `get_seasonality(...)`, eliminating `FutureWarning` noise while preserving exact `GluonTS` seasonal periods.

---

## Phase 3: Live Monitoring & Empirical Findings Across `Runs 0..5`

1. **`Run 0 -> Run 1` (`alpha=0.35` + $1/V_g$ Damped CRPS):**
   - Strictly monotonic step-wise validation gain across `Run 1` (`val_nmwql: 0.6902 -> 0.5434` over steps `2,000–18,000`; `Point-SN` soup `0.5696 -> 0.5200` vs. `Run 0`); translates directly to `-3.11%` overall on `GIFT-Eval` (`Healthcare -23.6%`, `Econ/Fin -12.6%`).
2. **`Run 1 -> Run 2` (Observed-Only + Blank-Key Masking + $K_{\text{hor}}\le 32$ + `30%` Blackouts):**
   - **Blackout Resilience vs. Clean Capacity Tradeoff:** `20`- and `64`-step pre-origin blackout error across the `90` `GIFT-Eval` stress tasks dropped **`0.9410 -> 0.7662` (`-18.6%`)** and **`1.0287 -> 0.8265` (`-19.7%`)** in [results/gifteval_blackout_stress.csv](results/gifteval_blackout_stress.csv), with a modest clean-series capacity tax (`+0.42%` `nMWQL` / `+1.32%` `nMASE`) balanced by `Horizon > 512` `Cov80` recovery (`0.664 -> 0.755`).
   - **Pre-Registered `5`-Snap Soup Rule (`3–3` Split):** Internal `5`-snapshot averaging (`16k–20k`) outperformed `step_20000` on `Runs 2, 3, 4` (e.g., `Run 2: 0.5333 < 0.5363`), whereas `Runs 0, 1, 5` saw `step_20000` `0.003–0.007` lower because `step_16000` sits at `LR = 2.73e-4` early in cooldown; we kept this uniform `5`-snap rule frozen across all `6` runs in [results/val_soup_vs_last.csv](results/val_soup_vs_last.csv) to avoid per-run cherry-picking.
3. **`Run 2 -> Run 3` (Probit + `W=64` Local Scale + 4-Cycle Modular Lags):**
   - Immediate `5.3x` Step-100 CRPS gain, improved validation `val_nmwql` (`0.5333 -> 0.5277`), recovered `GIFT-Eval` `nMASE` (`0.8778 -> 0.8713`), and lowered **`91`-task non-`10S` `nMWQL` by `-1.30%`** (`0.5740 -> 0.5665`), with `+0.38%` overall `nMWQL` explained entirely by high-variance `10S` tasks (`+29.8%`) prior to `Runs 4–5` stabilization.
4. **`Run 3 -> Run 4 -> Run 5` (Unlocking Cross-Variate & Past-Covariate `GroupAttention`):**
   - Three live-monitored design findings made cross-variate fusion work cleanly:
     - **Observed-Peer Horizon Guard:** Excluding unobserved covariate horizon slots (`key_ok = obs | (is_hor & (~icov))`) inside `GroupAttention` stops masked future covariate placeholders from diluting target representations while still allowing co-targets to share structure.
     - **Compact Subgroup Packing & Two-Timescale LR:** Capping grouped segments at `max_g = 6` within `4` columns of aligned strip cursors preserved strip occupancy (`~99.5%–99.6%`) without starving single-series throughput, while holding `GroupAttention` adapters at `0` for `15%` of steps and ramping to `0.20x` LR prevented destabilizing the temporal backbone.
     - **Sparsity of Real Covariates & Synthetic Synergy:** Because only **`13 / 93`** real pretraining directories supply `past_feat_dynamic_real` (`8.5%` of segments, `91.0%` concentrated in 3 cloud VM dirs), further gated by `0.35` group-packing probability, `Run 4` learned weak covariate utilization (`cov_role_emb` norm `0.130`). Mixing `15%` coupled synthetic priors in `Run 5` nearly tripled `cov_role_emb` norm (`0.378`), cutting **`Past Covariates (24 tasks)` by `-7.98%`** and **`Web/CloudOps` by `-9.24%`**, making `Run 5` `#1` on every overall metric (`0.5946` raw / `0.5890` `+SymAvg+Clamp`, `#15/28` against the `27` curated external baselines and insertion rank `#106` against the `141` complete official `GIFT-Eval` baselines [`105` ahead, `36` beaten]).
   - **Signed-Only Reflection Averaging:** Restricting reflection symmetry strictly to signed series (`(fin < 0.0).any()`) avoided skew distortion on bounded non-negative series (`electricity`, `solar`, `m4_hourly`) while improving signed multivariate/residual tasks (`0.5907 -> 0.5890`).

---

## Phase 4: AI-Tooling Transparency & Verification Discipline

AI coding assistance was used to scaffold repetitive Parquet inspection, plotting scripts, notebook boilerplate, and LaTeX/Markdown formatting, **under strict quantitative verification**:
- Every architectural tensor shape and attention mask invariant was verified explicitly via deterministic unit tests in [tests/](tests/) (`33/33` passing with local subset data and checkpoints; `28 passed, 5 skipped` on a fresh clone without `data/` or `checkpoints/`), including `bfloat16` MPS execution, unmasked `NaN` safety, small-grid packing, multi-batch [loss.py](src/tsfm_proto/loss.py) ID offsetting, multi-chunk [eval_metrics.py](src/tsfm_proto/eval_metrics.py) rollouts, and `100%`-`NaN` stress cases.
- Every `GIFT-Eval` benchmark score was cross-checked (`< 1e-6`) directly against `gluonts.model.evaluate_forecasts(..., axis=None)` and official `Salesforce/GIFT-Eval` `Seasonal_Naive` denominators.
- Every number reported across [README.md](README.md), [docs/01_data_pipeline_and_eda.md](docs/01_data_pipeline_and_eda.md), [docs/02_architecture_and_ablations.md](docs/02_architecture_and_ablations.md), [docs/03_evaluation_and_error_analysis.md](docs/03_evaluation_and_error_analysis.md), [docs/04_with_more_time_roadmap.md](docs/04_with_more_time_roadmap.md), [notebooks/01_data_pipeline_walkthrough.ipynb](notebooks/01_data_pipeline_walkthrough.ipynb), [notebooks/02_model_invariants_and_overfit_check.ipynb](notebooks/02_model_invariants_and_overfit_check.ipynb), [notebooks/03_checkpoint_eval_and_forecasts.ipynb](notebooks/03_checkpoint_eval_and_forecasts.ipynb), and [results/forecast_showcase_book/](results/forecast_showcase_book/) is read directly from committed CSV/JSON outputs and reproducible end-to-end from committed code and scripts.
