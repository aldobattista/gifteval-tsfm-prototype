# Engineering & Research Worklog (`gifteval-tsfm-prototype`)

Chronological log of design choices, empirical diagnostics, compute accounting, and verification across `Runs 0..5`. Full scalar telemetry is preserved in `runs/run*/events.out.tfevents.*`, [results/train_metrics_history.csv](results/train_metrics_history.csv), and the [CoreWeave W&B workspace](https://forge.coreweave.com/wandb/aldobattista/gifteval-tsfm-prototype).

## Timeline & Compute Accounting

- **Hardware:** Single Apple M2 Max laptop (`bfloat16` MPS autocast; `float32` scaling, heads, and loss; CPU `SeedSequence` packing).
- **Hands-On Engineering & Analysis:** `~7 hours` of design, corpus profiling, invariant testing, live debugging, and documentation.
- **Unattended Sequential Final Compute (`~8.9 hours`; `~10.4 hours` incl. early diagnostic runs):**
  - Stratified subset & full-corpus catalog ([scripts/build_subset.py](scripts/build_subset.py)): `39.3 min` (`2,359.7s`).
  - `6` progressive `20,000`-step runs (`seed=42`, `209–249 ms/step`): `~72–86 min/run` (`7.70 hours` TensorBoard scalar span; `~7.76 hours` training wall-clock incl. cache load, step-`20k` validation, and `float64` souping).
  - Full `97`-task (`326,490`-window) `GIFT-Eval` 6-soup evaluation + `0/20/64`-step blackouts ([scripts/eval_gifteval.py](scripts/eval_gifteval.py)): `~30 min`.

## Phase 1 — Data Profiling, Stratified Subset & Holdout Split

- Cataloged all `152` `GiftEvalPretrain` directories (`878.26 GB`, `3,490,516` series), uncovering severe Pareto skew (`63` climate dirs = `90.2%` bytes; `buildings_900k` = `51.4%` series) and `0 / 152` `10S` cadence dirs.
- Built deterministic `2.74 GB` (`93` dirs, `96,831` capped segments) subset keeping `100%` of `67` small/medium datasets.
- Permanently stripped `[-H_freq:]` across `66` qualifying datasets to create a leak-free `1,500`-series internal holdout, shared identically across `val` and `Run 5` `TSMixup`.
- Implemented modulo-`32` left padding and `[1, 32, 128]` strip packing (`99.54%` occupancy on live test batches).

## Phase 2 — Progressive Architecture, Loss & Synthetic Iterations (`Runs 0..5`)

- **`Run 0 -> 1` (`14.82M`, `0` params added):** `alpha = 0.35` balancing (`162x` ratio compression) + group-balanced `phi_12`-damped CRPS cut `nMWQL` `-3.11%` (`66/97` wins; `Healthcare -23.6%`, `Econ -12.6%`).
- **`Run 1 -> 2` (`14.82M`, `0` params added):** Observed-only `arcsinh` stats, blank-key exclusion (`eye_l`), `30%` blackouts, and `K_hor<=32` slashed `64`-step blackout error `-19.7%` and raised `H>512` `Cov80` (`0.664 -> 0.755`).
- **`Run 2 -> 3` (`+0.32M -> 15.14M`):** `MedianAnchoredProbitHead` eliminated Step-0 collapse (`Step-100 CRPS 3.546 -> 0.671`, `Cov80 0.213 -> 0.726`), while `W=64` local scale + 4-cycle lags improved `ex-10S` `-1.30%` (`91` tasks).
- **`Run 3 -> 4 -> 5` (`+1.18M -> 16.32M`):** Added zero-init `GroupAttention` (`L3, L7`) with `15%` freeze warmup; real covariates alone (`13/93` dirs, `0.130` `cov_role_emb` norm) left `Run 4` under-exposed, so `Run 5` injected `15%` coupled synthetic drivers, tripling `cov_role_emb` norm (`0.378`) and cutting `Past Covariates` `-7.98%` and `Web/CloudOps` `-9.24%`.

## Phase 3 — Official `97`-Task Evaluation & Error Audit

- Verified `< 1e-6` vectorized scoring parity against GluonTS across `18/18` smoke checks before running full `97`-task evaluation.
- Audited `step_20000` vs `5`-snap `16k..20k` `float64` soup (`3–3` split in [val_soup_vs_last.csv](results/val_soup_vs_last.csv)) and evaluated all `6` soups uniformly (`Run 5`: `0.5946` raw / `0.5890` `+SymAvg+Clamp`, `90/97` beats `Seasonal_Naive`, `#15/28` curated leaderboard / `#106/141` official).
- Isolated exact `7 / 97` residual failures above `1.0` `nMWQL` (`4x 10S`, `bizitobs_l2c/5T/long`, `m4_daily/D/short`, `us_births/M/short`), directly grounding the scaling roadmap in [docs/04_with_more_time_roadmap.md](docs/04_with_more_time_roadmap.md).

## Phase 4 — Final Verification & Cleanup

- Verified `33 / 33` pytest invariants, executed all `3` Jupyter notebooks cleanly top-to-bottom (embedding live Run 5 top univariate & multivariate showcase plots directly inside `Notebook 03` before failure diagnosis), confirmed `--recompute-summaries-only` & `upload_hf.py --dry-run`, and stripped redundant artifacts for a concise live presentation.
