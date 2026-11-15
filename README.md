# `gifteval-tsfm-prototype` — Evidence-Driven Time-Series Foundation Model Prototype

[![Tests](https://img.shields.io/badge/pytest-33%2F33%20passing-brightgreen)](tests/)
[![Pretrain](https://img.shields.io/badge/Data-GiftEvalPretrain%20%2893%20dirs%2C%202.74GB%29-blue)](https://huggingface.co/datasets/theforecastingcompany/GiftEvalPretrain)
[![Benchmark](https://img.shields.io/badge/GIFT--Eval-97%2F97%20tasks%20%28326%2C490%20windows%29-indigo)](https://huggingface.co/spaces/Salesforce/GIFT-Eval)
[![W&B](https://img.shields.io/badge/W%26B-6%20Canonical%20Runs-FFBE00?logo=weightsandbiases&logoColor=black)](https://forge.coreweave.com/wandb/aldobattista/gifteval-tsfm-prototype)

A standalone, modular PyTorch time-series foundation model (`TSFMModel`, `14.82M -> 16.32M` parameters) pretrained from scratch on a stratified `2.74 GB` multi-shard subset of [`theforecastingcompany/GiftEvalPretrain`](https://huggingface.co/datasets/theforecastingcompany/GiftEvalPretrain) and evaluated zero-shot across all **`97` official `Salesforce/GIFT-Eval` configurations (`326,490` test windows)**.

This repository evaluates **6 progressive iterations (`Runs 0..5`)** under identical compute (`20,000` steps each on a single Apple M2 Max laptop), explicitly auditing **every data, normalization, masking, objective, and architectural assumption** with disaggregated slices, `0/20/64`-step blackout stress tests, and post-processing decompositions (`Raw`, `+Clamp/Snap`, `+SymAvg+Clamp`).

---

## Executive Summary (5-Minute Read)

- **Overall Zero-Shot Progress (`97` Official `GIFT-Eval` Tasks, Geometric Mean Relative to `Seasonal_Naive`):**
  - `Run 0` (`14.82M` Baseline): **`0.6218` `nMWQL` / `0.8905` `nMASE`** (`Cov80 = 0.768`)
  - `Run 5` (`16.32M` Coupled-Synth + `GroupAttention` + Probit + Local Scale + Lags): **`0.5946` `nMWQL` / `0.8596` `nMASE`** (`Raw Neural`), **`0.5907 / 0.8556`** (`+Clamp/Snap`), and **`0.5890 / 0.8523`** (`+SymAvg+Clamp`, `Cov80 = 0.785`).
  - **Headline Improvement (`Run 5` vs. `Run 0`, `Raw`):** **`-4.37%` `nMWQL`** (`67 / 97` task wins, `69.1%`; **`-5.27%`** / `71 / 97` with `+SymAvg+Clamp`) and **`-3.47%` `nMASE`** (`67 / 97` task wins, `69.1%`; **`-4.29%`** / `72 / 97` with `+SymAvg+Clamp`), beating `Seasonal_Naive` on **`90 / 97` (`92.8%`)** tasks on `nMWQL` (`77 / 97` raw, `82 / 97` `+SymAvg+Clamp` on `nMASE`).
- **Four Core Empirical Takeaways:**
  1. **Corpus & Loss Balancing (`Run 0 -> Run 1`, `0` param delta):** Setting dataset sampling temperature $\alpha = 0.35$ + group-balanced $\phi_{12}$-damped CRPS lowers overall `nMWQL` by **`-3.11%`** (`66 / 97` task wins, `68.0%`; `nMASE -2.71%`, `63 / 97` wins), slashing **`Domain: Healthcare` by `-23.6%`** (`0.8749 -> 0.6686`) and **`Domain: Econ/Fin` by `-12.6%`** (`0.9682 -> 0.8463`).
  2. **Missingness Immunity & Single-Pass `H<=1024` (`Run 1 -> Run 2`, `0` param delta):** Observed-only unit-equivariant scaling, blank-key exclusion ($\mathcal{V}_{\text{kv}}$ with `eye_l` fallback), `30%` history blackouts, and $K_{\text{hor}} \le 32$ horizon training cut `20`- and `64`-step pre-origin blackout error by **`-18.6%`** and **`-19.7%`** and raise `Horizon > 512` `Cov80` (`0.664 -> 0.755`), while costing a transparent `+0.42%` clean `nMWQL` capacity tax.
  3. **Analytical Initialization & Dual-Scale Features (`Run 2 -> Run 3`, `+0.32M` params):** Anchoring `MedianAnchoredProbitHead` quantile levels at initialization to $\tilde{z}_q = \sinh^{-1}(\Phi^{-1}(q))$ via $\Delta\tilde{z} + 10^{-5}$ spacings eliminates Step-0 quantile collapse (`~10x` lower Step-0 / `5.3x` lower Step-100 CRPS, `Cov80: 0.2135 -> 0.7264`), while trailing-`64` local scaling and `4`-cycle modular lags recover `nMASE` (`0.8778 -> 0.8713`).
  4. **Why `GroupAttention` Needed `15%` Coupled Synthetic Priors (`Run 4 -> Run 5`):** Real `GiftEvalPretrain` includes `past_feat_dynamic_real` in only `13 / 93` subset directories (`8.5%` of segments, `91.0%` concentrated in 3 cloud VM dirs), further gated by `0.35` group-packing probability, leaving `Run 4` under-exposed to cross-series covariates. Adding `15%` synthetic mixtures (`KernelSynth` + truncated `TSMixup` + coupled lead-lag systems) nearly triples `cov_role_emb` norm (`0.130 -> 0.378`, `~2.9x`) and cuts **`Past Covariates (24 tasks)` by `-7.98%`** (`0.7052 -> 0.6489`) and **`Web/CloudOps (20 tasks)` by `-9.24%`** (`0.7850 -> 0.7125`).
- **Official `97`-Task `GIFT-Eval` Leaderboard Context (`141` Complete Baselines, [gifteval_leaderboard_comparison.csv](results/gifteval_leaderboard_comparison.csv)):**
  - **Leaderboard Position:** `Run 5 (+SymAvg+Clamp)` earns **`#15 / 28`** against the **`27` curated external baselines** (`14` external ahead, `13` external behind) and **insertion rank `#106` across `141` official baselines** (`105` ahead, `36` beaten; beating `Seasonal_Naive` on **`90 / 97` (`nMWQL`)** and **`82 / 97` (`nMASE`)**; **`0.5581` `geo_nMWQL` excluding `10S`**).
  - **Baseline & Task-Win Comparisons:** Outperforms earlier pretrained TSFMs on both aggregate `geo_nMWQL` and per-task wins—`TimesFM-1.0 (200M)` (`0.6804`, `-13.42%`, `69 / 97` wins), `Chronos_base (200M)` (`0.6521`, `-9.68%`, `68 / 97`), `Moirai_small (14M)` (`0.6504`, `-9.43%`, `71 / 97`), and `Chronos_large (710M)` (`0.6473`, `-9.01%`, `66 / 97`)—edges ahead of `iTransformer` (`0.6203`, `-5.05%`, `56 / 97`), `Moirai_base (91M)` (`0.6095`, `-3.36%`, `46 / 97`), `TFT` (`0.6047`, `-2.60%`, `54 / 97`), and `Moirai_large (311M)` (`0.5989`, `-1.65%`, `47 / 97`), and performs at near-parity with supervised `PatchTST` (`0.5870`, `+0.34%` aggregate `nMWQL`, `56 / 97` task wins).
  - **Quantified Frontier Gap:** Leaves a quantified **`~0.10` `nMWQL` gap** to frontier models (`Chronos-2` `0.4854` `#61`, `TimesFM-2.5` `0.4903` `#69`, `t0-gamma` `0.4520` `#11`), directly motivating our production scaling roadmap below.

### Scope, Compute & Time Calibration
- **Design Choice:** Scoped to `14.82M–16.32M` parameters and a `2.74 GB` stratified `93`-directory subset (`96,831` capped Parquet segments, `~2.8%` of full-corpus series / `0.31%` of bytes) so all `6` `20k`-step progressive runs fit cleanly on one laptop (`~7` hours hands-on engineering; `~8.9` hours sequential final pipeline compute [`39.3m` subset + `~7.76h` training (`7.70h` TB scalar span) + `~30m` 6-soup eval], `~10.4` hours total unattended compute including early diagnostic runs).
- **Single-Seed & Frozen-Soup Disclosure:** Each iteration trains once at `seed = 42` (multi-seed replication is explicitly scoped in [Priority 0](docs/04_with_more_time_roadmap.md)); similarly, the unweighted `5`-snapshot cooldown soup rule (`16k..20k`) was frozen upfront before seeing `GIFT-Eval` (`3–3` internal validation split across `Runs 0..5`).

---

## Architecture & Progressive Iteration Flow

```mermaid
flowchart TD
    D["theforecastingcompany/GiftEvalPretrain<br/>(152 dirs, 3.49M series, 878.26 GB)"] --> S["scripts/build_subset.py<br/>All 89 non-climate + 4 climate dirs (93 dirs)<br/>4 evenly spaced shards -> 2.74 GB, 96,831 capped segments<br/>+ eda_dataset_catalog.csv & subset_manifest.json"]
    S --> R0["Run 0: Baseline (14.82M)<br/>[1,32,128] strip canvas, mod-32 left-pad, C_in=4,<br/>alpha=1.0, flat CRPS, naive scaling, K_hor<=16"]
    R0 --> R1["Run 1: + Data & Loss Balancing (14.82M, +0)<br/>p_d ~ N_d^0.35 + 1/V_g phi_12 damped CRPS"]
    R1 --> R2["Run 2: + Missingness & H<=1024 CPM (14.82M, +0)<br/>Observed-only scaling, V_kv mask, K_hor<=32, 30% blackout"]
    R2 --> R3["Run 3: + Probit Head, W=64 Local & Lags (15.14M)<br/>asinh(Phi^-1) Probit head + W=64 local + 6 modular lags (C_in=17)"]
    R3 --> R4["Run 4: + Multivariate GroupAttention (16.32M)<br/>Zero-init out_proj (layers 3,7), cov_role_emb, firewall, hor guard"]
    R4 --> R5["Run 5: + 15% Coupled Synthetic Mix (16.32M, +0)<br/>40% KernelSynth, 30% [:-H_freq] TSMixup, 30% coupled MV"]
    R5 --> EV["scripts/eval_gifteval.py (97/97 GIFT-Eval tasks, <1e-6 GluonTS parity)<br/>Val step_20k vs 5-snap soup check -> 6 soups on test<br/>Raw vs +Clamp/Snap vs +SymAvg | Disaggregated slices & 0/20/64 blackouts"]
```

---

## Assumption-to-Evidence Traceability Matrix

| Design Choice / Assumption | First-Principles Justification | Verification & Code Evidence | Reference Connection |
| :--- | :--- | :--- | :--- |
| **1. Stratified `2.74 GB` Multi-Shard Subset (`93` dirs)** | `90.2%` of `878.26 GB` is `63` climate dirs and `51.4%` of series are `buildings_900k`; keeping `100%` of `67` small/med datasets ($\le 2\text{k}$) + `4` spaced shards preserves rare frequencies locally. | [build_subset.py](scripts/build_subset.py), [eda_dataset_catalog.csv](results/eda_dataset_catalog.csv), [Notebook 01](notebooks/01_data_pipeline_walkthrough.ipynb) | `Moirai` (`LOTSA`), `tfc-t0` |
| **2. Held-Out `[:-H_freq]` Internal Validation** | Deterministically holds out `1,500` validation tails across `66` datasets and permanently truncates `[:-H_freq]` off every held-out item in `train_views` and `TSMixup` so training never touches validation horizons ([docs/01](docs/01_data_pipeline_and_eda.md)). | [data.py](src/tsfm_proto/data.py), [test_data_and_leakage.py](tests/test_data_and_leakage.py) | `GIFT-Eval`, `TimesFM` |
| **3. $\alpha=0.35$ Balancing & $1/V_g$ Damped CRPS** | Compresses max/min sampling ratio from `2500x` to `15.5x` and bounds sensor outliers with $\phi_{12}(s) = 12 \ln(1 + s/12)$. | `Run 0 -> Run 1`, [loss.py](src/tsfm_proto/loss.py), [Notebook 01](notebooks/01_data_pipeline_walkthrough.ipynb) | `Moirai`, `Chronos` |
| **4. $P=32$, `[1, 32, 128]` Strip & Mod-32 Left-Pad** | `left_pad = (-T_ctx) % 32` locks $T_{\text{origin}}$ at index `31` of `rel_patch_idx = 0` for every $T_{\text{ctx}} \ge 1$ with **`~99.5%–99.6%`** strip occupancy. | [packing.py](src/tsfm_proto/packing.py), [test_causality_and_masks.py](tests/test_causality_and_masks.py), [Notebook 01](notebooks/01_data_pipeline_walkthrough.ipynb) | `TimesFM`, `tfc-t0` |
| **5. `rel_origin_pos` Stem Channel (`C_in >= 4`)** | `RoPE` rotates `Q` and `K` only; `rel_origin_pos` (`ch 3`) disambiguates Layer-0 residual horizon slots `+1..+K` on 1-patch short series. | [scaling.py](src/tsfm_proto/scaling.py), [Notebook 02](notebooks/02_model_invariants_and_overfit_check.ipynb) | Extends `Chronos-2` & `tfc-t0` |
| **6. Unit-Equivariant Scaling, `eye_l` $\mathcal{V}_{\text{kv}}$ & Blackouts** | Preserves SI unit-equivariance (`< 2e-15` in `float64` moments, `< 5e-5` in `float32` patch features above the `1e-12` underflow floor), applies an `eye_l` fallback on `100%`-NaN crops to prevent NaN softmax, and trains up to $H \le 1024$ with `30%` `20..64`-step blackouts. | `Run 1 -> Run 2`, [scaling.py](src/tsfm_proto/scaling.py), [model.py](src/tsfm_proto/model.py), [eval_metrics.py](src/tsfm_proto/eval_metrics.py) | `Timestral`, `Moirai`, `tfc-t0` |
| **7. `arcsinh`-Probit Head, `W=64` Local & 6 Lags** | `z_tilde = asinh(Phi^-1(q))` cuts Step-0 CRPS `~10x` (`~0.44` vs `~4.44`, `Cov80 ~ 80%`; Step-100 CRPS `5.3x`: `0.671` vs `3.546`); `W=64` local scaling & 4-cycle modular lags resist blackouts & patch phase drift. | `Run 2 -> Run 3`, [heads.py](src/tsfm_proto/heads.py), [Notebook 02](notebooks/02_model_invariants_and_overfit_check.ipynb) | `Chronos-2`, `Sundial` |
| **8. `GroupAttention` + Firewall + Observed-Peer Guard** | Zero-init `out_proj`, two-timescale LR, `has_group_peer`, `~(icov_q & ~icov_k)`, and `key_ok = obs \| (is_hor & ~icov)` exploit covariates safely without corrupting univariate rows. | `Run 3 -> Run 4`, [model.py](src/tsfm_proto/model.py), [test_causality_and_masks.py](tests/test_causality_and_masks.py) | `Chronos-2`, `TabICL` |
| **9. `15%` Coupled Synthetic Augmentation (`Run 5`)** | Overcomes `13 / 93` real `past_cov` sparsity (`8.5%` segments; `KernelSynth` + `TSMixup` + coupled lead-lag MV systems) cleanly without touching val tails. | `Run 4 -> Run 5`, [synthetic.py](src/tsfm_proto/synthetic.py), [docs/03](docs/03_evaluation_and_error_analysis.md) | `Chronos`, `Chronos-2` |
| **10. `< 1e-6` GluonTS Parity & Attribution** | Exact vectorized `GIFT-Eval` scorer verified `< 1e-6` against `gluonts`, auditing soup selection on `1,500` val series and separating `+Clamp/Snap` vs. signed `+SymAvg`. | [eval_gifteval.py](scripts/eval_gifteval.py), [results/](results/), [Notebook 03](notebooks/03_checkpoint_eval_and_forecasts.ipynb) | `GIFT-Eval`, `TimesFM` |

---

## Comparison Against Reference Repositories

| Repository | Shared Principles in `gifteval-tsfm-prototype` | Explicit Architectural & Objective Contrast |
| :--- | :--- | :--- |
| **`Moirai` (`uni2ts`)** | Any-variate cross-series attention + `LOTSA`-style corpus skewness mitigation, missingness masking | Uses causal temporal attention + fixed $P=32$ with 4-cycle modular lags instead of multi-patch projections (`8..128`) and non-parametric `Q21` Probit quantiles instead of Student-$t$ mixtures |
| **`TimesFM`** | Causal $P=32$ patching, left-padding alignment, multi-step quantile prediction | `TSFMModel` uses parallel masked horizon slots in a causal Transformer (vs. decoder-only autoregressive patching), replaces `RevIN` + zero-fill with unit-equivariant `arcsinh`, blank-key $\mathcal{V}_{\text{kv}}$ isolation, and `GroupAttention` |
| **`Chronos-2`** | Multi-channel stem symmetry breaking (`rel_origin_pos`), `GroupAttention` over targets/covariates, `arcsinh` `Q21` pinball, `KernelSynth`/`TSMixup`, reflection symmetry | Causal `P=32` `16.32M` single-stream model (vs. `Chronos-2`'s `120M` bidirectional `P=16` encoder), adding `W=64` local scaling, 4-cycle modular lags, and signed-only `+SymAvg` |
| **`Timer` / `Sundial` (`OpenLTM`)** | Unified long-context sequence modeling (`4,096` steps/strip) across frequencies | Replaces autoregressive next-patch / `TimeFlow` flow-matching heads with parallel masked horizon slots, exact $2\times$ pinball CRPS, and `20..64`-step blackout resilience |
| **`TabICL`** | Column-wise set/group attention with order-invariant peer aggregation | `[1, R, L]` swaps row/column axes relative to `TabICL` (`GroupAttention` runs across variate axis `R`); inspires Pillar 7's `50%` zero-MI decoy SCM curriculum |
| **`Timestral`** | Single-pass multi-horizon calibration up to $H \le 1024$ | Captures multi-horizon calibration via term-stratified suffix budgets ($50\% / 25\% / 25\%$) over masked horizon slots (vs. `Timestral`'s 2-stream `DHF` + Advancing Refine Frame) |
| **`tfc-t0`** | Unified `target` + `past_feat_dynamic_real` schema, `arcsinh` scaling, strip packing, 2D attention | Distills core mechanics cleanly at `~15M` scale while isolating `eye_l` blank-key safety, Probit calibration, horizon covariate guards, and 97-task attribution |

---

## Official 97-Task `GIFT-Eval` Progressive Results

Geometric-mean `nMWQL` and `nMASE` (lower is better; normalized per task by official `Seasonal_Naive`) and mean 80% coverage across all `97` tasks (`326,490` windows):

| Iteration | Params | Raw `nMWQL` | Raw `nMASE` | Raw `Cov80` | `+Clamp/Snap` `nMWQL / nMASE` | `+SymAvg+Clamp` `nMWQL / nMASE` | Key Slice Highlight |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `run0_v0_baseline` | `14.82M` | `0.6218` | `0.8905` | `0.7677` | `0.6188 / 0.8873` | `—` | End-to-end anchor |
| `run1_data_balanced` | `14.82M` | `0.6025` | `0.8664` | `0.7609` | `0.5992 / 0.8635` | `—` | **`Healthcare -23.6%`**, **`Econ/Fin -12.6%`** |
| `run2_missingness_suffix_cpm` | `14.82M` | `0.6050` | `0.8778` | **`0.7911`** | `0.6018 / 0.8755` | `—` | **`20`/`64`-blk `-18.6%` / `-19.7%`** |
| `run3_probit_localscale_lags` | `15.14M` | `0.6074` | `0.8713` | `0.7731` | `0.6038 / 0.8679` | `—` | **`~10x` Step-0 / `5.3x` Step-100 CRPS**, `nMASE` recovery |
| `run4_multivariate_group_attn` | `16.32M` | `0.6047` | `0.8678` | `0.7749` | `0.6010 / 0.8647` | `0.5998 / 0.8624` | `Low-Res A/Q/M/W: 0.7397 (-10.0%)` |
| **`run5_coupled_synth_mix`** | `16.32M` | **`0.5946`** | **`0.8596`** | `0.7789` | **`0.5907 / 0.8556`** | **`0.5890 / 0.8523`** | **`Past Cov -7.98%`**, **`CloudOps -9.24%`**, `#1` overall |

*Note on `Run 2 -> Run 3`:* While 97-task `nMWQL` ticked up `+0.38%` (`0.6050 -> 0.6074`), `Run 3` beat `Run 2` on internal validation (`0.5333 -> 0.5277`), on `97`-task `nMASE` (`0.8778 -> 0.8713`), on `55 / 97` (`56.7%`) tasks, and lowered **`91`-task non-`10S` `nMWQL` by `-1.30%`** (`0.5740 -> 0.5665`, beating `Run 1`'s `0.5706`). The net `+0.38%` overall uptick (`-1.30%` on `91` tasks offset by `+29.8%` on `6` tasks) came entirely from high-variance amplification on the `6` unseen `10S` tasks (`0 / 152` `10S` dirs in pretraining) before `Runs 4–5` regularized them (`1.7451 -> 1.3363`).

---

## Key Figures

![Corpus Skew and Temperature Sampling](results/figures/fig1_corpus_skew_and_sampling.png)
![Training and Internal Validation Curves](results/figures/fig2_training_and_val_curves.png)
![Progressive 97-Task GIFT-Eval Progression](results/figures/fig3_ablation_waterfall.png)
![Disaggregated GIFT-Eval Slices](results/figures/fig4_disaggregated_slices.png)
![Pre-Origin Blackout Stress Tests](results/figures/fig5_blackout_stress.png)

- **Top-100 Zero-Shot Forecast & Covariate Ablation Showcase Book (`Run 5 +SymAvg+Clamp`):** Browse all 50 univariate and 50 multivariate with-vs-without-covariates plates in [results/forecast_showcase_book/README.md](results/forecast_showcase_book/README.md) or download the 108-page vector compendium [Run5_Top100_Forecast_Showcase_Book.pdf](results/forecast_showcase_book/Run5_Top100_Forecast_Showcase_Book.pdf).

---

## Quickstart & Exact Reproduction

```bash
# 1. Install in editable mode + official GIFT-Eval loader for fresh clone reproduction
pip install -e ".[eval,dev]"
pip install git+https://github.com/SalesforceAIResearch/gift-eval.git

# 2. Verify all 33 invariance, holdout-truncation, causality, GroupAttention, Probit, packing, and GluonTS parity tests
# (28 passed, 5 skipped on a fresh clone prior to local subset/checkpoint build; 33/33 passed once artifacts exist)
python -m pytest -v

# 3. Optional: build the 93-directory 2.74 GB stratified pretraining subset
python scripts/build_subset.py

# 4. Train any progressive iteration (e.g. Run 5, ~79 min step compute on Apple M2 Max bfloat16 MPS)
python scripts/train.py --config configs/run5_coupled_synth_mix.yaml

# 5. Evaluate checkpoints on GIFT-Eval (or recompute summary CSVs/plots offline without binary weights)
# Note: binary *.safetensors checkpoints & data/ are gitignored; `--recompute-summaries-only` & `export_plots.py`
# work offline on committed results/ (leaderboard CSV refresh uses `data/gift_eval_official_results` when present):
python scripts/eval_gifteval.py --recompute-summaries-only
python scripts/export_plots.py
# With GIFT_EVAL + trained checkpoints present (or pass `--hf-repo-id <owner/repo>` / `--checkpoints ...`):
# huggingface-cli download Salesforce/gift-eval --repo-type dataset --local-dir data/GiftEval
# export GIFT_EVAL="$(pwd)/data/GiftEval"
python scripts/eval_gifteval.py --smoke --batch-size 64
python scripts/eval_gifteval.py --batch-size 64
python scripts/generate_showcase_book.py  # no CLI flags; builds 108p PDF (requires GIFT_EVAL + Run 5 checkpoint)

# 6. Validate checkpoint staging for Hugging Face Hub (requires local checkpoints/run*/model_soup.safetensors)
python scripts/upload_hf.py --repo-id user/gifteval-tsfm-proto --dry-run
```

---

## Repository Guide & With More Time (Priority 0 + 12-Pillar Scaling Roadmap)

- **Core Library ([src/tsfm_proto/](src/tsfm_proto/)):** `config.py`, `data.py`, `synthetic.py`, `scaling.py`, `packing.py`, `model.py`, `heads.py`, `loss.py`, `eval_metrics.py`, `hub.py`.
- **Executed Interactive Walkthroughs ([notebooks/](notebooks/)):**
  - [01_data_pipeline_walkthrough.ipynb](notebooks/01_data_pipeline_walkthrough.ipynb) — Corpus skew, `[:-H_freq]` val holdout, $\alpha=0.35$, scale equivariance, `W=64` lags & `[1,32,128]` packing.
  - [02_model_invariants_and_overfit_check.ipynb](notebooks/02_model_invariants_and_overfit_check.ipynb) — Causality, 1-patch `V` symmetry breaking, `100%`-NaN safety, Step-0 Probit fan, `GroupAttention` firewall & overfit curve.
  - [03_checkpoint_eval_and_forecasts.ipynb](notebooks/03_checkpoint_eval_and_forecasts.ipynb) — Offline-first loading, val `step_20k` vs. soup, 97-task ablation deltas & win counts, official leaderboard table, live blackout & covariate forecast gallery, residual failure inspection.
- **Top-100 Forecast Showcase Book ([results/forecast_showcase_book/](results/forecast_showcase_book/)):** [Catalog & Gallery](results/forecast_showcase_book/README.md) | [108-Page Vector PDF](results/forecast_showcase_book/Run5_Top100_Forecast_Showcase_Book.pdf) | [scripts/generate_showcase_book.py](scripts/generate_showcase_book.py).
- **Technical Deep Dives & Candid Log:** [WORKLOG.md](WORKLOG.md) | [docs/01_data_pipeline_and_eda.md](docs/01_data_pipeline_and_eda.md) | [docs/02_architecture_and_ablations.md](docs/02_architecture_and_ablations.md) | [docs/03_evaluation_and_error_analysis.md](docs/03_evaluation_and_error_analysis.md)
- **With More Time ([docs/04_with_more_time_roadmap.md](docs/04_with_more_time_roadmap.md)):** Equation-backed production blueprint pairing **Priority 0** quick wins with **12 vetted scaling pillars** and **6 empirical guardrails** across two main tracks:
  - *Architecture & Conditioning:* `3:1` hybrid gated delta-rule (`Kalman-KDA`) + `HSTU`/`XPOS` attention (`32k+` `10S` memory), future covariates $\mathbf{z}$ (`2*sigmoid` decoy gate), `4x8` `u-µP` sub-patch + dual-cadence (`P=16/8`) low-res adapters, and Björck `W=64` origin readout.
  - *Calibration, Decontamination & Training Scale:* `Spline-IQF` + `5`-anchor horizon quadrature ($H>1024$), `50%` zero-MI decoy SCM + Tobit stockouts, `3`-gate decontamination (`GIFT-Eval`, `fev-bench`, `TIME`), `FlexAttention` jagged packing, `NorMuon`/`CWD`/`OnlineEMA`, and `70/30` `CPM`.
