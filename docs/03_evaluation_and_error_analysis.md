# 03 — Evaluation Protocol, Official Leaderboard & Error Analysis

## 1. Zero-Shot Protocol & Internal Validation Split

[scripts/eval_gifteval.py](../scripts/eval_gifteval.py) evaluates all **`97` official `Salesforce/GIFT-Eval` tasks (`326,490` windows)**, verifying **`< 1e-6` exact parity** against `gluonts.ev` (`18/18` smoke checks). Before benchmarking `GIFT-Eval`, internal `1,500`-series validation (`val_soup_vs_last.csv`) yielded an honest **`3–3` split** between `step_20000` (`Runs 0, 1, 5`: `0.5664, 0.5161, 0.5217` `nMWQL`) and the `16k..20k` WSD soup (`Runs 2, 3, 4`: `0.5333, 0.5277, 0.5303` vs `0.5363, 0.5292, 0.5314`) because `step_16000` still sits at high LR (`2.73e-4`); all `6` `model_soup.safetensors` checkpoints were evaluated uniformly per protocol.

## 2. Official `97`-Task Progression & Post-Processing Decomposition

| Run | `Raw` `nMWQL / nMASE` | `+Clamp` `nMWQL` | `+SymAvg+Clamp` `nMWQL / nMASE` | `Cov80` (`Raw -> Sym`) | `64`-Blk `nMWQL` (`90` tasks) | Wins vs `Run 0` / `SN` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Run 0`** `v0_baseline` | `0.6218 / 0.8905` | `0.6188` | — | `0.768` | `0.9665` | `— / 89` |
| **`Run 1`** `data_balanced` | `0.6025 / 0.8664` | `0.5992` | — | `0.761` | `1.0287` | `66 / 89` |
| **`Run 2`** `missingness_suffix_cpm` | `0.6050 / 0.8778` | `0.6018` | — | `0.791` | `0.8265` (`-19.7%`) | `67 / 90` |
| **`Run 3`** `probit_localscale_lags` | `0.6074 / 0.8713` | `0.6038` | — | `0.773` | `0.8237` | `63 / 89` |
| **`Run 4`** `multivariate_group_attn` | `0.6047 / 0.8678` | `0.6010` | `0.5998 / 0.8624` | `0.775 -> 0.781` | `0.8339` | `57 / 90` |
| **`Run 5`** `coupled_synth_mix` | **`0.5946 / 0.8596`** | **`0.5907`** | **`0.5890 / 0.8523`** | `0.779 -> 0.785` | **`0.8204`** | **`67 / 90` (`71` Sym)** |

- **Critical Slices ([fig4_disaggregated_slices.png](../results/figures/fig4_disaggregated_slices.png)):** `Run 1` cuts `Healthcare` `-23.6%` (`0.8749 -> 0.6686`) and `Econ/Fin` `-12.6%` (`0.9682 -> 0.8463`); `Run 2` lifts `H > 512` `Cov80` (`0.664 -> 0.755`) and `20/64`-step blackout (`fig5_blackout_stress.png`: `0.7662 / 0.8265`); `Run 3` cuts `ex-10S` by `-1.30%` (`0.5665`) despite a `+29.8%` variance spike on the `6` un-pretrained `10S` tasks; `Run 5` slashes `Past Covariates (24)` `-7.98%` (`0.7052 -> 0.6489`) and `Web/CloudOps (20)` `-9.24%` (`0.7850 -> 0.7125`). See also [fig3_ablation_waterfall.png](../results/figures/fig3_ablation_waterfall.png).

## 3. Official `GIFT-Eval` Leaderboard Context (`141` Complete Baselines)

| Official Rank (/ 141) | Model | `geo_nMWQL` | `geo_nMASE` | `ex-10S nMWQL` | Wins vs `SN` (`nMWQL / nMASE`) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| `#11` | `t0-gamma` (`TFC`) | `0.4520` | `0.6608` | `0.4502` | `96 / 95` |
| `#61` | `Chronos-2 (120M)` | `0.4854` | `0.6978` | `0.4836` | `97 / 94` |
| `#69` | `TimesFM-2.5 (200M)` | `0.4903` | `0.7050` | `0.4879` | `96 / 95` |
| `#104` | `PatchTST` (Supervised) | `0.5870` | `0.8486` | `0.5708` | `89 / 71` |
| **`#106` (`#15 / 28` curated)** | **`Run 5 (+SymAvg+Clamp)` (`16.32M`)** | **`0.5890`** | **`0.8523`** | **`0.5581`** | **`90 / 82`** |
| `#107` | `Moirai_large (311M)` | `0.5989` | `0.8749` | `0.5646` | `85 / 74` |
| `#109` | `TFT` (Supervised) | `0.6047` | `0.9153` | `0.5804` | `85 / 59` |
| `#111` | `Moirai_base (91M)` | `0.6095` | `0.9010` | `0.5693` | `85 / 78` |
| `#114` | `iTransformer` (Supervised) | `0.6203` | `0.8932` | `0.6074` | `86 / 65` |
| `#119` | `Chronos_large (710M)` | `0.6473` | `0.8696` | `0.6160` | `83 / 78` |

## 4. Residual Failure Audit (`7 / 97` Tasks Above `1.0` `nMWQL`)

Normalized per-task metrics measure error directly relative to `SeasonalNaive`:

$$
\text{nMWQL}_{k} = \frac{\text{MWQL}_{k}^{\text{model}}}{\text{MWQL}_{k}^{\text{SeasonalNaive}}}, \qquad \text{Run 5} < 1.0 \ \text{on } 90 / 97 \ (92.8\%)
$$

| Task (`Run 5 Raw`) | Freq / `H` | `nMWQL` (`Run 0 -> 5`) | `Cov80` | Concrete Root Cause |
| :--- | :---: | :---: | :---: | :--- |
| `bizitobs_application` (`long`) | `10S / 900` | `1.873 -> 1.847` | `0.290` | `0/152` `10S` pretrain dirs; `S=360` drifts against `P=32` patches; burst damping |
| `bizitobs_application` (`med`) | `10S / 600` | `1.729 -> 1.803` | `0.402` | Same `10S` cadence gap + `S=360 > 96` seasonal lag cap |
| `bizitobs_service` (`long`) | `10S / 900` | `1.635 -> 1.625` | `0.711` | `10S` cadence gap + `29`-patch single-pass horizon without recurrence |
| `bizitobs_service` (`med`) | `10S / 600` | `1.541 -> 1.487` | `0.723` | `10S` cadence gap + coarse `P=32` stem blurring sharp service spikes |
| `m4_daily` (`short`) | `D / 14` | `1.317 -> 1.208` | `0.747` | Non-stationary random-walk levels where naive persistence dominates |
| `us_births` (`short`) | `M / 12` | `1.501 -> 1.052` | `0.917` | Short context + coarse `P=32` patch blurring monthly calendar effects |
| `bizitobs_l2c` (`long`) | `5T / 720` | `0.998 -> 1.050` | `0.543` | Long `23`-patch horizon with `P=32` phase drift (`<1.7` weekly cycles) |

Interactive top showcase wins and bottleneck plots run live in [03_checkpoint_eval_and_forecasts.ipynb](../notebooks/03_checkpoint_eval_and_forecasts.ipynb).
