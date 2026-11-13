# 03 — Official 97-Task `GIFT-Eval` Evaluation, Bootstrap CIs & Error Analysis

## 1. Evaluation Protocol & Exact `GluonTS` Parity

All zero-shot evaluation in [scripts/eval_gifteval.py](../scripts/eval_gifteval.py) and [src/tsfm_proto/eval_metrics.py](../src/tsfm_proto/eval_metrics.py) follows the official `Salesforce/GIFT-Eval` leaderboard definition:
- **Normalized Per-Task Metrics Before Geometric Mean:** For each of the `97` tasks (`326,490` test windows across `28` source dataset families), we evaluate all `9` deciles $\{0.1, \dots, 0.9\}$:
  $$
  \text{nMWQL}_t = \frac{\text{MWQL}_t^{\text{model}}}{\text{MWQL}_t^{\text{Seasonal\_Naive}}}, \qquad
  \text{nMASE}_t = \frac{\text{MASE}_t^{\text{model}}}{\text{MASE}_t^{\text{Seasonal\_Naive}}},
  $$
  against the official `Salesforce/GIFT-Eval` reference denominators saved in [seasonal_naive_official_reference.csv](../results/seasonal_naive_official_reference.csv), then compute the unweighted geometric mean across all `97` tasks.
- **Verified Numerical Equivalence:** `eval_gifteval.py --smoke` asserts exact `< 1e-6` agreement between our vectorized `score_task_fast` and `gluonts.model.evaluate_forecasts(..., axis=None, mask_invalid_label=True, allow_nan_forecast=False)` across edge-case short/weekly/10-second regimes (`m4_yearly/A/short`, `electricity/W/short`, `bizitobs_application/10S/short`). Across all `6` soup checkpoints x `3` smoke tasks = **`18 / 18` parity checks passed (`max diff < 1e-6`)** and **`0` non-finite predictions** occurred across all `326,490` test windows.
- **No Test-Time Selection:** Checkpoint selection (`step_20000` vs. 5-snapshot `float64` soup) was audited strictly on our `1,500`-series internal validation split ([val_soup_vs_last.csv](../results/val_soup_vs_last.csv)); only the `6` `model_soup.safetensors` checkpoints were scored on `GIFT-Eval`.

---

## 2. Internal Validation Audit (`step_20000` vs. 5-Snapshot `float64` Soup; `Point Seasonal_Naive` Denominator)

| Run Name | Params | `ms/step` | `last val_nmwql` | `soup val_nmwql` | `last val_nmase` | `soup val_nmase` | `soup val_cov80` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `run0_v0_baseline` | `14,816,800` | `208.64` | `0.566439` | `0.569612` | `0.694360` | `0.695827` | `0.754589` |
| `run1_data_balanced` | `14,816,800` | `209.85` | `0.516116` | `0.520025` | `0.628948` | `0.633225` | `0.764294` |
| `run2_missingness_suffix_cpm` | `14,816,800` | `214.45` | `0.536348` | `0.533334` | `0.651914` | `0.643432` | `0.766280` |
| `run3_probit_localscale_lags` | `15,136,288` | `220.64` | `0.529235` | `0.527693` | `0.647850` | `0.643792` | `0.759574` |
| `run4_multivariate_group_attn` | `16,317,344` | `248.57` | `0.531440` | `0.530329` | `0.627344` | **`0.624109`** | `0.763398` |
| `run5_coupled_synth_mix` | `16,317,344` | `237.94` | `0.521743` | `0.528877` | `0.633796` | `0.642129` | `0.749028` |

Note that `evaluate_val_split` ([eval_metrics.py:566-598](../src/tsfm_proto/eval_metrics.py#L566-L598)) normalizes internal `val_nmwql` against deterministic zero-spread **Point `Seasonal_Naive`** (matching Section 5's `90`-task blackout stress denominator rather than `GIFT-Eval`'s official probabilistic `Seasonal_Naive`). On initialization, `Run 3` transforms training dynamics: Step-100 training `raw_crps` drops **`5.3x`** (`3.5464 -> 0.6708`) and Step-100 `cov80` jumps from `0.2135` to `0.7264` (`0.0` crossing rate throughout). Across `Runs 0..5`, the `5`-snapshot soup wins internal validation cleanly on `Runs 2, 3, 4` (`3–3` split vs `step_20000`, which edges out `Runs 0, 1, 5` by `0.003–0.007` because `step_16000` sits at `LR = 2.73e-4` early in cooldown); we froze the uniform 5-snap soup rule upfront to prevent per-run checkpoint cherry-picking.

> **Why Internal Validation (`val_nmwql`) Ranks `Run 1` Ahead of `Run 5` (and `Run 4` Slightly Ahead of `Run 5` After De-Duplication) While `GIFT-Eval` Ranks `Run 5` `#1`:** The internal `1,500`-segment validation tails (`710` plain + `790` items with a paired same-`base_id` `__recent`/`__hist` mate in the local subset; **`1,425 / 1,500`** free of exact `[-H_freq:]` mate-tail sequence collisions in `train_views`—scoring `0.5662 -> 0.5195 -> 0.5324 -> 0.5286 -> 0.5289 -> 0.5301` `Point-SN` soup `val_nmwql`, and **`0.5631 -> 0.5168 -> 0.5301 -> 0.5256 -> 0.5267 -> 0.5281`** on the **`1,336`** segments [`1,326` with finite labels; `10` of the `11` zero-finite-label holdout segments remain in `1,336` while `largest_2019` `3046032__recent` is already excluded inside `mate_75`] remaining after excluding **`89`** non-constant cross-directory upstream duplicates across `london_smart_meters_with_missing`, `smart`, `largest_2017..2021`, `oikolab_weather`, and `traffic_weekly`; `0` overlap with external zero-shot `GIFT-Eval`) come from the *exact same `66` seen training datasets* (`train_views`), where `Run 1` optimizes clean-history memorization (`soup val_nmwql = 0.5200`, `last = 0.5161`). On both de-duplicated internal subsets (`1,425` and `1,336`), `Run 4` (`0.5289` / `0.5267`) also slightly edges out `Run 5` (`0.5301` / `0.5281`) by `0.0012–0.0014` because `Run 4` devotes `100%` of steps to real subset series whereas `Run 5` devotes `15%` to synthetic coupled priors and `30%` blackouts + $H \le 1024$ rollouts—regularizers that trade internal tail accuracy for a **`-1.67%` raw / `-1.80%` post-processed zero-shot gain from `Run 4` to `Run 5` on `GIFT-Eval`** (`0.6047 -> 0.5946` raw, `0.5998 -> 0.5890` `+SymAvg+Clamp`; **`-4.37%` raw / `-5.27%` post-processed cumulative vs. `Run 0`**, `0.6218 -> 0.5946` / `0.5890`).

---

## 3. Overall 97-Task `GIFT-Eval` Results & Post-Processing Decomposition

| Run Name | Mode | Params | `geo_nMWQL` (97) | `geo_nMASE` (97) | `Mean Cov80` |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `run0_v0_baseline` | Raw Neural | `14.82M` | `0.621814` | `0.890478` | `0.7677` |
| `run0_v0_baseline` | `+Clamp/Snap` | `14.82M` | `0.618752` | `0.887276` | `0.7722` |
| `run1_data_balanced` | Raw Neural | `14.82M` | `0.602506` | `0.866374` | `0.7609` |
| `run1_data_balanced` | `+Clamp/Snap` | `14.82M` | `0.599215` | `0.863516` | `0.7654` |
| `run2_missingness_suffix_cpm` | Raw Neural | `14.82M` | `0.605029` | `0.877849` | **`0.7911`** |
| `run2_missingness_suffix_cpm` | `+Clamp/Snap` | `14.82M` | `0.601798` | `0.875455` | **`0.7955`** |
| `run3_probit_localscale_lags` | Raw Neural | `15.14M` | `0.607354` | `0.871302` | `0.7731` |
| `run3_probit_localscale_lags` | `+Clamp/Snap` | `15.14M` | `0.603837` | `0.867949` | `0.7783` |
| `run4_multivariate_group_attn` | Raw Neural | `16.32M` | `0.604711` | `0.867795` | `0.7749` |
| `run4_multivariate_group_attn` | `+Clamp/Snap` | `16.32M` | `0.600977` | `0.864650` | `0.7801` |
| `run4_multivariate_group_attn` | `+SymAvg+Clamp` | `16.32M` | `0.599847` | `0.862356` | `0.7809` |
| **`run5_coupled_synth_mix`** | **Raw Neural** | `16.32M` | **`0.594630`** | **`0.859599`** | `0.7789` |
| **`run5_coupled_synth_mix`** | **`+Clamp/Snap`** | `16.32M` | **`0.590723`** | **`0.855619`** | `0.7841` |
| **`run5_coupled_synth_mix`** | **`+SymAvg+Clamp`** | `16.32M` | **`0.589038`** | **`0.852315`** | `0.7849` |

Key post-processing findings:
- **`Run 2 -> Run 3` Decomposition (`excl-10S` vs. `10S`):** Although 97-task aggregate `nMWQL` ticks up `+0.38%` (`0.6050 -> 0.6074`), `Run 3` actually beats `Run 2` on internal validation (`0.5333 -> 0.5277`), on `97`-task `nMASE` (`0.8778 -> 0.8713`), on `56.7%` of tasks (`55/97`), and improves **`91`-task non-`10S` `nMWQL` by `-1.30%`** (`0.5740 -> 0.5665`, beating `Run 1`'s `0.5706`). The net `+0.38%` overall uptick (`-1.30%` across `91` tasks offset by `+29.8%` across `6` tasks) comes entirely from those `6` out-of-distribution `10S` tasks (`1.3441 -> 1.7451`, because `0 / 152` pretraining dirs are `10S`) before `Runs 4–5` regularize them back down (`1.3363`).
- **`+Clamp/Snap`** (zero-overhead on cached predictions: clamping strictly non-negative observed targets at `>= 0` and rounding `< 2^23` integer series) consistently improves every iteration by `~0.5%–0.6%` (`0.5946 -> 0.5907` on `Run 5`).
- **`+SymAvg` (Sign-Reflection Averaging):** Restricting $0.5\,(F(y)_q - F(-y)_{1-q})$ strictly to signed series with `(fin < 0.0).any()` avoids distorting skewed positive/count distributions and yields clean additive gains on zero-centered anomalies/residuals (`0.5907 -> 0.5890` `nMWQL`, `0.8556 -> 0.8523` `nMASE`).

### 3b. Official 97-Task `GIFT-Eval` Leaderboard Context (`results/gifteval_leaderboard_comparison.csv`)

Evaluating a **`16.32M`-parameter scratch prototype** (`~8.9` hours sequential final pipeline compute [`39.3m` subset + `~7.76h` training (`7.70h` TensorBoard scalar span) + `~30m` 6-soup eval]; `~10.4` hours total unattended compute incl. initial diagnostic runs) on one laptop (`M2 Max`) against `141` complete official `GIFT-Eval` entries provides honest scale calibration (`35` rows in the curated CSV = `27` external baselines + `8` own-prototype rows, so `Run 5 (+SymAvg+Clamp)` at `table_rank = 15` places **`#15 / 28`** against the `27` curated external baselines [`14` ahead, `13` behind] and **insertion rank `#106` against `141` official baselines**):

| Rank (Table / Official `141`) | Model | Category | `geo_nMWQL` (97) | `geo_nMWQL` (excl-`10S`) | `geo_nMASE` (97) | Wins vs `SN` (`nMWQL / nMASE`) |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| `1 / 11` | `t0-gamma` | Pretrained TSFM (TFC) | `0.4520` | `0.4502` | `0.6608` | `96 / 95` |
| `2 / 42` | `t0-beta` | Pretrained TSFM (TFC) | `0.4738` | `0.4697` | `0.6865` | `97 / 95` |
| `4 / 61` | `Chronos-2` (`120M`) | Pretrained TSFM | `0.4854` | `0.4836` | `0.6978` | `97 / 94` |
| `6 / 69` | `TimesFM-2.5` (`200M`) | Pretrained TSFM | `0.4903` | `0.4879` | `0.7050` | `96 / 95` |
| `7 / 72` | `t0-alpha` | Pretrained TSFM (TFC) | `0.4941` | `0.4872` | `0.7240` | `96 / 90` |
| `14 / 104` | `PatchTST` | Task-Trained Deep | `0.5870` | `0.5708` | `0.8486` | `89 / 71` |
| **`15 / 106`** | **`run5_coupled_synth_mix (+SymAvg+Clamp)`** | **Our Prototype (`16.32M`)** | **`0.5890`** | **`0.5581`** | **`0.8523`** | **`90 / 82`** |
| `16 / 106` | `run5_coupled_synth_mix (Raw)` | Our Prototype (`16.32M`) | `0.5946` | `0.5637` | `0.8596` | `90 / 77` |
| `17 / 107` | `Moirai_large` (`311M`) | Pretrained TSFM | `0.5989` | `0.5646` | `0.8749` | `85 / 74` |
| `21 / 109` | `TFT` | Task-Trained Deep | `0.6047` | `0.5804` | `0.9153` | `85 / 59` |
| `24 / 111` | `Moirai_base` (`91M`) | Pretrained TSFM | `0.6095` | `0.5693` | `0.9010` | `85 / 78` |
| `25 / 114` | `iTransformer` | Task-Trained Deep | `0.6203` | `0.6074` | `0.8932` | `86 / 65` |
| `27 / 119` | `Chronos_large` (`710M`) | Pretrained TSFM | `0.6473` | `0.6160` | `0.8696` | `83 / 78` |
| `28 / 120` | `Moirai_small` (`14M`) | Pretrained TSFM | `0.6504` | `0.6119` | `0.9461` | `81 / 63` |
| `31 / 124` | `TimesFM-1.0` (`200M`) | Pretrained TSFM | `0.6804` | `0.6301` | `1.0771` | `81 / 64` |
| `32 / 135` | `Auto_Arima` / `33 / 136` `Seasonal_Naive` | Statistical | `0.9117` / `1.0000` | `0.8758` / `1.0000` | `1.0736` / `1.0000` | `55 / 61` / `0 / 0` |

Two honest takeaways stand out immediately:
1. **`Run 5` (`0.5890`, insertion rank `#106` against the `141` official baselines — `105` ahead, `36` beaten; `90/97` `nMWQL` and `82/97` `nMASE` wins vs. `Seasonal_Naive`) genuinely learns transferable cross-domain forecasting**: running the exact same `B = 10,000` (`seed = 42`) paired bootstrap against `data/gift_eval_official_results`, `Run 5 (+SymAvg+Clamp)` achieves statistically significant paired gains (`p < 0.05` on both `97`-task and `28`-family cluster CIs) over `TimesFM-1.0` (`GMR = 0.8658`, task CI `[0.822, 0.907]`, `69/97` wins), `Chronos_base` (`GMR = 0.9032`, `[0.860, 0.947]`, `68/97`), `Moirai_small` (`GMR = 0.9057`, `[0.872, 0.938]`, `71/97`), and `Chronos_large` (`GMR = 0.9099`, `[0.865, 0.955]`, `66/97`), sits ahead on aggregate point estimate of `iTransformer` (`GMR = 0.9495`, `[0.884, 1.015]`, `56/97`), `Moirai_base` (`GMR = 0.9664`, `[0.925, 1.008]`, `46/97`), `TFT` (`GMR = 0.9740`, `[0.923, 1.026]`, `54/97`), and `Moirai_large` (`GMR = 0.9835`, `[0.941, 1.027]`, `47/97`), and statistically ties supervised `PatchTST` (`GMR = 1.0034`, `[0.960, 1.050]`, `56/97` task wins; all five within paired `95%` CIs spanning `1.0`).
2. **Excluding `10S` context-starved tasks (`geo_nmwql_excl_10s = 0.5581`) cuts `0.031` off `Run 5`'s error**, cleanly isolating the `128`-patch (`4,096`-step) window limit while quantifying the remaining `~0.10` gap to frontier long-context / full-corpus models (`Chronos-2` `0.4854` `#61`, `TimesFM-2.5` `0.4903` `#69`, `t0-gamma` `0.4520` `#11`).

---

## 4. Paired Bootstrap 95% Confidence Intervals (`B = 10,000, seed = 42`)

From [gifteval_bootstrap_cis.csv](../results/gifteval_bootstrap_cis.csv) (comparing every consecutive step pair + headline baseline across `97` tasks and cluster-bootstrapping across `28` source dataset families):

| Comparison | Metric | Geometric Mean Ratio | `% Change` | 97-Task Paired 95% CI | 28-Family Cluster 95% CI | Task Win Rate |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`Run 1` vs `Run 0`** | `nMWQL` | `0.9689` | **`-3.11%`** | `[0.9477, 0.9885]` | `[0.9366, 0.9915]` | `68.0%` |
| **`Run 1` vs `Run 0`** | `nMASE` | `0.9729` | **`-2.71%`** | `[0.9554, 0.9896]` | `[0.9473, 0.9942]` | `65.0%` |
| **`Run 2` vs `Run 1`** | `nMWQL` | `1.0042` | `+0.42%` | `[0.9940, 1.0152]` | `[0.9944, 1.0138]` | `50.5%` |
| **`Run 2` vs `Run 1`** | `nMASE` | `1.0132` | `+1.32%` | `[1.0032, 1.0249]` | `[1.0044, 1.0233]` | `46.4%` |
| **`Run 3` vs `Run 2`** | `nMWQL` | `1.0038` | `+0.38%` | `[0.9831, 1.0251]` | `[0.9743, 1.0402]` | `56.7%` |
| **`Run 3` vs `Run 2`** | `nMASE` | `0.9925` | `-0.75%` | `[0.9750, 1.0084]` | `[0.9683, 1.0145]` | `58.8%` |
| **`Run 4` vs `Run 3`** | `nMWQL` | `0.9956` | `-0.44%` | `[0.9803, 1.0095]` | `[0.9735, 1.0147]` | `45.4%` |
| **`Run 4` vs `Run 3`** | `nMASE` | `0.9960` | `-0.40%` | `[0.9822, 1.0088]` | `[0.9784, 1.0126]` | `48.5%` |
| **`Run 5` vs `Run 4`** | `nMWQL` | `0.9833` | **`-1.67%`** | `[0.9676, 0.9987]` | `[0.9575, 1.0044]` | `55.7%` |
| **`Run 5` vs `Run 4`** | `nMASE` | `0.9906` | `-0.94%` | `[0.9746, 1.0071]` | `[0.9674, 1.0119]` | `66.0%` |
| **`Run 5` vs `Run 0`** | `nMWQL` | **`0.9563`** | **`-4.37%`** | **`[0.9329, 0.9792]`** | **`[0.9178, 0.9856]`** | **`69.1%`** |
| **`Run 5` vs `Run 0`** | `nMASE` | **`0.9653`** | **`-3.47%`** | **`[0.9448, 0.9857]`** | **`[0.9326, 0.9921]`** | **`69.1%`** |
| **`Run 5 +SymAvg+Clamp` vs `Raw`** | `nMWQL` | `0.9906` | `-0.94%` | `[0.9871, 0.9938]` | `[0.9855, 0.9958]` | **`87.6%`** |
| **`Run 5 +SymAvg+Clamp` vs `Raw`** | `nMASE` | `0.9915` | `-0.85%` | `[0.9879, 0.9948]` | `[0.9855, 0.9968]` | **`74.2%`** |
| **`Run 5 +SymAvg+Clamp` vs `Run 0 Raw`** | `nMWQL` | **`0.9473`** | **`-5.27%`** | **`[0.9240, 0.9701]`** | **`[0.9107, 0.9748]`** | **`73.2%`** |
| **`Run 5 +SymAvg+Clamp` vs `Run 0 Raw`** | `nMASE` | **`0.9571`** | **`-4.29%`** | **`[0.9364, 0.9778]`** | **`[0.9258, 0.9831]`** | **`74.2%`** |

Both task-level and 28-family cluster-robust 95% CIs for `Run 1 vs Run 0`, `Run 5 vs Run 0`, and `Run 5 (+SymAvg+Clamp) vs Run 5 (Raw)` lie **strictly below `1.0`** (`p < 0.05` across dataset families, avoiding horizon pseudo-replication). All runs use a single fixed training seed (`seed = 42`), so CIs measure task/family sampling uncertainty over our benchmark evaluation rather than multi-seed training variance.

---

## 5. Mechanism Attribution Across Disaggregated Slices & Blackout Stress

From [gifteval_slices_summary.csv](../results/gifteval_slices_summary.csv) and [gifteval_blackout_stress.csv](../results/gifteval_blackout_stress.csv) (`Raw Neural` soups):

1. **Data & Loss Balancing (`Run 0 -> Run 1`):** `Domain: Healthcare` drops **`0.8749 -> 0.6686` (`-23.6%`)**, `Domain: Econ/Fin` drops **`0.9682 -> 0.8463` (`-12.6%`)**, `Domain: Sales` drops `0.4794 -> 0.4352` (`-9.2%`).
2. **Missingness, Blackout & Single-Pass `H<=1024` (`Run 1 -> Run 2`):**
   - **Pre-Origin Blackout Resilience (`90` Tasks with `T_ctx >= 96`):** Normalized against the local `Point Seasonal Naive` baseline (excluding `7` ultra-short tasks lacking `96` context steps; note that Point `Seasonal_Naive` has zero interval spread so probabilistic decile `nMWQL` equals `0.6661` on clean `0`-step windows relative to official probabilistic `0.5946`, while `nMASE` sits at `0.8573` near the `97`-task `0.8596`), `20`-step blackout drops **`0.9410 -> 0.7662` (`-18.6%`)** and `64`-step (2 full `P=32` patches missing before origin!) drops **`1.0287 -> 0.8265` (`-19.7%`)**.
   - **Long Horizons & Missing Data:** `Horizon > 512 (23 tasks)` improves `0.6500 -> 0.6354` (`Cov80: 0.6636 -> 0.7552`); `Missing-Flagged (6)` improves `0.5671 -> 0.5623`.
   - **Candid Capacity Trade-off:** `Run 2` pays a modest clean-series tax (`+0.42%` `nMWQL`, `+1.32%` `nMASE`) because `30%` of training budgets experience artificial context destruction and long `17..32` horizons consume sequence width.
3. **Probit + `W=64` Local Scale + Modular Lags (`Run 2 -> Run 3`):**
   Recovers point accuracy (`nMASE: 0.8778 -> 0.8713`, `Horizon <= 512 nMASE: 0.8339 -> 0.8224`), improves `91`-task non-`10S` `nMWQL` by **`-1.30%`** (`0.5740 -> 0.5665`), lowers `Low-Res Freq (A/Q/M/W, n=15)` (`nMWQL: 0.7784 -> 0.7591 -> 0.7397` in `Run 4`; `nMASE: 0.9017 -> 0.8539 -> 0.8342`), and tightens `64`-step blackout further (`0.8265 -> 0.8237`).
4. **Why `Run 4` + `Run 5` Act Synergistically (`GroupAttention` Needs Rich Coupled Covariates):**
   Only `13 / 93` real subset directories contain `past_feat_dynamic_real` (`8.5%` of series, `91.0%` concentrated in 3 cloud VM dirs), further gated by a `35%` group-packing probability on real draws (`100%` on coupled synthetic multivariate draws in `Run 5`), giving `Run 4` thin supervision on cross-channel drivers (`cov_role_emb` L2 norm = `0.1300`). Adding `15%` synthetic mixtures including multi-scale coupled lead-lag systems in `Run 5` lifts `cov_role_emb` norm to `0.3775` and produces dramatic downstream jumps on real benchmark tasks:
   - **`Past Covariates (24 tasks)`:** `0.7052 -> 0.6489` (**`-7.98%`**)
   - **`Domain: Web/CloudOps (20 tasks)`:** `0.7850 -> 0.7125` (**`-9.24%`**, e.g., `bizitobs_l2c/H/short: 0.6907 -> 0.4929`, `bizitobs_application/10S/short: 1.1490 -> 0.9521`)
   - **`Multivariate (43 tasks)`:** `0.6170 -> 0.5938` (`0.5862` with `+SymAvg+Clamp`)
   - **90-Task Blackout (`0 / 20 / 64`):** `#1` on `5 / 6` cells—all three `Point-SN nMWQL` levels (**`0.6661 / 0.7543 / 0.8204`**) and `0 / 20`-step `Point-SN nMASE` (**`0.8573 / 0.9660`**), and `#2` on `64`-step `Point-SN nMASE` (**`1.0559`**, behind `Run 3` at `1.0538`)

---

## 6. Candid Residual Failure Analysis (`7 / 97` Tasks Above `1.0`)

`Run 5` outperforms `Seasonal_Naive` on **`90 / 97` (`92.8%`)** tasks on `nMWQL` (`77 / 97` raw, **`82 / 97` `+SymAvg+Clamp`** on `nMASE`). Notice two crucial scientific facts about the `7` remaining tasks above `1.0`: (a) across the **entire `152`-directory `878.26 GB` `GiftEvalPretrain` corpus, `0 / 152` directories are `10S`** (only two `1`-series `4S` shards exist, both already in our subset); and (b) we **deliberately stopped at `Run 5`** rather than launching an ad-hoc `Run 6` tuned specifically to `S = 360` after inspecting `GIFT-Eval` residuals—preventing post-hoc benchmark leakage while mapping every structural failure cleanly onto [docs/04_with_more_time_roadmap.md](04_with_more_time_roadmap.md):

| Task | Horizon | `Run 5` Raw `nMWQL` | `Run 5` Raw `nMASE` | Root Cause & Exact Next-Step Fix |
| :--- | :---: | :---: | :---: | :--- |
| `bizitobs_application/10S/medium` | `600` | `1.803` | `2.290` | Multi-factor `10S` bottleneck: raw test contexts span `< 24h` (`21.21–22.87h`), our `128`-patch window caps context to `581.3` min (`9.69h`), `season_length = 360` (`60` min) = `11.25` patches (`8`-step phase drift vs `P=32`), `SEASONAL_LAGS` stops at `S = 96 < 360`, and `0 / 152` `10S` dirs exist in `GiftEvalPretrain`. Solved by Pillar 1 (`32k+` memory + high-cadence lags) + Pillar 3 (`4x8` sub-patches). |
| `bizitobs_application/10S/long` | `900` | `1.847` | `2.118` | Same compounding `10S` limit: full series is `8,834` steps (`24.54h` total = `7,934` steps / `22.04h` `< 24h` raw test context + `900` steps / `2.50h` future), capped at `99` patches (`8.80h`) while `29`-patch readout damps sharp `S=360` bursts (`Cov80 = 0.290`, raw-unit `7,934`-step `win0` lag-360 Pearson `autocorr = 0.9959` on ch0 / `0.9782` on ch1, mean `0.9870`). Solved by Pillar 1 + `S in {144,360,720,1440}` + Pillar 6 quadrature rollout. |
| `bizitobs_service/10S/medium` | `600` | `1.487` | `2.535` | Same ultra-high-frequency `10S` (`S=360 > 96`, `P=32` phase drift, `0` pretrain dirs, `9.69h` capped context) + `36`-variate long rollout. |
| `bizitobs_service/10S/long` | `900` | `1.625` | `2.545` | Same `10S` multi-factor bottleneck (`8.80h` capped lookback, `S=360` lag gap, single-pass burst damping) across `36` channels. |
| `bizitobs_l2c/5T/long` | `720` | `1.050` | `0.964` | Remaining `5T` context (`105` patches = `3,360` steps = `11.67` days vs. `2.50`-day forecast) captures `<1.7` full weekly cycles across coarse `P=32` patches *(addressed by Pillar 1 `3:1 Kalman-KDA` memory extending `5T` history to `113.8` days + Pillar 3 `4x8` sub-patch stem)*. |
| `m4_daily/D/short` | `14` | `1.208` | `1.344` | Non-stationary daily finance/economic random walks where naive persistence is tough for non-autoregressive cross-domain quantile priors *(addressed by Pillar 4 origin-blended readout frame $\mu^{\text{dec}}_{v,T}$)*. |
| `us_births/M/short` | `12` | `1.052` | `1.054` | Down `-30%` from `Run 0` (`1.501 -> 1.052`), yet coarse `P=32` patching on short monthly demographic series still blurs calendar month-length & holiday spikes; solved by Pillar 2 (known calendar covariates $\mathbf{z}$) + Pillar 3 (`4x8` sub-patch stem). |
