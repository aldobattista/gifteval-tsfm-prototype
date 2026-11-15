# 01 — Data Pipeline, Exploratory Analysis & Strip Packing

## 1. Full Corpus Skew & Stratified `2.74 GB` Subset

Raw [`theforecastingcompany/GiftEvalPretrain`](https://huggingface.co/datasets/theforecastingcompany/GiftEvalPretrain) contains `878.26 GB` across `152` Parquet directories (`3,490,516` series):
- `63` gridded climate directories (`cmip6_*`, `era5_*`) consume `90.2%` of total bytes, and `buildings_900k` holds `51.4%` of series (`1.795M`).
- Conversely, all `67` small/medium datasets (`<= 2,000` series) total just `2.60 GB` (`0.3%`).
- [scripts/build_subset.py](../scripts/build_subset.py) builds a deterministic `2.74 GB` subset across `93` dirs (`89` non-climate + `4` climate: `cmip6_1995/2010`, `era5_2005/2018`, `96,831` capped segments, `~1.02B` timesteps), keeping `100%` of small/medium datasets and `4` evenly spaced shards on huge dirs ([subset_manifest.json](../results/subset_manifest.json)).

## 2. Temperature Balancing (`alpha = 0.35`) & Leak-Free Holdout

Directory sampling weights follow a power-law temperature across the `93` subset directories:

$$
p_d \propto N_d^{\alpha}, \qquad \alpha = 1.0 \ (\text{Run 0}) \;\longrightarrow\; \alpha = 0.35 \ (\text{Runs 1--5})
$$

Lowering `alpha` from `1.0` to `0.35` compresses the max/min sampling ratio `162x` (`2,500.0x -> 15.5x`, normalized entropy `0.825 -> 0.954`) without losing dataset hierarchy. `SubsetDatasetStore` holds out the tail `[-H_freq:]` across `1,500` qualifying segments (`66` datasets) and permanently removes it from training views and synthetic `TSMixup`. Full-corpus EDA also reveals **`0 / 152` `10S` cadence directories**—directly explaining zero-shot error spikes on `GIFT-Eval` `10S` cloud tasks.

## 3. Unit-Equivariant `arcsinh` Scaling & `W = 64` + 4-Cycle Lags

Implemented in [src/tsfm_proto/scaling.py](../src/tsfm_proto/scaling.py) across observed context timesteps only (preventing `NaN` gaps or blackouts from shrinking scale):

$$
\tilde{y}_t = \sinh^{-1}\!\left(\frac{y_t - \mu_{\text{obs}}}{\sigma_{\text{obs}}}\right), \qquad \sigma_{\text{obs}} = s_{\text{obs}} \ \text{for } s_{\text{obs}} > \max\!\bigl(10^{-6}\lvert \mu_{\text{obs}} \rvert,\, 10^{-12}\bigr)
$$

- Replacing a fixed `+0.01` floor with relative variance checking (with `1e-3 * |mu_obs|` constant-signal and `eps_sigma = 0.01` all-zero fallbacks) yields `1.50e-15` `float64` invariance when rescaling `[1e-6, 1e-5]` micro-SI signals to `[1, 10]` (`< 5e-5` inside `float32`-cast patch stems).
- `Runs 3..5` append `W = 64` trailing local scaling and `4`-cycle observed seasonal lags (`S in {7, 12, 24, 48, 52, 96}` across multiples `0..3`), preserving finite anchors even across full `64`-step pre-origin blackouts.

## 4. Modulo-32 Origin Alignment & `[1, 32, 128]` Strip Packing

In [src/tsfm_proto/packing.py](../src/tsfm_proto/packing.py), every window receives `left_pad = (-T_ctx) % 32` slack steps so `T_origin` always aligns to timestep `31` of patch `rel_patch_idx == 0`. Target channels (`icov = 0`) and past covariates (`icov = 1`) pack side-by-side onto static `[1, 32, 128]` strips at `~99.5%` occupancy (`4,077 / 4,096` slots in [01_data_pipeline_walkthrough.ipynb](../notebooks/01_data_pipeline_walkthrough.ipynb)), visualized in [fig1_corpus_skew_and_sampling.png](../results/figures/fig1_corpus_skew_and_sampling.png).
