# 02 — Architecture, Progressive Iterations (`Runs 0..5`) & Literature Positioning

## 1. Backbone Overview (`src/tsfm_proto/model.py`)

`TSFMModel` is a clean `8`-layer **single-stream causal Patch-Transformer** with **parallel masked horizon query slots** (`d_model = 384`, `n_heads = 6`, `d_ff = 1024` SwiGLU, `patch_size = 32`, `max_patches = 128`) predicting `Q21` horizon quantiles in-place across masked future slots `+1 .. +K_hor`:

- **Exact Trainable Parameter Counts:**
  - `Runs 0..2` (`C_in = 4`, temporal-only): **`14,816,800`** (`14.82M`)
  - `Run 3` (`C_in = 17` local scale + modular lags): **`15,136,288`** (`15.14M`)
  - `Runs 4..5` (`GroupAttention` at layers `3, 7` + `cov_role_emb`): **`16,317,344`** (`16.32M`)

### Core Architectural Blocks
1. **`PatchResStem` (`[B, R, L, P, C_in] -> [B, R, L, D]`):**
   Flattens each `P=32` patch into `32 * C_in` features and passes them through a linear skip path plus a 2-layer `SiLU` residual MLP followed by `float32` `RMSNorm`. When `use_group_attn = True`, adds a zero-initialized `cov_role_emb` strictly on past-covariate rows (`patch_is_cov`).
2. **`OriginAnchoredRoPE` (`max_pos = 128`, $\theta = 10{,}000$):**
   Applies Rotary Position Embeddings indexed by signed relative patch distance `rel_patch_idx` $\in [-127, \dots, 0, +1, \dots, +K_{\text{hor}}]$ rather than arbitrary strip buffer column indices, ensuring complete translation invariance across packed strips.
3. **Causal Blank-Key-Excluding & All-`NaN`-Safe `TemporalAttention`:**
   Computes intra-segment temporal attention with per-head `float32` query/key `RMSNorm(self.head_dim)` and mask:
   $$
   \mathcal{M}_{b,r,q,k} = \bigl(\text{seg}_q = \text{seg}_k \ge 0\bigr) \land \bigl(\text{rel}_q \ge \text{rel}_k\bigr) \land \mathcal{V}_{\text{kv}}(k),
   $$
   where $\mathcal{V}_{\text{kv}}(k)$ excludes keys with zero observed timesteps when `exclude_blank_keys=True` (`Runs 2..5`). Whenever a row has zero valid keys (including all-`NaN` series or unused slack rows), we fall back deterministically to the identity diagonal `eye_l`:
   ```python
   base_mask = same_seg & causal & valid_kv
   time_mask = base_mask | (~base_mask.any(dim=-1, keepdim=True) & eye_l)
   ```
   preventing `0/0 = NaN` in `scaled_dot_product_attention` in both `float32` and `bfloat16`.
4. **Column-Wise `GroupAttention` (Layers `3` & `7` in `Runs 4..5`):**
   Operates across rows within the same aligned timestamp column `(b, l)` and same `patch_group_id >= 0` under four strict invariants (placing layer `7` after final `TemporalAttention` fuses co-target `V_tgt >= 2` horizon slots at readout, whereas shifting `(3, 7) -> (3, 6)` + pruning `past_cov` rows after layer `6` offers a clean `~45%` late-layer FLOPs savings at scale):
   - **Zero-Init Residual Adapter (`nn.init.zeros_(out_proj.weight)`):** Exact identity initialization at Step 0 with nonzero gradients once zero-init `head.fc2` wakes (Step 2 under unit SGD; gated after the `15%` adapter warmup freeze and linearly ramped to `0.20x` backbone LR by `25%` of steps in `scripts/train.py`).
   - **One-Way Covariate Firewall (`~(icov_q & ~icov_k)`):** Historical covariates inform target variables and other covariates, but never read from target horizon representations.
   - **Observed-Peer Horizon Guard (`key_ok = obs | (is_hor & (~icov))`):** At historical context columns, target rows only attend to genuinely observed peers; at horizon columns (`rel_patch_idx > 0`), supervised co-targets still exchange cross-series structure while masked historical covariate placeholders (`obs == False` & `icov == True`) are excluded from softmax keys so zeroed future covariate slots cannot pollute targets.
   - **Slack & Univariate Invariance (`eye_r` + `upd * has_group_peer.to(dtype)`):** Any row without an eligible peer receives an exact `0.0` residual update regardless of neighboring series on the `[1, R, L]` strip.

---

## 2. `arcsinh`-Equivariant Quantile Head & Group-Balanced Loss (`heads.py`, `loss.py`)

Both heads output strictly ordered quantiles across symmetric `Q21 = (0.01, 0.05, 0.10, ..., 0.90, 0.95, 0.99)` with identical parameter count:
- **`LeftAnchoredSoftplusHead` (`Runs 0..2`):** Anchors $q_{0.01}$ and accumulates `softplus(u) + 1e-5` gaps rightward. At zero initialization ($\text{fc}_2 = 0$), adjacent quantile gaps equal $\ln 2 + 10^{-5}$ (cumulative offsets $k(\ln 2 + 10^{-5})$ above $q_{0.01}$), giving Step-0 `arcsinh` CRPS `~4.44` (`4.4327` in `Notebook 02`, `Cov80 = 0.030`, `3.0%`).
- **`MedianAnchoredProbitHead` (`Runs 3..5`):** Anchors $q_{0.50}$ at the center and accumulates outward positive gaps parameterized as $\Delta\tilde{q}_k = \Delta\tilde{z}_k \cdot \frac{\operatorname{softplus}(u_k)}{\ln 2} + 10^{-5}$ (`= \Delta\tilde{z}_k + 10^{-5}` at $u_k = 0$), where:
  $$
  \tilde{z}_q = \sinh^{-1}\!\bigl(\Phi^{-1}(q)\bigr), \qquad \Delta\tilde{z}_k = |\tilde{z}_{q_{k\pm 1}} - \tilde{z}_{q_k}|.
  $$
  Because inverse quantile prediction (`invert_quantiles_np`) maps $\hat{y}^{(q)} = \mu + \sigma \sinh\!\bigl(\operatorname{clip}(\hat{\tilde{y}}^{(q)}, -15.0, 15.0)\bigr)$ in `float64` before casting to `float32`, $\sinh(\hat{\tilde{y}}^{(q)})$ matches $\Phi^{-1}(q)$ at initialization (`fc2 = 0`) up to the $\pm 10 \times 10^{-5}$ monotonicity gap floor (`max |sinh(pred) - Phi^-1(q)| = 2.54e-4`, analytical `N(0,1)` `Cov80 = 80.005%`, `Step-0` `arcsinh` CRPS `~0.44` [`0.4366` in `Notebook 02`], `5.3x` lower Step-100 real-batch training CRPS `3.546 -> 0.671`, `0.0` crossing rate).

### Loss Function (`compute_pinball_crps_loss`)
Computes exact $2\times$ average pinball loss over `Q21` in `float32`:
- `Run 0` (`flat_token`): Equal mean over all masked target horizon timesteps in `[1, 32, 4096]`.
- `Runs 1..5` (`per_series_damped`): First averages step CRPS within each active `series_id`, applies logarithmic spike damping $\phi_{\tau}(s) = \tau \ln(1 + s / \tau)$ ($\tau = 12.0$) to bound extreme sensor glitches, and weights each target series by $1 / V_g^{\text{target}}$ so multi-channel series do not drown out univariate ones.

---

## 3. Optimization, WSD Schedule & `float64` Weight Souping (`scripts/train.py`)

Each iteration runs `20,000` steps (`seed=42`, `bfloat16` autocast on Apple M2 Max MPS, `float32` norms/head/loss) with:
- Decoupled `AdamW` ($\eta_{\max} = 3 \times 10^{-4}$, `weight_decay = 0.01` on backbone 2D+ matrices, `0.05` on 2D `GroupAttention` adapter matrices in `Runs 4..5`, `0.0` on 1D params including `cov_role_emb`, `grad_clip = 1.0`).
- **Warmup-Stable-Decay (WSD):** `1,000`-step linear warmup, `14,000`-step plateau, `5,000`-step cosine decay to `0.05 * lr`.
- **`float64` Cooldown Soup:** Unweighted average in `float64` of snapshots at steps `(16000, 17000, 18000, 19000, 20000)` stored back as `model_soup.safetensors`. Across all `6` runs, internal validation ([results/val_soup_vs_last.csv](../results/val_soup_vs_last.csv)) splits cleanly `3–3`: souping improves held-out `val_nmwql` on **`Run 2` (`0.5363 -> 0.5333`)**, **`Run 3` (`0.5292 -> 0.5277`)**, and **`Run 4` (`0.5314 -> 0.5303`)**, whereas `step_20000` scores `0.003–0.007` lower on **`Runs 0, 1, 5`** because `step_16000` sits at `LR = 2.73e-4` early in the `15k->20k` cooldown. We kept the pre-registered uniform 5-snap soup rule frozen across all `6` models so test-time comparisons never cherry-pick checkpoints per run.

---

## 4. Progressive Iteration Ladder (`Runs 0..5`)

| Iteration | Key Upgrades Added | Params | Why It Was Added | Measured Primary Effect on `GIFT-Eval` (97 Tasks) |
| :--- | :--- | :---: | :--- | :--- |
| **`Run 0` Baseline** | `[1,32,128]` strip packer, mod-32 pad, `C_in=4` (`rel_origin_pos`), $\alpha=1.0$, `flat_token`, naive scaling, $K_{\text{hor}} \le 16$ | `14.82M` | Clean end-to-end anchor without extra complexity | `nMWQL = 0.6218`, `nMASE = 0.8905`, `Cov80 = 0.768` |
| **`Run 1` Balanced** | `alpha = 0.35` dataset temperature + $1/V_g$ $\phi_{12}$ damped CRPS | `14.82M` (`+0`) | Prevent giant datasets & extreme outliers from monopolizing gradients | **`-3.11%` `nMWQL`** (`0.6025`, CI `[0.948, 0.988]`), `Healthcare -23.6%`, `Econ/Fin -12.6%` |
| **`Run 2` Missingness + CPM** | Observed-only frozen moments, `exclude_blank_keys=True`, $K_{\text{hor}} \le 32$, `30%` history blackouts | `14.82M` (`+0`) | Single-pass $H \le 1024$ + immunity to pre-origin gaps | **`-18.6%` / `-19.7%`** on `20`/`64`-step blackouts; `H > 512` `0.6500 -> 0.6354` (costs `+0.42%` clean `nMWQL`) |
| **`Run 3` Probit + Local + Lags** | `MedianAnchoredProbitHead`, trailing-64 local scale, 6 observed-aware 4-cycle lags (`C_in=17`) | `15.14M` | Fix Step-0 quantile collapse, trend amplitude squashing & patch-phase drift | `5.3x` lower Step-100 CRPS; `nMASE` improves `0.8778 -> 0.8713`; **`excl-10S` `nMWQL` `-1.30%`** (`0.5740 -> 0.5665`), `Low-Res A/Q/M/W` `nMASE -5.3%` (offset only by `+29.8%` on unseen `10S` variance before `Runs 4–5`) |
| **`Run 4` Group Attention** | Zero-init `GroupAttention` (layers `3, 7`), `cov_role_emb`, firewall, dynamic `R_eval` | `16.32M` | Enable cross-variate and historical-covariate fusion safely | `nMWQL = 0.6047`, `nMASE = 0.8678`; `Low-Res A/Q/M/W` `0.7397` (limited by `13/93` real cov dirs, `8.5%` series) |
| **`Run 5` Coupled Synth Mix** | `85%` real + `15%` synthetic (`40%` `KernelSynth`, `30%` `TSMixup`, `30%` coupled MV systems) | `16.32M` (`+0`) | Supply rich lead-lag covariate dynamics under sparse real `past_cov` coverage | **`#1` on every overall metric:** `0.5946` raw / `0.5890` `+SymAvg+Clamp`; **`Past Cov -7.98%`**, **`CloudOps -9.24%`** |

---

## 5. Relation to Literature & Reference Repositories

- **`Moirai` (`uni2ts`):** Bidirectional any-variate flattening + cross-variate attention + `LOTSA`-style corpus skewness mitigation; our prototype keeps causal temporal attention, fixes patch size at $P=32$ with multi-scale modular lags instead of multi-patch-size projections (`8..128`), and replaces Student-$t$ mixtures with monotone non-parametric `Q21` Probit quantiles.
- **`TimesFM`:** Causal decoder-only $P=32$ patching, left-padding alignment, and multi-step `Q`-head supervision; our prototype uses parallel masked horizon slots in a single causal forward pass, adds causal forward-fill + blank-key $\mathcal{V}_{\text{kv}}$ isolation, `arcsinh` unit-equivariant normalization, and cross-variate `GroupAttention`.
- **`Chronos-2`:** Already uses `arcsinh`-space `Q21` quantile pinball loss, `GroupAttention` over targets and past covariates, `KernelSynth`/`TSMixup`, and reflection symmetry (whereas `Chronos-1` used categorical binning); directly inspires `rel_origin_pos` symmetry breaking on 1-patch stems, while our causal `16.32M` model adds `W=64` local scaling, 4-cycle modular lags, and signed-only `+SymAvg`.
- **`Timer` / `Sundial` (`OpenLTM`):** Autoregressive next-patch pretraining and `TimeFlow` continuous flow-matching heads over long contexts; we replace autoregressive point/flow heads with parallel masked horizon slots, exact $2\times$ pinball CRPS, and explicit pre-origin blackout training.
- **`TabICL`:** Column-wise set/group attention over tabular feature columns (note that `[1, R, L]` swaps row/column axes relative to `TabICL` so our `GroupAttention` acts along the variate axis `R` at each patch step `l`) and directly inspires Pillar 7's `50%` zero-MI decoy SCM curriculum.
- **`Timestral`:** Runs single-pass $H \le 1024$ inference via two-stream Dual-Horizon Forecasting (`DHF`) and an Advancing Refine Frame; we capture multi-horizon calibration in one pass via term-stratified $K_{\text{hor}}$ (`50%` short, `25%` med, `25%` long) over masked horizon slots.
- **`tfc-t0`:** Public TFC baseline using unified Parquet loading (`target` + `past_feat_dynamic_real`), `arcsinh` normalization, strip packing, 2D attention, and `+SymAvg+Clamp` inference; our prototype distills those core mechanics at `~15M` scale, isolates `obs_mask` + `V_kv` + `GroupAttention` horizon guards, and benchmarks progressive iterations end-to-end across all `97` `GIFT-Eval` tasks.
