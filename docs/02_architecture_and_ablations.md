# 02 — Architecture, Optimization & Progressive Ablations

## 1. Backbone, Attention Masks & Parameter Progression

Implemented in [src/tsfm_proto/model.py](../src/tsfm_proto/model.py), `TSFMModel` is an `8`-layer Pre-RMSNorm SwiGLU Transformer (`P = 32`, `L <= 128` patches = `4,096` steps, `d_model = 384`, `6` heads, `d_ff = 1024`, `RoPE` `base = 10000` anchored at `rel_patch_idx = 0`):
- **Intra-Series Temporal Mask (`build_temporal_mask`):** Patch attention is strictly causal (`rel_patch_idx[k] <= rel_patch_idx[q]`) within each packed segment (while `PatchResStem` mixes all `32` steps densely inside each patch). In `Runs 2..5` (`exclude_blank_keys = True`), `patch_obs_any` excludes all-`NaN` patches and blanked horizon slots from keys so horizon tokens attend strictly to observed context patches (`<= 0`), with self-diagonal fallback `eye_l` preventing `0/0` softmax `NaN`s on `100%`-missing windows.
- **Short-Series Horizon Disambiguation (`rel_origin_pos`):** Stem channel `3` (`clip((t - T_origin) / 1024, -2, 1)`) guarantees distinct `PatchResStem` embeddings and `Layer 0` residual states across `+1 .. +K_hor` horizon patches even on single-patch histories (`T_ctx <= 32`).
- **Parameters (`Runs 0..5`):** `14,816,800` (`Runs 0..2`) -> `15,136,288` (`Run 3`, `17`-channel local/lag stem) -> `16,317,344` (`Runs 4..5`, `GroupAttention` at layers `3, 7` + `cov_role_emb`). Verified in [02_model_invariants_and_overfit_check.ipynb](../notebooks/02_model_invariants_and_overfit_check.ipynb).

## 2. Analytical Step-0 Calibration & Damped Pinball Loss

`Run 3` replaces `LeftAnchoredSoftplusHead` (whose left anchor `q_0 = 0` drifts rightward as `q_k = k (ln 2 + 1e-5)` at init, giving `3%` Step-0 `Cov80`) with `MedianAnchoredProbitHead` ([src/tsfm_proto/heads.py](../src/tsfm_proto/heads.py)), matching `arcsinh`-normalized Gaussian quantiles outward (`k = 1..10` steps above and below zero-initialized median `q = 0.50`, `Q21` index `10`; `fc2 == 0 -> u_k == 0`, so `log(1 + e^0) / log 2 = 1`):

$$
\tilde{z}_q = \sinh^{-1}\!\bigl(\Phi^{-1}(q)\bigr), \qquad \Delta\tilde{q}_{k} = \Delta\tilde{z}_k \cdot \frac{\log(1 + e^{u_k})}{\log 2} + 10^{-5}
$$

This guarantees strict monotonicity (`0.0` crossing rate) and drops Step-100 `raw_crps` **`5.3x`** (`3.546 -> 0.671`, `Cov80: 0.2135 -> 0.7264`; `~10x` at Step 0). Training minimizes `2x` average pinball CRPS over `21` symmetric quantiles (`Q21: 0.01..0.99`; `9` `GIFT-Eval` deciles `q10..q90` sliced at evaluation) in [src/tsfm_proto/loss.py](../src/tsfm_proto/loss.py), where each series' horizon-mean `2x` pinball loss `s` is outlier-damped by `phi_12(s)` and weighted by `1 / V_g` (`group_target_count`) so multivariate groups never drown out univariate series (`Runs 1..5`):

$$
\phi_{\tau}(s) = \tau \log\!\left(1 + \frac{s}{\tau}\right), \qquad \tau = 12.0
$$

## 3. WSD Cooldown Soup, `GroupAttention` & Synthetic Coupling

- **Warmup-Stable-Decay (`WSD`) Schedule & `float64` Soup:** `1k` warmup -> `14k` constant (`lr = 3e-4`) -> `5k` cosine cooldown (`-> 1.5e-5`), averaged unweighted across steps `16k..20k` in `float64`.
- **`GroupAttention` (`Runs 4..5`):** Zero-init `out_proj.weight` guarantees exact `0.0` Step-0 univariate invariance; adapters stay frozen for the first `15%` of steps (then ramp to `0.20x` backbone LR), enforce one-way covariate-to-target attention (`~(icov_q & ~icov_k)`, where `icov` marks past-covariate rows), and block unobserved context or blanked horizon covariate keys (`obs | (is_hor & ~icov)`, with `obs = patch_obs_any` and `is_hor = (rel_patch_idx > 0)`).
- **Coupled Synthetic Drivers (`Run 5`):** Real covariates exist in just `13 / 93` dirs, leaving `Run 4` under-trained (`cov_role_emb` norm `0.130`). Mixing `15%` synthetic mix (`40%` KernelSynth + `30%` truncated `TSMixup` + `30%` coupled VAR/lead-lag drivers, [src/tsfm_proto/synthetic.py](../src/tsfm_proto/synthetic.py)) raises `cov_role_emb` norm `~2.9x` (`0.378`) and cuts `Past Covariates` by **`-7.98%`** vs `Run 4`. See [fig2_training_and_val_curves.png](../results/figures/fig2_training_and_val_curves.png).
