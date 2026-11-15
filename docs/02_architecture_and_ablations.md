# 02 — Architecture, Optimization & Progressive Ablations

## 1. Backbone, Attention Masks & Parameter Progression

Implemented in [src/tsfm_proto/model.py](../src/tsfm_proto/model.py#L301-L369) ([fig0_model_architecture.png](../results/figures/fig0_model_architecture.png)), `TSFMModel` is an `8`-layer Pre-RMSNorm SwiGLU Transformer (`P = 32`, `L <= 128` patches = `4,096` steps, `d_model = 384`, `6` heads, `d_ff = 1024`, `RoPE` `base = 10000` anchored at `rel_patch_idx = 0`):
- **Intra-Series Temporal Mask ([OriginAnchoredRoPE & build_temporal_mask](../src/tsfm_proto/model.py#L49-L106), [TemporalAttention](../src/tsfm_proto/model.py#L133-L170)):** Patch attention is strictly causal (`rel_patch_idx[k] <= rel_patch_idx[q]`) within each packed segment (while `PatchResStem` mixes all `32` steps densely inside each patch). In `Runs 2..5` (`exclude_blank_keys = True`), `patch_obs_any` excludes all-`NaN` patches and blanked horizon slots from keys so horizon tokens attend strictly to observed context patches (`<= 0`), with `L x L` self-diagonal fallback `eye_l` preventing `0/0` softmax `NaN`s on `100%`-missing windows.
- **Short-Series Horizon Disambiguation ([PatchResStem](../src/tsfm_proto/model.py#L109-L130)):** Stem channel `3` (`rel_origin_pos = clip((t - T_origin) / 1024, -2, 1)`) guarantees distinct `PatchResStem` embeddings and `Layer 0` residual states across `+1 .. +K_hor` horizon patches even on single-patch histories (`T_ctx <= 32`).
- **Parameters (`Runs 0..5`):** `14,816,800` (`Runs 0..2`) -> `15,136,288` (`Run 3`, `17`-channel local/lag stem) -> `16,317,344` (`Runs 4..5`, `GroupAttention` at layers `3, 7` + `cov_role_emb`). Verified in [02_model_invariants_and_overfit_check.ipynb](../notebooks/02_model_invariants_and_overfit_check.ipynb).

## 2. Analytical Step-0 Calibration & Damped Pinball Loss

`Run 3` replaces `LeftAnchoredSoftplusHead` (whose left anchor `q_0 = 0` drifts rightward as `q_k = k (ln 2 + 1e-5)` at init, giving `3%` Step-0 `80%` prediction-interval coverage `Cov80` on `q10..q90`) with `MedianAnchoredProbitHead` ([src/tsfm_proto/heads.py](../src/tsfm_proto/heads.py#L90-L118)), stepping outward (`k = 1..10`) from the zero-initialized median (`q_10 = 0.50` in `Q21`, `u_k^+/-` = upper/lower `head.fc2` logits, `u_k^+/- = 0` at Step 0 so `log(1 + e^0) / log 2 = 1`):

$$
\tilde{z}(q) = \sinh^{-1}\!\bigl(\Phi^{-1}(q)\bigr), \qquad \Delta\tilde{z}_k = \tilde{z}(q_{10 + k}) - \tilde{z}(q_{10 + k - 1}), \qquad \Delta\tilde{q}_{k}^{\pm} = \Delta\tilde{z}_k \cdot \frac{\log(1 + e^{u_k^{\pm}})}{\log 2} + 10^{-5}
$$

This guarantees strict monotonicity (`0.0` crossing rate) and drops Step-100 Continuous Ranked Probability Score (`raw_crps` / `CRPS`) **`5.3x`** (`3.546 -> 0.671`, `Cov80: 0.2135 -> 0.7264`; `~10x` at Step 0). Training minimizes `2x` average pinball CRPS over `Q21` (`0.01..0.99`; `9` `GIFT-Eval` deciles sliced at evaluation) in [compute_pinball_crps_loss](../src/tsfm_proto/loss.py#L34-L112), where each series `i`'s horizon-mean `2x` pinball loss `s_i` is outlier-damped by `phi_12(s_i)` and inversely weighted by `V_{g(i)}` (`group_target_count`) so multivariate groups never drown out univariate series (`Runs 1..5`):

$$
\phi_{\tau}(s_i) = \tau \log\!\left(1 + \frac{s_i}{\tau}\right) \;\; (\tau = 12.0), \qquad \mathcal{L} = \frac{\sum_{i} V_{g(i)}^{-1}\,\phi_{\tau}(s_i)}{\sum_{i} V_{g(i)}^{-1}}
$$

## 3. Warmup-Stable-Decay (`WSD`) Cooldown Soup, `GroupAttention` & Synthetic Coupling

- **WSD Schedule & `float64` Soup:** `1k` warmup -> `14k` constant (`lr = 3e-4`) -> `5k` cosine cooldown (`-> 1.5e-5`), averaged unweighted across steps `16k..20k` in `float64`.
- **`GroupAttention` (`Runs 4..5`):** Zero-init `out_proj.weight` guarantees exact `0.0` Step-0 univariate invariance; adapters stay frozen for the first `15%` of steps (then ramp to `0.20x` backbone LR), enforce one-way covariate-to-target attention (`~(icov_q & ~icov_k)`, where `icov` marks past-covariate rows), and block unobserved context or blanked horizon covariate keys (`obs | (is_hor & ~icov)`, with `obs = patch_obs_any` and `is_hor = (rel_patch_idx > 0)`).
- **Coupled Synthetic Drivers (`Run 5`):** Real covariates exist in just `13 / 93` dirs, leaving `Run 4` under-trained (`cov_role_emb` norm `0.130`). Mixing `15%` synthetic mix (`40%` KernelSynth + `30%` truncated `TSMixup` + `30%` coupled `2`-factor autoregressive `AR(2)`/KernelSynth lead-lag drivers, [synthetic.py](../src/tsfm_proto/synthetic.py#L110-L178)) raises `cov_role_emb` norm `~2.9x` (`0.378`) and cuts `Past Covariates` by **`-7.98%`** vs `Run 4` (with `Run 3` showing a `0.435 -> 0.432` training CRPS optimization dip between `Run 2` and `Runs 4–5` `0.440 / 0.447`; [fig2_training_and_val_curves.png](../results/figures/fig2_training_and_val_curves.png)).
