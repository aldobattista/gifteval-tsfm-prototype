# 04 — Scaling Roadmap: Closing the `0.589 -> ~0.48` Frontier Gap

Every item below maps directly to a residual failure isolated in [docs/03_evaluation_and_error_analysis.md](03_evaluation_and_error_analysis.md).

## Priority 0 — Compute, Corpus & `10S` Coverage Scale-Up

- **Scale Capacity & Data First:** Expand from `16.32M` params / `20k` steps / `2.74 GB` subset on one Apple M2 Max to `120M+` parameters (`d_model >= 768`, `12–16` layers, `150k+` steps) on multi-GPU compute across the full corpus plus `10S` sub-hourly telemetry.
- **Keep Zero-Shot Discipline:** Continue excluding `GIFT-Eval` from training and validate synthetic `10S` priors strictly out-of-distribution.

## Tier 1 — High-ROI Architectural Upgrades

1. **Pillar 1 (`3:1` Interleaved Gated Linear State-Space Recurrence + Full Attention):** Replace fixed `128`-patch (`4,096`-step = `11.4h` at `10S`) full-attention memory walls with linear-time decaying state-space memory interleaved `3:1` with pointwise-gated full-attention blocks, capturing full diurnal/multi-day cycles (`8,640+` steps/day on `10S`, alongside hourly `S=360`) across `H=720..900`.
2. **Pillar 2 (Known Future Covariates `z` + Multi-Role Attention):** Extend `cov_role_emb` and `GroupAttention` with a third modality role for known deterministic horizon drivers `z` (`calendar`, `holidays`, planned events) alongside targets `y` and past covariates `x`, fixing low-resolution calendar misses like `us_births/M`.
3. **Pillar 3 (Multi-Scale `4x8` Sub-Patch Stem):** Split each `P=32` patch into `4` fine `8`-step tokens (`360 mod 8 = 0` vs `360 mod 32 = 8`) to eliminate `8`-step phase drift on `S=360` and sharp `5T/10S` bursts while Pillar 1's `3:1` linear recurrence keeps long-context compute tractable.
4. **Pillar 4 (Origin-Blended Readout Frame):** Anchor non-stationary random walks (`m4_daily`) by interpolating between global context mean `mu_glob` (`mu_obs`) and trailing `W=64` origin mean `mu_loc` via a learned drift gate `lambda in [0, 1]`:

$$
\mu_{\text{dec}} = (1 - \lambda)\mu_{\text{glob}} + \lambda\mu_{\text{loc}}, \qquad \lambda \in [0, 1]
$$

## Tier 2 — Data Priors, Loss & Probabilistic Calibration

- **Pillar 5 (`TSMixup` + `GARCH` Burst Priors + Adaptive Lags):** Inject stochastic volatility clustering (`GARCH`, generalized autoregressive conditional heteroskedastic volatility) and frequency-adaptive seasonal lags (`S in {144, 288, 360, 720, 1440}`) so sharp telemetry spikes are not averaged away (`bizitobs_*` `80%` coverage `Cov80` on `q10..q90`: `0.290 -> ~0.80`).
- **Pillar 6 (Frequency-Stratified Loss & Tail Calibration):** Balance gradient magnitude across cadences (`10S` to annual `A`), optimize `51`-quantile dense grids with Student-t extreme-tail heads, and average multi-seed cooldown soups from later cooldown points (the low-LR tail of Warmup-Stable-Decay `WSD` cooldown, e.g. final `10–12%` of steps `0.88 * S_end .. S_end`, avoiding early-cooldown high-LR snapshots like `step_16000`).

## Non-Negotiable Engineering Guardrails

- **Preserve Invariants:** Keep origin-anchored causal masks (`build_temporal_mask`), `obs | (is_hor & ~icov)` peer firewalls, observed-only scale equivariance (`1.5e-15` `float64` scaling; `< 1e-5` end-to-end `float32` model parity), and `< 1e-6` GluonTS scoring parity under automated regression suites ([tests/](../tests/)).
- **Stage One Mechanism Per Run:** Follow the `Run 0 -> Run 5` protocol—isolate data, architecture, and objective changes cleanly with paired internal holdout and slice verification.
