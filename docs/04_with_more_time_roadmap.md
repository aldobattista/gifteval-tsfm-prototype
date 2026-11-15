# 04 — Scaling Roadmap: Closing the `0.589 -> ~0.48` Frontier Gap

Every item below maps directly to a residual failure isolated in [docs/03_evaluation_and_error_analysis.md](03_evaluation_and_error_analysis.md).

## Priority 0 — Compute, Corpus & `10S` Coverage Scale-Up

- **Scale Capacity & Data First:** Expand from `16.32M` params / `20k` steps / `2.74 GB` subset on one Apple M2 Max to `120M+` parameters (`d_model >= 768`, `12–16` layers, `150k+` steps) on multi-GPU compute across the full corpus plus `10S` sub-hourly telemetry.
- **Keep Zero-Shot Discipline:** Continue excluding `GIFT-Eval` from training and validate synthetic `10S` priors strictly out-of-distribution.

## Tier 1 — High-ROI Architectural Upgrades

1. **Pillar 1 (`3:1` Interleaved `Kalman-KDA` + Pointwise-Gated Full Attention):** Replace fixed `128`-patch (`4,096`-step) full-attention memory walls with linear-time decaying state-space memory interleaved with sparse full-attention blocks, capturing multi-day cycles (`S=360` on `10S`) across `H=720..900`.
2. **Pillar 2 (Known Future Covariates `z` + Multi-Role Attention):** Add dedicated `ROLE_Z` causal embeddings for known deterministic drivers (`calendar`, `holidays`, planned events), fixing low-resolution calendar misses like `us_births/M`.
3. **Pillar 3 (Multi-Scale `4x8` Sub-Patch Stem):** Split each `P=32` patch into `4` fine `8`-step tokens before attention to eliminate `8`-step phase drift on `S=360` and `5T/10S` bursts while keeping sequence cost manageable.
4. **Pillar 4 (Origin-Blended Readout Frame):** Anchor non-stationary random walks (`m4_daily`) by interpolating smoothly between global context moments and local trailing origin level:

$$
\mu_{\text{dec}} = (1 - \lambda)\,\mu_{\text{glob}} + \lambda\,\mu_{\text{loc}}, \qquad \lambda \in [0, 1]
$$

## Tier 2 — Data Priors, Loss & Probabilistic Calibration

- **Pillar 5 (`TSMixup` + GARCH Burst Priors + Adaptive Lags):** Inject stochastic volatility clustering (`GARCH`) and frequency-adaptive seasonal lags (`S in {144, 360, 720, 1440}`) so sharp telemetry spikes are not averaged away (`bizitobs_* Cov80: 0.290 -> ~0.80`).
- **Pillar 6 (Frequency-Stratified Loss & Tail Calibration):** Balance gradient magnitude across cadences (`10S`..`A`), optimize `51`-quantile dense grids with Student-t extreme-tail heads, and average multi-seed cooldown soups from later cooldown points (`0.75 * S_end .. S_end`) rather than early high-LR snapshots.

## Non-Negotiable Engineering Guardrails

- **Preserve Invariants:** Keep origin-anchored causal masks (`build_temporal_mask`), `obs | (is_hor & ~icov)` peer firewalls, observed-only scale equivariance (`< 1e-14`), and `< 1e-6` GluonTS scoring parity under automated regression suites ([tests/](../tests/)).
- **Stage One Mechanism Per Run:** Follow the `Run 0 -> Run 5` protocol—isolate data, architecture, and objective changes cleanly with paired internal holdout and slice verification.
