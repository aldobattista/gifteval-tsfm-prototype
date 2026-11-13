"""Configuration dataclass and YAML/JSON serialization for all experimental iterations."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import yaml

Q21_LEVELS: tuple[float, ...] = (
    0.01,
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
    0.99,
)

DECILE_LEVELS: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
DECILE_INDICES_IN_Q21: tuple[int, ...] = (2, 4, 6, 8, 10, 12, 14, 16, 18)


@dataclass(frozen=True)
class TSFMConfig:
    """Immutable specification of data, architecture, training, and objective knobs.

    Each progressive iteration (`run0` .. `run5`) toggles a clearly documented set
    of options while holding model capacity, optimizer, and LR schedule aligned.
    """

    run_name: str = "run0_v0_baseline"

    # Patch canvas geometry: [B=1, R=32, L=128] -> 4,096 timesteps per row.
    patch_size: int = 32
    max_patches: int = 128
    strip_rows: int = 32

    # 8-layer Transformer backbone geometry.
    d_model: int = 384
    n_layers: int = 8
    n_heads: int = 6
    d_ff: int = 1024
    rope_max_pos: int = 128
    rope_base: float = 10000.0
    norm_eps: float = 1e-6

    # Quantile grid (symmetric 21 levels with 9 GIFT-Eval deciles at indices 2,4,...,18).
    quantiles: tuple[float, ...] = Q21_LEVELS

    # Data sampling, synthetic augmentation, and loss weighting.
    sampling_alpha: float = 1.0
    loss_mode: Literal["flat_token", "per_series_damped"] = "flat_token"
    loss_damping_tau: float = 12.0
    synth_prob: float = 0.0

    # Normalization, masking, horizon span, and dropout regularizers.
    # Across all Runs 0..5, eps_sigma acts strictly as the identically-zero-window fallback
    # eps_zero (std <= 1e-12 and |mu| <= 1e-12) inside _scale_equivariant_stats_1d
    # (`Run 0..1` zero-fills NaNs first; `Run 2+` uses observed-only stats).
    scaling_mode: Literal["naive_zero_fill", "observed_frozen_origin"] = "naive_zero_fill"
    eps_sigma: float = 0.01
    exclude_blank_keys: bool = False
    max_horizon_patches: int = 16
    history_blackout_prob: float = 0.0

    # Stem feature expansion, output head, and cross-variate fusion.
    # Run 4..5 group packing & adapter warmup schedules use fixed empirical constants in
    # packing.py (`0.35` real / `1.0` coupled-synth group probability, `max_g=6`) and
    # train.py (`15%->25%` GroupAttention warmup with `0.20x` adapter LR and `wd=0.05`).
    use_local_scale_and_lags: bool = False
    head_type: Literal["left_softplus", "median_probit"] = "left_softplus"
    use_group_attn: bool = False
    group_attn_layers: tuple[int, ...] = (3, 7)

    # Training schedule (WSD: warmup -> constant plateau -> cosine cooldown soup).
    seed: int = 42
    total_steps: int = 20000
    warmup_steps: int = 1000
    cooldown_start_step: int = 15000
    soup_steps: tuple[int, ...] = (16000, 17000, 18000, 19000, 20000)
    lr: float = 3e-4
    min_lr_ratio: float = 0.05
    weight_decay: float = 1e-2
    grad_clip: float = 1.0
    batch_size: int = 1
    num_workers: int = 2
    val_every_steps: int = 2000
    val_num_series: int = 1500

    @property
    def c_in(self) -> int:
        """Stem input channels per timestep: 4 base + 13 (local scale + 6 seasonal lags/masks)."""
        return 17 if self.use_local_scale_and_lags else 4

    @property
    def head_dim(self) -> int:
        """Per-head attention dimension."""
        return self.d_model // self.n_heads

    @property
    def n_quantiles(self) -> int:
        """Number of output quantile levels."""
        return len(self.quantiles)

    def to_dict(self) -> dict:
        """Convert config to plain JSON/YAML-serializable dictionary."""
        raw = asdict(self)
        for key in ("quantiles", "group_attn_layers", "soup_steps"):
            raw[key] = list(raw[key])
        return raw

    @classmethod
    def from_dict(cls, data: dict) -> "TSFMConfig":
        """Construct ``TSFMConfig`` from a mapping, normalizing sequence fields to tuples."""
        clean = dict(data)
        for key in ("quantiles", "group_attn_layers", "soup_steps"):
            if key in clean and isinstance(clean[key], list):
                clean[key] = tuple(clean[key])
        return cls(**clean)

    def to_yaml(self, path: str | Path) -> None:
        """Write configuration as formatted YAML."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TSFMConfig":
        """Load configuration from a YAML file."""
        p = Path(path)
        return cls.from_dict(yaml.safe_load(p.read_text()))

    def to_json(self, path: str | Path) -> None:
        """Write configuration as formatted JSON."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def from_json(cls, path: str | Path) -> "TSFMConfig":
        """Load configuration from a JSON file."""
        p = Path(path)
        return cls.from_dict(json.loads(p.read_text()))
