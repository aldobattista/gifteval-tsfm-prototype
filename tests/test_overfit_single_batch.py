"""Tests for exact parameter counts and >80% single-batch CRPS reduction in 140 steps."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from tsfm_proto.config import TSFMConfig
from tsfm_proto.data import SubsetDatasetStore, make_step_rng
from tsfm_proto.loss import compute_pinball_crps_loss
from tsfm_proto.model import TSFMModel
from tsfm_proto.packing import pack_strip_batch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUBSET_DIR = PROJECT_ROOT / "data" / "gifteval_pretrain_subset"


def test_exact_parameter_counts() -> None:
    c0 = TSFMConfig.from_yaml(PROJECT_ROOT / "configs" / "run0_v0_baseline.yaml")
    c3 = TSFMConfig.from_yaml(PROJECT_ROOT / "configs" / "run3_probit_localscale_lags.yaml")
    c4 = TSFMConfig.from_yaml(PROJECT_ROOT / "configs" / "run4_multivariate_group_attn.yaml")
    assert TSFMModel(c0).count_parameters() == 14_816_800
    assert TSFMModel(c3).count_parameters() == 15_136_288
    assert TSFMModel(c4).count_parameters() == 16_317_344


@pytest.mark.parametrize(
    "config_name",
    ["run0_v0_baseline.yaml", "run4_multivariate_group_attn.yaml"],
)
def test_overfit_single_real_batch_over_80_pct(config_name: str) -> None:
    if not SUBSET_DIR.exists() or not any(SUBSET_DIR.glob("*/part-00000.parquet")):
        pytest.skip("Run scripts/build_subset.py first to materialize local Parquet subset")

    cfg = TSFMConfig.from_yaml(PROJECT_ROOT / "configs" / config_name)
    store = SubsetDatasetStore(SUBSET_DIR, val_num_series=cfg.val_num_series)
    probs = store.sampling_probs(cfg.sampling_alpha)
    rng = make_step_rng(cfg.seed, 0, 0)
    batch = pack_strip_batch(lambda r: store.sample_train_item(r, cfg, probs), rng=rng, cfg=cfg)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    model = TSFMModel(cfg).to(device)
    base_lr = 1e-3
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.0)

    x = batch.patch_inputs.to(device)
    tgt = batch.target_scaled.to(device)
    lmask = batch.loss_mask.to(device)
    rel = batch.rel_patch_idx.to(device)
    seg = batch.segment_id.to(device)
    obs = batch.patch_obs_any.to(device)
    grp = batch.patch_group_id.to(device)
    cov = batch.patch_is_cov.to(device)
    sid = batch.series_id.to(device)
    gcnt = batch.group_target_count.to(device)

    model.train()
    raw_losses: list[float] = []
    obj_losses: list[float] = []
    for step_i in range(140):
        if step_i >= 90:
            mult = 0.02 + 0.49 * (1.0 + math.cos(math.pi * (step_i - 90) / 50.0))
            for pg in optimizer.param_groups:
                pg["lr"] = base_lr * mult
        optimizer.zero_grad(set_to_none=True)
        pred = model(x, rel, seg, obs, grp, cov)
        stats = compute_pinball_crps_loss(pred, tgt, lmask, sid, gcnt, cfg)
        stats.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        raw_losses.append(float(stats.raw_crps.item()))
        obj_losses.append(float(stats.loss.item()))

    raw_drop = (raw_losses[0] - raw_losses[-1]) / max(raw_losses[0], 1e-8)
    obj_drop = (obj_losses[0] - obj_losses[-1]) / max(obj_losses[0], 1e-8)
    assert raw_losses[-1] < raw_losses[0]
    assert raw_drop > 0.80, f"{config_name} raw CRPS drop was only {raw_drop:.2%} ({raw_losses[0]:.4f} -> {raw_losses[-1]:.4f})"
    assert obj_drop > 0.70, f"{config_name} obj loss drop was only {obj_drop:.2%} ({obj_losses[0]:.4f} -> {obj_losses[-1]:.4f})"
