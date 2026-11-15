"""Tests for causality, modulo-32 left-pad, all-NaN safety, dynamic R_eval, lags, and GroupAttention."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from tsfm_proto.config import TSFMConfig
from tsfm_proto.eval_metrics import EvalEntry, predict_entries
from tsfm_proto.model import TSFMModel, build_temporal_mask
from tsfm_proto.packing import compute_left_pad, trim_leading_missing
from tsfm_proto.scaling import SEASONAL_LAGS, build_group_patch_features


def test_universal_mod32_left_padding() -> None:
    for t_ctx in (1, 5, 18, 32, 42, 72, 1005):
        left_pad, n_ctx = compute_left_pad(t_ctx, patch_size=32)
        assert 0 <= left_pad < 32
        assert (left_pad + t_ctx) == n_ctx * 32
        assert n_ctx >= 1

    # Joint leading NaN trim always keeps >= 1 step even on all-NaN arrays.
    all_nan = np.full((2, 17), np.nan, dtype=np.float32)
    trimmed, _, lead = trim_leading_missing(all_nan)
    assert lead == 16 and trimmed.shape == (2, 1)


def test_strict_forecast_origin_causality() -> None:
    cfg = TSFMConfig(
        run_name="causality_test",
        use_local_scale_and_lags=True,
        head_type="median_probit",
        use_group_attn=True,
        exclude_blank_keys=True,
    )
    model = TSFMModel(cfg).eval()
    torch.manual_seed(11)
    # Non-zero head weights ensure causal perturbations produce measurable non-vacuous outputs.
    torch.nn.init.normal_(model.head.fc2.weight, std=0.1)
    torch.nn.init.normal_(model.head.fc2.bias, std=0.1)

    x = torch.randn(1, 4, 6, 32, cfg.c_in)
    rel = torch.tensor([-2, -1, 0, 1, 2, 3], dtype=torch.long).view(1, 1, 6).expand(1, 4, 6)
    seg = torch.arange(4, dtype=torch.long).view(1, 4, 1).expand(1, 4, 6)
    obs = torch.tensor([True, True, True, False, False, False]).view(1, 1, 6).expand(1, 4, 6)
    grp = torch.zeros((1, 4, 6), dtype=torch.long)
    cov = torch.zeros((1, 4, 6), dtype=torch.bool)
    cov[:, 3, :] = True

    with torch.no_grad():
        y1 = model(x, rel, seg, obs, grp, cov)
        # 1) Perturbing observed origin patch (rel=0, idx=2) must NOT leak to rel<0 (:64),
        #    while strictly changing rel=0 (64:96) and horizon >= +1 (96:).
        x_orig_pert = x.clone()
        x_orig_pert[:, :, 2, :, :] += 50.0
        y_orig_pert = model(x_orig_pert, rel, seg, obs, grp, cov)
        assert torch.max(torch.abs(y1[:, :, :64, :] - y_orig_pert[:, :, :64, :])).item() == 0.0
        assert torch.max(torch.abs(y1[:, :, 64:96, :] - y_orig_pert[:, :, 64:96, :])).item() > 1e-3
        assert torch.max(torch.abs(y1[:, :, 96:, :] - y_orig_pert[:, :, 96:, :])).item() > 1e-3

        # 2) Under exclude_blank_keys=True, horizon patches >= +1 cannot leak back to context (:96).
        x_pert = x.clone()
        x_pert[:, :, 3:, :, :] += 50.0
        y2 = model(x_pert, rel, seg, obs, grp, cov)
        assert torch.max(torch.abs(y1[:, :, :96, :] - y2[:, :, :96, :])).item() == 0.0

        # 3) With exclude_blank_keys=False, perturbing rel=+2 (idx=4) must leave rel<=+1 (:128)
        #    unchanged while changing rel>=+2 (128:), proving causal temporal masking independently.
        cfg_unmasked = TSFMConfig(
            run_name="causality_test_unmasked",
            use_local_scale_and_lags=True,
            head_type="median_probit",
            use_group_attn=True,
            exclude_blank_keys=False,
        )
        m_unmasked = TSFMModel(cfg_unmasked).eval()
        m_unmasked.load_state_dict(model.state_dict())
        y_u1 = m_unmasked(x, rel, seg, obs, grp, cov)
        x_h2_pert = x.clone()
        x_h2_pert[:, :, 4, :, :] += 50.0
        y_u2 = m_unmasked(x_h2_pert, rel, seg, obs, grp, cov)
        assert torch.max(torch.abs(y_u1[:, :, :128, :] - y_u2[:, :, :128, :])).item() == 0.0
        assert torch.max(torch.abs(y_u1[:, :, 128:, :] - y_u2[:, :, 128:, :])).item() > 1e-3


def test_blanked_key_isolation_and_all_nan_window_finiteness() -> None:
    cfg = TSFMConfig(
        run_name="nan_safety",
        exclude_blank_keys=True,
        use_local_scale_and_lags=True,
        head_type="median_probit",
        use_group_attn=False,
    )
    # 1) Blanked key isolation via temporal mask.
    rel = torch.tensor([[[-2, -1, 0, 1]]], dtype=torch.long)
    seg = torch.zeros((1, 1, 4), dtype=torch.long)
    obs = torch.tensor([[[True, False, True, False]]], dtype=torch.bool)
    tmask = build_temporal_mask(seg, rel, obs, exclude_blank_keys=True)
    # Query at rel=0 must not attend to blanked key rel=-1, and must keep rel=-2 and rel=0.
    assert bool(tmask[0, 0, 2, 0]) is True
    assert bool(tmask[0, 0, 2, 1]) is False
    assert bool(tmask[0, 0, 2, 2]) is True

    # Query with 0 observed keys falls back to identity self-key so SDPA never hits 0/0.
    obs_none = torch.zeros((1, 1, 4), dtype=torch.bool)
    tmask_none = build_temporal_mask(seg, rel, obs_none, exclude_blank_keys=True)
    assert bool(tmask_none[0, 0, 0, 0]) is True
    assert bool(tmask_none[0, 0, 0, 1:].any()) is False

    # 2) All-NaN input entries remain strictly finite in both float32 and bfloat16 inference.
    model = TSFMModel(cfg).eval()
    nan_entry = EvalEntry(
        target=np.full((2, 128), np.nan, dtype=np.float32),
        past_cov=None,
        prediction_length=48,
    )
    out_f32 = predict_entries(
        model, cfg, [nan_entry], device=torch.device("cpu"), batch_size=1, use_autocast=False
    )
    assert np.isfinite(out_f32.raw_q21[0]).all()

    if torch.backends.mps.is_available():
        model_mps = TSFMModel(cfg).to("mps").eval()
        out_mps = predict_entries(
            model_mps,
            cfg,
            [nan_entry],
            device=torch.device("mps"),
            batch_size=1,
            use_autocast=True,
        )
        assert np.isfinite(out_mps.raw_q21[0]).all()


def test_dynamic_r_eval_37_variates_and_blackout_lag_bridging() -> None:
    cfg = TSFMConfig(
        run_name="mv_37",
        use_local_scale_and_lags=True,
        head_type="median_probit",
        use_group_attn=True,
        exclude_blank_keys=True,
    )
    model = TSFMModel(cfg).eval()

    # Simulate bizitobs_application (2 targets + 35 past covariates = 37 > strip_rows=32).
    rng = np.random.default_rng(5)
    entry_37 = EvalEntry(
        target=rng.normal(size=(2, 160)).astype(np.float32),
        past_cov=rng.normal(size=(35, 160)).astype(np.float32),
        prediction_length=60,
    )
    preds = predict_entries(
        model, cfg, [entry_37], device=torch.device("cpu"), batch_size=1, use_autocast=False
    )
    assert preds.raw_deciles[0].shape == (9, 60, 2)
    assert np.isfinite(preds.raw_deciles[0]).all()

    # Simulate 64-step pre-origin blackout: 4-cycle modular lags (S=24 -> jumps m=0..2 to hit earlier cycles).
    t_len = 256
    y = np.sin(np.arange(t_len, dtype=np.float32) * (2.0 * np.pi / 24.0))[None, :]
    y[:, -64:] = np.nan
    patched, _, _, _ = build_group_patch_features(
        y_ctx_real=y,
        left_pad=0,
        n_ctx_patches=8,
        n_hor_patches=1,
        patch_size=32,
        scaling_mode="observed_frozen_origin",
        use_local_scale_and_lags=True,
    )
    # Channel offset for S=24 lag_ok is 11 + index(24) = 13; must be 1.0 across horizon patch despite blackout!
    s24_ok_idx = 11 + SEASONAL_LAGS.index(24)
    assert float(patched[0, 8, :, s24_ok_idx].min()) == 1.0


def test_group_attention_step0_identity_step1_grads_invariance_and_firewall() -> None:
    cfg_base = TSFMConfig(
        run_name="g0",
        use_local_scale_and_lags=True,
        head_type="median_probit",
        use_group_attn=False,
        exclude_blank_keys=True,
    )
    cfg_grp = TSFMConfig(
        run_name="g1",
        use_local_scale_and_lags=True,
        head_type="median_probit",
        use_group_attn=True,
        exclude_blank_keys=True,
    )
    torch.manual_seed(0)
    m_base = TSFMModel(cfg_base).eval()
    torch.manual_seed(0)
    m_grp = TSFMModel(cfg_grp).eval()

    x = torch.randn(1, 6, 4, 32, cfg_grp.c_in)
    rel = torch.tensor([-2, -1, 0, 1], dtype=torch.long).view(1, 1, 4).expand(1, 6, 4)
    seg = torch.arange(6, dtype=torch.long).view(1, 6, 1).expand(1, 6, 4)
    obs = torch.tensor([True, True, True, False]).view(1, 1, 4).expand(1, 6, 4)
    grp = torch.tensor([0, 0, 0, 1, 2, -1], dtype=torch.long).view(1, 6, 1).expand(1, 6, 4)
    cov = torch.zeros((1, 6, 4), dtype=torch.bool)
    cov[:, 2, :] = True

    # Non-vacuous Step-0 identity check (warm-start parity + shared non-zero head fc2):
    # outputs change when inputs change, yet zero-init out_proj + cov_role_emb
    # preserve exact 0.0 equivalence between non-group and group models.
    torch.manual_seed(99)
    fc2_w = torch.randn_like(m_base.head.fc2.weight) * 0.1
    fc2_b = torch.randn_like(m_base.head.fc2.bias) * 0.1
    m_base_nz = TSFMModel(cfg_base).eval()
    m_base_nz.load_state_dict(m_base.state_dict())
    m_grp_nz = TSFMModel(cfg_grp).eval()
    m_grp_nz.load_state_dict(m_base.state_dict(), strict=False)
    with torch.no_grad():
        for m_nz in (m_base_nz, m_grp_nz):
            m_nz.head.fc2.weight.copy_(fc2_w)
            m_nz.head.fc2.bias.copy_(fc2_b)
        y_base = m_base_nz(x, rel, seg, obs, grp, cov)
        y_grp = m_grp_nz(x, rel, seg, obs, grp, cov)
        assert torch.max(torch.abs(y_base - y_grp)).item() == 0.0
        x_row3_pert = x.clone()
        x_row3_pert[:, 3, :3, :, :] += 10.0
        y_grp_row3 = m_grp_nz(x_row3_pert, rel, seg, obs, grp, cov)
        assert torch.max(torch.abs(y_grp[:, 3] - y_grp_row3[:, 3])).item() > 1e-3

    # Step-1 updates zero-init head fc2; next step immediately sends nonzero gradients through out_proj,
    # and the following step sends full-rank nonzero gradients into GroupAttention.qkv_proj.
    m_grp.train()
    opt = torch.optim.SGD(m_grp.parameters(), lr=0.2)
    y_step1 = m_grp(x, rel, seg, obs, grp, cov)
    (y_step1[:, :2, 96:, :] - 2.0).pow(2).mean().backward()
    opt.step()
    opt.zero_grad()

    y_step2 = m_grp(x, rel, seg, obs, grp, cov)
    (y_step2[:, :2, 96:, :] - 2.0).pow(2).mean().backward()
    out_grad = m_grp.blocks[3].group_attn.out_proj.weight.grad
    assert out_grad is not None and float(out_grad.abs().max().item()) > 0.0
    opt.step()
    opt.zero_grad()

    y_step3 = m_grp(x, rel, seg, obs, grp, cov)
    (y_step3[:, :2, 96:, :] - 2.0).pow(2).mean().backward()
    qkv_grad = m_grp.blocks[3].group_attn.qkv_proj.weight.grad
    assert qkv_grad is not None and float(qkv_grad.abs().max().item()) > 0.0
    opt.step()
    opt.zero_grad()

    # Post-SGD woken model checks in CPU float32:
    # 1) Univariate row (grp=1, idx=3) must stay invariant regardless of group 0 peers or slack row 5.
    # 2) One-way firewall on GroupAttention: perturbing target rows (0,1) leaves cov row (2) unchanged.
    # 3) Unobserved horizon past_cov (row 2, col 3) has 0.0 effect on targets, while observed
    #    context past_cov (row 2, col 0) produces > 1e-4 cross-variate sensitivity in Group 0.
    m_grp.eval()
    ga = m_grp.blocks[3].group_attn
    h_in = torch.randn(1, 6, 4, cfg_grp.d_model)
    h_pert = h_in.clone()
    h_pert[:, 0:2] += 100.0
    x_alt = x.clone()
    x_alt[:, 0:3] += 10.0
    x_alt[:, 5] += 20.0
    with torch.no_grad():
        c1 = ga(h_in, grp, cov)[:, 2]
        c2 = ga(h_pert, grp, cov)[:, 2]
        assert torch.max(torch.abs(c1 - c2)).item() == 0.0

        y_base_eval = m_grp(x, rel, seg, obs, grp, cov)
        y_alt_eval = m_grp(x_alt, rel, seg, obs, grp, cov)
        assert torch.max(torch.abs(y_base_eval[:, 3] - y_alt_eval[:, 3])).item() == 0.0

        x_hor_cov = x.clone()
        x_hor_cov[:, 2, 3, :, :] += 50.0
        y_hor_cov = m_grp(x_hor_cov, rel, seg, obs, grp, cov)
        assert torch.max(torch.abs(y_base_eval[:, :2, 96:, :] - y_hor_cov[:, :2, 96:, :])).item() == 0.0

        x_ctx_cov = x.clone()
        x_ctx_cov[:, 2, 0, :, :] += 50.0
        y_ctx_cov = m_grp(x_ctx_cov, rel, seg, obs, grp, cov)
        assert torch.max(torch.abs(y_base_eval[:, :2, 96:, :] - y_ctx_cov[:, :2, 96:, :])).item() > 1e-4

        # Verify bfloat16 univariate & slack packing invariance on MPS when available.
        if torch.backends.mps.is_available():
            m_mps = m_grp.to("mps").eval()
            with torch.autocast(device_type="mps", dtype=torch.bfloat16):
                y_bf1 = m_mps(x.to("mps"), rel.to("mps"), seg.to("mps"), obs.to("mps"), grp.to("mps"), cov.to("mps"))
                y_bf2 = m_mps(
                    x_alt.to("mps"), rel.to("mps"), seg.to("mps"), obs.to("mps"), grp.to("mps"), cov.to("mps")
                )
            assert torch.max(torch.abs(y_bf1[:, 3] - y_bf2[:, 3])).item() == 0.0


def test_scale_equivariance_across_si_units() -> None:
    rng = np.random.default_rng(19)
    t_ax = np.arange(160, dtype=np.float64)
    y_base = np.stack(
        [
            2.5 + 0.02 * np.sin(2.0 * np.pi * t_ax / 24.0) + rng.normal(0.0, 0.005, size=160),
            -1.2 + 0.015 * np.cos(2.0 * np.pi * t_ax / 12.0) + rng.normal(0.0, 0.004, size=160),
        ],
        axis=0,
    ).astype(np.float32)
    y_base[0, 50:60] = np.nan
    y_base[1, 10:15] = np.nan

    p_ref, obs_ref, _, _ = build_group_patch_features(
        y_ctx_real=y_base,
        left_pad=0,
        n_ctx_patches=5,
        n_hor_patches=1,
        patch_size=32,
        scaling_mode="observed_frozen_origin",
        use_local_scale_and_lags=True,
    )

    cfg = TSFMConfig(
        run_name="scale_eq_test",
        use_local_scale_and_lags=True,
        head_type="median_probit",
        use_group_attn=True,
        exclude_blank_keys=True,
        scaling_mode="observed_frozen_origin",
    )
    torch.manual_seed(7)
    model = TSFMModel(cfg).eval()
    pred_ref = predict_entries(
        model,
        cfg,
        [EvalEntry(target=y_base, past_cov=None, prediction_length=32)],
        device=torch.device("cpu"),
        batch_size=1,
        use_autocast=False,
    ).raw_q21[0]
    ref_scale = float(np.max(np.abs(pred_ref)))
    assert ref_scale > 0.0

    for factor, pat_tol in ((2.0**-20, 1e-7), (2.0**17, 1e-7), (1e-6, 5e-5), (1e5, 5e-5)):
        y_scaled = (y_base.astype(np.float64) * factor).astype(np.float32)
        p_s, obs_s, _, _ = build_group_patch_features(
            y_ctx_real=y_scaled,
            left_pad=0,
            n_ctx_patches=5,
            n_hor_patches=1,
            patch_size=32,
            scaling_mode="observed_frozen_origin",
            use_local_scale_and_lags=True,
        )
        assert np.array_equal(obs_s, obs_ref)
        assert float(np.max(np.abs(p_s - p_ref))) < pat_tol

        pred_s = predict_entries(
            model,
            cfg,
            [EvalEntry(target=y_scaled, past_cov=None, prediction_length=32)],
            device=torch.device("cpu"),
            batch_size=1,
            use_autocast=False,
        ).raw_q21[0]
        rel_err = float(np.max(np.abs((pred_s.astype(np.float64) / factor) - pred_ref))) / ref_scale
        assert rel_err < 1e-5, f"Scale equivariance failed at factor={factor}: rel_err={rel_err:.3e}"


def test_tsfm_model_from_pretrained_offline() -> None:
    ckpt_file = Path(__file__).resolve().parent.parent / "checkpoints" / "run5_coupled_synth_mix" / "model_soup.safetensors"
    if not ckpt_file.exists():
        pytest.skip("Saved checkpoint not present in checkout")

    model, cfg = TSFMModel.from_pretrained("run5_coupled_synth_mix", device=torch.device("cpu"))
    assert cfg.run_name == "run5_coupled_synth_mix"
    assert model.count_parameters() == 16_317_344

    rng = np.random.default_rng(7)
    y = rng.normal(50.0, 4.0, size=(1, 96)).astype(np.float32)
    out = predict_entries(
        model,
        cfg,
        [EvalEntry(target=y, past_cov=None, prediction_length=24)],
        device=torch.device("cpu"),
        batch_size=1,
        use_autocast=False,
    )
    assert out.raw_deciles[0].shape == (9, 24, 1)
    assert np.isfinite(out.raw_deciles[0]).all()
