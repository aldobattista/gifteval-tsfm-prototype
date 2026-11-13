"""Tests for quantile heads, pinball CRPS loss, GluonTS parity, bootstrap CI, and rollout bounds."""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from tsfm_proto.config import DECILE_LEVELS, TSFMConfig
from tsfm_proto.data import _align_cov_width
from tsfm_proto.eval_metrics import (
    EvalEntry,
    paired_bootstrap_geomean_ci,
    predict_entries,
    score_task_fast,
)
from tsfm_proto.heads import LeftAnchoredSoftplusHead, MedianAnchoredProbitHead
from tsfm_proto.loss import compute_pinball_crps_loss
from tsfm_proto.model import TSFMModel
from tsfm_proto.packing import PackedBatch


def test_heads_strict_monotonicity_and_step0_probit_calibration() -> None:
    cfg_soft = TSFMConfig(run_name="h_soft", head_type="left_softplus")
    cfg_prob = TSFMConfig(run_name="h_prob", head_type="median_probit")
    h_soft = LeftAnchoredSoftplusHead(cfg_soft).eval()
    h_prob = MedianAnchoredProbitHead(cfg_prob).eval()

    torch.manual_seed(0)
    # 10,000 random hidden tokens across [B=10, R=10, L=100, D=384].
    h_rand = torch.randn(10, 10, 100, cfg_soft.d_model)
    with torch.no_grad():
        q_soft = h_soft(h_rand)
        q_prob = h_prob(h_rand)
    assert bool((q_soft[..., 1:] >= q_soft[..., :-1]).all())
    assert bool((q_prob[..., 1:] >= q_prob[..., :-1]).all())

    # Zero hidden activations represent Step-0 initial predictive distribution on arcsinh(N(0,1)).
    h_zero = torch.zeros(1, 32, 4, cfg_prob.d_model)
    y_true = torch.asinh(torch.randn(1, 32, 128))
    mask = torch.ones_like(y_true, dtype=torch.bool)
    sid = torch.arange(32).view(1, 32, 1).expand(1, 32, 128)
    gcnt = torch.ones_like(y_true)
    with torch.no_grad():
        q0_soft = h_soft(h_zero)
        q0_prob = h_prob(h_zero)
    l_soft = compute_pinball_crps_loss(q0_soft, y_true, mask, sid, gcnt, cfg_soft)
    l_prob = compute_pinball_crps_loss(q0_prob, y_true, mask, sid, gcnt, cfg_prob)
    assert float(l_soft.raw_crps.item()) > 4.0
    assert float(l_prob.raw_crps.item()) < 0.50
    assert 0.77 <= float(l_prob.cov80.item()) <= 0.83


def test_loss_group_balance_and_outlier_damping() -> None:
    cfg_bal = TSFMConfig(run_name="loss_bal", loss_mode="per_series_damped", loss_damping_tau=12.0)
    q = torch.zeros(1, 3, 4, len(cfg_bal.quantiles))
    # Series 0 (univariate, V_g=1): error 1.0. Rows 1 & 2 (multivariate V_g=2): extreme outlier 1000.0.
    tgt = torch.tensor([[[1.0] * 4, [1000.0] * 4, [1000.0] * 4]], dtype=torch.float32)
    mask = torch.ones_like(tgt, dtype=torch.bool)
    sid = torch.tensor([[0, 1, 2]]).unsqueeze(-1).expand(1, 3, 4)
    gcnt = torch.tensor([[1.0, 2.0, 2.0]]).unsqueeze(-1).expand(1, 3, 4)
    stats = compute_pinball_crps_loss(q, tgt, mask, sid, gcnt, cfg_bal)
    # Damping phi_12 keeps loss well below 60 despite raw error ~1000.
    assert float(stats.loss.item()) < 60.0
    assert float(stats.raw_crps.item()) > 500.0


def test_score_task_fast_matches_gluonts_exact_parity() -> None:
    ListDataset = pytest.importorskip("gluonts.dataset.common").ListDataset
    split = pytest.importorskip("gluonts.dataset.split").split
    ev_mod = pytest.importorskip("gluonts.ev.metrics")
    MASE, MeanWeightedSumQuantileLoss = ev_mod.MASE, ev_mod.MeanWeightedSumQuantileLoss
    evaluate_forecasts = pytest.importorskip("gluonts.model").evaluate_forecasts
    QuantileForecast = pytest.importorskip("gluonts.model.forecast").QuantileForecast

    rng = np.random.default_rng(7)
    # Cover normal, constant zero-seasonal-error, all-NaN past, and NaN labels.
    series_list = [
        {
            "item_id": "norm",
            "start": pd.Period("2022-01-01", freq="D"),
            "target": (rng.normal(size=40) + 10.0).astype(np.float32),
        },
        {
            "item_id": "const_past",
            "start": pd.Period("2022-01-01", freq="D"),
            "target": np.array([5.0] * 32 + list(rng.uniform(4.0, 8.0, size=8)), dtype=np.float32),
        },
        {
            "item_id": "nan_past",
            "start": pd.Period("2022-01-01", freq="D"),
            "target": np.array([np.nan] * 32 + [2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], dtype=np.float32),
        },
    ]
    ds = ListDataset(series_list, freq="D")
    _, test_tmpl = split(ds, offset=-8)
    td = test_tmpl.generate_instances(prediction_length=8, windows=1)

    q_arrs = []
    entries: list[EvalEntry] = []
    forecasts: list[QuantileForecast] = []
    keys = [str(q) for q in DECILE_LEVELS]

    for inp, lbl in zip(td.input, td.label):
        base = rng.normal(loc=5.0, scale=0.5, size=(1, 8))
        offsets = np.linspace(-1.5, 1.5, len(DECILE_LEVELS))[:, None]
        fc = (base + offsets).astype(np.float32)  # [9, 8]
        q_arrs.append(fc[:, :, None])  # [9, 8, 1]
        entries.append(
            EvalEntry(
                target=np.asarray(inp["target"], dtype=np.float32)[None, :],
                past_cov=None,
                prediction_length=8,
                future_target=np.asarray(lbl["target"], dtype=np.float32)[None, :],
                season_length=7,
            )
        )
        forecasts.append(
            QuantileForecast(
                forecast_arrays=fc,
                forecast_keys=keys,
                start_date=inp["start"] + len(inp["target"]),
                item_id=inp["item_id"],
            )
        )

    fast = score_task_fast(entries, q_arrs)
    ref = evaluate_forecasts(
        forecasts,
        test_data=td,
        metrics=[MASE(), MeanWeightedSumQuantileLoss(DECILE_LEVELS)],
        axis=None,
        mask_invalid_label=True,
        allow_nan_forecast=False,
        seasonality=7,
    )
    assert abs(fast["MASE[0.5]"] - float(ref["MASE[0.5]"].iloc[0])) < 1e-6
    assert (
        abs(fast["mean_weighted_sum_quantile_loss"] - float(ref["mean_weighted_sum_quantile_loss"].iloc[0]))
        < 1e-6
    )


def test_bootstrap_determinism_multichunk_recrop_and_postproc() -> None:
    rng = np.random.default_rng(1)
    ratios = rng.uniform(0.7, 1.05, size=97)
    fams = [f"fam_{i % 28}" for i in range(97)]
    c1 = paired_bootstrap_geomean_ci(ratios, fams, n_boot=1000, seed=42)
    c2 = paired_bootstrap_geomean_ci(ratios, fams, n_boot=1000, seed=42)
    assert c1 == c2
    assert c1["cluster_ci95_lo"] <= c1["gmr"] <= c1["cluster_ci95_hi"]

    # 2-chunk autoregressive rollout on H=720 with T_ctx=4096 in Run 0 (max_horizon_patches=16 -> H1=512).
    cfg = TSFMConfig(run_name="rollout_test", max_horizon_patches=16)
    model = TSFMModel(cfg).eval()
    # Strictly positive integer counts -> triggers clamp (>=0) + integer snap (< 2^23); sym_avg skips min_obs==0.
    ctx_counts = rng.integers(0, 50, size=(1, 4096)).astype(np.float32)
    ent = EvalEntry(target=ctx_counts, past_cov=None, prediction_length=720)
    out = predict_entries(
        model, cfg, [ent], device=torch.device("cpu"), batch_size=1, sym_avg=True, use_autocast=False
    )
    assert out.raw_deciles[0].shape == (9, 720, 1)
    assert np.isfinite(out.raw_deciles[0]).all()
    assert bool((out.clamp_deciles[0] >= 0.0).all())
    assert bool(np.all(out.clamp_deciles[0] == np.rint(out.clamp_deciles[0])))

    # Blackout applied across chunk 1 and chunk 2 (apply_blackout_steps=32) must isolate last 32 steps.
    ctx_corrupt = ctx_counts.copy()
    ctx_corrupt[:, -32:] += 9999.0
    ent_a = EvalEntry(target=ctx_counts, past_cov=None, prediction_length=720)
    ent_b = EvalEntry(target=ctx_corrupt, past_cov=None, prediction_length=720)
    out_a = predict_entries(
        model, cfg, [ent_a], device=torch.device("cpu"), batch_size=1, apply_blackout_steps=32, use_autocast=False
    )
    out_b = predict_entries(
        model, cfg, [ent_b], device=torch.device("cpu"), batch_size=1, apply_blackout_steps=32, use_autocast=False
    )
    assert float(np.max(np.abs(out_a.raw_deciles[0] - out_b.raw_deciles[0]))) == 0.0

    # Strictly positive series (min > 0, all >= 0) must stay identical under sym_avg=True,
    # whereas signed series containing negative values apply reflection-symmetric averaging.
    cfg_sym = TSFMConfig(
        run_name="sym_test",
        use_local_scale_and_lags=True,
        head_type="median_probit",
        use_group_attn=True,
    )
    torch.manual_seed(2)
    m_sym = TSFMModel(cfg_sym).eval()
    pos_only = rng.uniform(5.0, 25.0, size=(1, 128)).astype(np.float32)
    signed_arr = (pos_only - 15.0).astype(np.float32)
    ent_pos = EvalEntry(target=pos_only, past_cov=None, prediction_length=32)
    ent_sig = EvalEntry(target=signed_arr, past_cov=None, prediction_length=32)
    out_nosym = predict_entries(
        m_sym, cfg_sym, [ent_pos, ent_sig], device=torch.device("cpu"), batch_size=2, sym_avg=False, use_autocast=False
    )
    out_sym = predict_entries(
        m_sym, cfg_sym, [ent_pos, ent_sig], device=torch.device("cpu"), batch_size=2, sym_avg=True, use_autocast=False
    )
    assert float(np.max(np.abs(out_nosym.raw_deciles[0] - out_sym.raw_deciles[0]))) == 0.0
    assert float(np.max(np.abs(out_nosym.raw_deciles[1] - out_sym.raw_deciles[1]))) > 0.0


def test_loss_balance_1_over_vg_vs_flat_token() -> None:
    # Verify exact 1/V_g group balance on gradient norms: Group A has V_g=4 rows and Group B has V_g=1 row.
    # Under flat_token the 4-row group receives 4x total gradient; under per_series_damped it receives 1x (equal).
    b, r, t, q = 1, 5, 8, 21
    target = torch.zeros((b, r, t), dtype=torch.float32)
    mask = torch.ones((b, r, t), dtype=torch.bool)
    sid = torch.arange(r, dtype=torch.long).view(1, r, 1).expand(b, r, t).clone()
    v_cnt = torch.tensor([4.0, 4.0, 4.0, 4.0, 1.0], dtype=torch.float32).view(1, r, 1).expand(b, r, t).clone()

    pred_flat = torch.full((b, r, t, q), 1.0, dtype=torch.float32, requires_grad=True)
    cfg_flat = TSFMConfig(run_name="flat", loss_mode="flat_token")
    m_flat = compute_pinball_crps_loss(pred_flat, target, mask, sid, v_cnt, cfg_flat)
    m_flat.loss.backward()
    grad_flat_grp_a = float(pred_flat.grad[:, :4].abs().sum().item())
    grad_flat_grp_b = float(pred_flat.grad[:, 4:].abs().sum().item())
    assert abs(grad_flat_grp_a / grad_flat_grp_b - 4.0) < 1e-5

    pred_bal = torch.full((b, r, t, q), 1.0, dtype=torch.float32, requires_grad=True)
    cfg_bal = TSFMConfig(run_name="bal", loss_mode="per_series_damped", loss_damping_tau=12.0)
    m_bal = compute_pinball_crps_loss(pred_bal, target, mask, sid, v_cnt, cfg_bal)
    m_bal.loss.backward()
    grad_bal_grp_a = float(pred_bal.grad[:, :4].abs().sum().item())
    grad_bal_grp_b = float(pred_bal.grad[:, 4:].abs().sum().item())
    assert abs(grad_bal_grp_a / grad_bal_grp_b - 1.0) < 1e-5


def test_eval_gifteval_compute_slices_freq_filter() -> None:
    eval_path = Path(__file__).resolve().parent.parent / "scripts" / "eval_gifteval.py"
    spec = importlib.util.spec_from_file_location("eval_gifteval_mod", eval_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    rows = []
    for i, freq in enumerate(["10S", "5T", "H", "D", "W", "M", "Q-DEC", "A-DEC"]):
        rows.append(
            {
                "run_name": "run5_coupled_synth_mix",
                "postproc": "raw",
                "task_id": f"ds{i}/{freq}/short",
                "dataset": f"ds{i}",
                "frequency": freq,
                "term": "short",
                "domain": "Energy",
                "horizon": 24,
                "is_mv": False,
                "has_past_cov": False,
                "is_missing_ds": False,
                "nmwql": 0.5,
                "nmase": 0.8,
                "cov80": 0.8,
            }
        )
    rows_out = mod._compute_slices(pd.DataFrame(rows), "run5_coupled_synth_mix", "raw")
    slices = pd.DataFrame(rows_out)
    low = slices[slices["slice"] == "Low-Res Freq (A/Q/M/W)"].iloc[0]
    high = slices[slices["slice"] == "High-Res Freq (D/H/Min/S)"].iloc[0]
    assert int(low["n_tasks"]) == 4
    assert int(high["n_tasks"]) == 4


def test_loss_per_series_damped_multibatch_no_sid_collision() -> None:
    cfg = TSFMConfig(run_name="loss_mb", loss_mode="per_series_damped", loss_damping_tau=12.0)
    n_q = len(cfg.quantiles)
    q0 = torch.zeros((1, 2, 8, n_q), dtype=torch.float32)
    q1 = torch.zeros((1, 2, 8, n_q), dtype=torch.float32)
    # Identical local sid = [0, 1] in both strips, different magnitudes (1.0 vs 100.0).
    # Because phi_12 is strictly concave, colliding sid=0 across strips would mis-average before log1p!
    t0 = torch.full((1, 2, 8), 1.0, dtype=torch.float32)
    t1 = torch.full((1, 2, 8), 100.0, dtype=torch.float32)
    mask = torch.ones((1, 2, 8), dtype=torch.bool)
    sid = torch.tensor([[0, 1]], dtype=torch.long).unsqueeze(-1).expand(1, 2, 8)
    gcnt = torch.ones((1, 2, 8), dtype=torch.float32)

    m0 = compute_pinball_crps_loss(q0, t0, mask, sid, gcnt, cfg)
    m1 = compute_pinball_crps_loss(q1, t1, mask, sid, gcnt, cfg)
    m_both = compute_pinball_crps_loss(
        torch.cat([q0, q1], dim=0),
        torch.cat([t0, t1], dim=0),
        torch.cat([mask, mask], dim=0),
        torch.cat([sid, sid], dim=0),
        torch.cat([gcnt, gcnt], dim=0),
        cfg,
    )
    expected_loss = 0.5 * (float(m0.loss.item()) + float(m1.loss.item()))
    assert abs(float(m_both.loss.item()) - expected_loss) < 1e-6

    # Also verify train._next_collated_batch stacks B=2 PackedBatch items cleanly.
    train_path = Path(__file__).resolve().parent.parent / "scripts" / "train.py"
    spec = importlib.util.spec_from_file_location("train_script_mod", train_path)
    assert spec is not None and spec.loader is not None
    train_mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = train_mod
    spec.loader.exec_module(train_mod)

    def _dummy_pb(val: float) -> PackedBatch:
        return PackedBatch(
            patch_inputs=torch.zeros((1, 2, 2, 32, cfg.c_in), dtype=torch.float32),
            target_scaled=torch.full((1, 2, 64), val, dtype=torch.float32),
            loss_mask=torch.ones((1, 2, 64), dtype=torch.bool),
            rel_patch_idx=torch.zeros((1, 2, 2), dtype=torch.long),
            segment_id=torch.zeros((1, 2, 2), dtype=torch.long),
            patch_obs_any=torch.ones((1, 2, 2), dtype=torch.bool),
            patch_group_id=torch.zeros((1, 2, 2), dtype=torch.long),
            patch_is_cov=torch.zeros((1, 2, 2), dtype=torch.bool),
            series_id=torch.tensor([[0, 1]], dtype=torch.long).unsqueeze(-1).expand(1, 2, 64),
            group_target_count=torch.ones((1, 2, 64), dtype=torch.float32),
        )

    collated = train_mod._next_collated_batch(iter([_dummy_pb(1.0), _dummy_pb(100.0)]), batch_size=2)
    assert collated.target_scaled.shape == (2, 2, 64)
    assert float(collated.target_scaled[0].mean().item()) == 1.0
    assert float(collated.target_scaled[1].mean().item()) == 100.0


def test_predict_entries_multichunk_while_and_blackout_no_tail_peek() -> None:
    cfg = TSFMConfig(run_name="mc3", max_horizon_patches=4)  # 4 * 32 = 128 steps/chunk -> H=320 takes 3 chunks
    torch.manual_seed(3)
    model = TSFMModel(cfg).eval()
    rng = np.random.default_rng(19)
    ctx_3 = rng.uniform(5.0, 20.0, size=(1, 256)).astype(np.float32)
    ent_3 = EvalEntry(target=ctx_3, past_cov=None, prediction_length=320)
    out_3 = predict_entries(model, cfg, [ent_3], device=torch.device("cpu"), batch_size=1, use_autocast=False)
    assert out_3.raw_deciles[0].shape == (9, 320, 1)
    assert np.isfinite(out_3.raw_deciles[0]).all()

    # Strictly positive visible context [10..30] with negative sentinel -999 in the final 32 masked blackout steps.
    # Neither clamp_deciles nor sym_avg may peek at the masked tail.
    pos_with_masked_neg = np.linspace(10.0, 30.0, 160, dtype=np.float32)[None, :].copy()
    pos_with_masked_neg[:, -32:] = -999.0
    ent_b = EvalEntry(target=pos_with_masked_neg, past_cov=None, prediction_length=32)
    out_nosym = predict_entries(
        model, cfg, [ent_b], device=torch.device("cpu"), batch_size=1, apply_blackout_steps=32, sym_avg=False, use_autocast=False
    )
    out_sym = predict_entries(
        model, cfg, [ent_b], device=torch.device("cpu"), batch_size=1, apply_blackout_steps=32, sym_avg=True, use_autocast=False
    )
    assert bool((out_sym.clamp_deciles[0] >= 0.0).all())
    assert float(np.max(np.abs(out_sym.raw_deciles[0] - out_nosym.raw_deciles[0]))) == 0.0


def test_loss_ignores_nan_in_masked_out_targets() -> None:
    for mode in ("flat_token", "per_series_damped"):
        cfg = TSFMConfig(run_name=f"nan_mask_{mode}", loss_mode=mode)
        pred = torch.zeros((1, 2, 8, len(cfg.quantiles)), dtype=torch.float32)
        tgt = torch.full((1, 2, 8), 1.5, dtype=torch.float32)
        mask = torch.ones((1, 2, 8), dtype=torch.bool)
        tgt[:, :, 4:] = float("nan")
        mask[:, :, 4:] = False
        sid = torch.tensor([[0, 1]], dtype=torch.long).unsqueeze(-1).expand(1, 2, 8)
        gcnt = torch.ones((1, 2, 8), dtype=torch.float32)

        stats = compute_pinball_crps_loss(pred, tgt, mask, sid, gcnt, cfg)
        assert torch.isfinite(stats.loss)
        assert torch.isfinite(stats.raw_crps)
        assert torch.isfinite(stats.cov80)


def test_pack_strip_batch_small_grid_and_origin_invariants() -> None:
    from tsfm_proto.data import SeriesItem
    from tsfm_proto.packing import pack_strip_batch

    cfg = TSFMConfig(
        run_name="small_grid",
        patch_size=32,
        max_patches=8,
        max_horizon_patches=4,
        strip_rows=6,
        use_group_attn=True,
        use_local_scale_and_lags=True,
        scaling_mode="observed_frozen_origin",
        history_blackout_prob=0.3,
    )
    rng = np.random.default_rng(123)

    def _sampler(r: np.random.Generator) -> SeriesItem:
        t_len = int(r.integers(48, 160))
        tgt = r.normal(size=(2, t_len)).astype(np.float32)
        cov = r.normal(size=(1, t_len)).astype(np.float32)
        return SeriesItem(dataset_name="synth_coupled_mv", item_id="mv_0", freq="H", target=tgt, past_cov=cov)

    batch = pack_strip_batch(_sampler, rng, cfg)
    assert batch.patch_inputs.shape == (1, 6, 8, 32, cfg.c_in)

    # Every committed segment must anchor at least one rel_patch_idx == 0 and keep covariate loss_mask False.
    seg = batch.segment_id[0]
    rel = batch.rel_patch_idx[0]
    cov_patches = batch.patch_is_cov[0]
    for s_id in range(int(seg.max().item()) + 1):
        in_seg = seg == s_id
        assert bool((in_seg & (rel == 0)).any())
        if bool((in_seg & cov_patches).any()):
            step_cov = cov_patches.unsqueeze(-1).expand(6, 8, 32).reshape(6, 256)
            assert not bool(batch.loss_mask[0][step_cov].any())


def test_hub_partial_local_dir_falls_back_cleanly(tmp_path: Path) -> None:
    from safetensors.torch import save_file
    from tsfm_proto.hub import from_pretrained

    partial_dir = tmp_path / "partial_run"
    partial_dir.mkdir(parents=True)
    save_file({"dummy": torch.zeros(1)}, str(partial_dir / "model_soup.safetensors"))

    import pytest

    with pytest.raises(FileNotFoundError, match="Could not find"):
        from_pretrained("partial_run", checkpoints_root=tmp_path, hf_repo_id=None)


def test_pack_strip_batch_drops_all_nan_cov_and_uses_post_blackout_moments() -> None:
    from tsfm_proto.data import SeriesItem
    from tsfm_proto.packing import pack_strip_batch
    from tsfm_proto.scaling import _scale_equivariant_stats_1d

    # 1) All-NaN covariate channel in drawn window must be dropped before scaling/packing.
    cfg_nan = TSFMConfig(
        run_name="drop_nan_cov",
        patch_size=32,
        max_patches=4,
        max_horizon_patches=1,
        strip_rows=3,
        use_group_attn=True,
        use_local_scale_and_lags=False,
        scaling_mode="observed_frozen_origin",
        history_blackout_prob=0.0,
    )
    t_ax = np.arange(96, dtype=np.float32)

    def _nan_cov_sampler(r: np.random.Generator) -> SeriesItem:
        del r
        tgt = (10.0 + np.sin(2.0 * np.pi * t_ax / 24.0))[None, :].astype(np.float32)
        cov = np.stack([np.full(96, np.nan, dtype=np.float32), np.cos(2.0 * np.pi * t_ax / 12.0)], axis=0)
        return SeriesItem(dataset_name="synth_mv", item_id="nan_cov_0", freq="H", target=tgt, past_cov=cov)

    batch_nan = pack_strip_batch(_nan_cov_sampler, np.random.default_rng(3), cfg_nan)
    assert torch.isfinite(batch_nan.patch_inputs).all()
    assert int((batch_nan.patch_is_cov[0].any(dim=-1)).sum()) == 1

    # 2) Post-blackout scaling uses surviving observations (not pre-blackout moments).
    cfg_blk = TSFMConfig(
        run_name="post_blk",
        patch_size=32,
        max_patches=3,
        max_horizon_patches=1,
        strip_rows=1,
        use_group_attn=False,
        use_local_scale_and_lags=False,
        scaling_mode="observed_frozen_origin",
        history_blackout_prob=1.0,
    )
    full_line = np.arange(96, dtype=np.float32)[None, :]  # ctx=64 (0..63), hor=32 (64..95)

    def _blk_sampler(r: np.random.Generator) -> SeriesItem:
        del r
        return SeriesItem(dataset_name="synth_blk", item_id="blk_0", freq="H", target=full_line, past_cov=None)

    batch_blk = pack_strip_batch(_blk_sampler, np.random.default_rng(7), cfg_blk)
    obs_ctx = batch_blk.patch_inputs[0, 0, :2, :, 1].reshape(-1).numpy() > 0.5
    assert obs_ctx.sum() < 64
    mu_pre, sig_pre = _scale_equivariant_stats_1d(full_line[0, :64])
    mu_post, sig_post = _scale_equivariant_stats_1d(full_line[0, :64][obs_ctx])
    got_hor0 = float(batch_blk.target_scaled[0, 0, 64])
    exp_post = float(np.arcsinh((64.0 - mu_post) / sig_post))
    exp_pre = float(np.arcsinh((64.0 - mu_pre) / sig_pre))
    assert abs(got_hor0 - exp_post) < 1e-5
    assert abs(got_hor0 - exp_pre) > 1e-4


def test_align_cov_width_and_prepare_chunk_job_mismatched_cov() -> None:
    short_cov = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    padded = _align_cov_width(short_cov, 5)
    assert padded.shape == (1, 5)
    assert np.isnan(padded[0, :2]).all()
    assert np.allclose(padded[0, 2:], [1.0, 2.0, 3.0])

    long_cov = np.arange(8, dtype=np.float32)[None, :]
    sliced = _align_cov_width(long_cov, 5)
    assert sliced.shape == (1, 5)
    assert np.allclose(sliced[0], [3.0, 4.0, 5.0, 6.0, 7.0])

    cfg = TSFMConfig.from_yaml(Path(__file__).resolve().parents[1] / "configs" / "run4_multivariate_group_attn.yaml")
    model = TSFMModel(cfg).eval()
    rng = np.random.default_rng(19)
    tgt = rng.normal(0.0, 1.0, size=(2, 96)).astype(np.float32)
    cov_mismatch = rng.normal(0.0, 1.0, size=(1, 80)).astype(np.float32)
    out = predict_entries(
        model,
        cfg,
        [EvalEntry(target=tgt, past_cov=cov_mismatch, prediction_length=32)],
        device=torch.device("cpu"),
        batch_size=1,
        use_autocast=False,
    )
    assert out.raw_deciles[0].shape == (9, 32, 2)
    assert np.isfinite(out.raw_deciles[0]).all()


def test_eval_gifteval_recompute_summaries_without_gluonts_and_yaml_nparams(tmp_path: Path) -> None:
    proj_root = Path(__file__).resolve().parents[1]
    res_src = proj_root / "results"
    res_tmp = tmp_path / "results"
    res_tmp.mkdir()
    empty_ckpts = tmp_path / "empty_ckpts"
    empty_ckpts.mkdir()

    copied = 0
    for p in res_src.glob("gifteval_per_task_run*.csv"):
        shutil.copy2(p, res_tmp / p.name)
        copied += 1
    shutil.copy2(res_src / "seasonal_naive_official_reference.csv", res_tmp / "seasonal_naive_official_reference.csv")
    assert copied == 6

    script_path = proj_root / "scripts" / "eval_gifteval.py"
    no_official_dir = tmp_path / "no_official"
    code = (
        "import runpy, sys\n"
        "for k in ['gluonts', 'gluonts.ev', 'gluonts.ev.metrics', 'gluonts.model', "
        "'gluonts.model.forecast', 'gluonts.time_feature']:\n"
        "    sys.modules[k] = None\n"
        f"sys.argv = [{str(script_path)!r}, '--recompute-summaries-only', "
        f"'--results-dir', {str(res_tmp)!r}, '--ckpt-root', {str(empty_ckpts)!r}, "
        f"'--official-leaderboard-dir', {str(no_official_dir)!r}]\n"
        f"runpy.run_path({str(script_path)!r}, run_name='__main__')\n"
    )
    env = {**os.environ, "PYTHONPATH": str(proj_root / "src")}
    proc = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}"
    summary_df = pd.read_csv(res_tmp / "gifteval_ablation_summary.csv")
    assert set(summary_df["n_params"].astype(int).tolist()) == {14_816_800, 15_136_288, 16_317_344}
