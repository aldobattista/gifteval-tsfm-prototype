"""M2 Max bfloat16 MPS / CUDA trainer for progressive TSFM prototype runs.

Implements AdamW with decoupled 2D+ weight decay, Warmup-Stable-Decay (WSD) LR
scheduling, 5-snapshot float64 cooldown weight averaging (soup), dual W&B and
scalar TensorBoard logging (plus CSV metric archival in ``results/`` on full runs),
and held-out validation comparison between the final checkpoint and the soup
without ever touching GIFT-Eval test splits.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import time
from dataclasses import replace
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from torch.utils.tensorboard import SummaryWriter

from tsfm_proto.config import TSFMConfig
from tsfm_proto.data import SubsetDatasetStore, make_step_rng
from tsfm_proto.eval_metrics import evaluate_val_split
from tsfm_proto.hub import build_float64_soup, load_safetensors_state, save_checkpoint
from tsfm_proto.loss import compute_pinball_crps_loss
from tsfm_proto.model import TSFMModel
from tsfm_proto.packing import PackedBatch, pack_strip_batch

DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _next_collated_batch(data_iter: Iterator[PackedBatch], batch_size: int) -> PackedBatch:
    """Fetch one packed strip batch or concatenate ``batch_size`` strips along dim 0."""
    first = next(data_iter)
    if batch_size <= 1:
        return first
    items = [first] + [next(data_iter) for _ in range(batch_size - 1)]
    return PackedBatch(
        patch_inputs=torch.cat([x.patch_inputs for x in items], dim=0),
        target_scaled=torch.cat([x.target_scaled for x in items], dim=0),
        loss_mask=torch.cat([x.loss_mask for x in items], dim=0),
        rel_patch_idx=torch.cat([x.rel_patch_idx for x in items], dim=0),
        segment_id=torch.cat([x.segment_id for x in items], dim=0),
        patch_obs_any=torch.cat([x.patch_obs_any for x in items], dim=0),
        patch_group_id=torch.cat([x.patch_group_id for x in items], dim=0),
        patch_is_cov=torch.cat([x.patch_is_cov for x in items], dim=0),
        series_id=torch.cat([x.series_id for x in items], dim=0),
        group_target_count=torch.cat([x.group_target_count for x in items], dim=0),
    )


class _PackedStripStream(IterableDataset):
    """CPU-side parallel stream of pre-packed [1, 32, 128] strip batches."""

    def __init__(self, subset_dir: Path, cfg: TSFMConfig) -> None:
        super().__init__()
        self.subset_dir = Path(subset_dir)
        self.cfg = cfg

    def __iter__(self) -> Iterator[PackedBatch]:
        info = get_worker_info()
        worker_id = 0 if info is None else info.id
        num_workers = 1 if info is None else info.num_workers

        store = SubsetDatasetStore(
            self.subset_dir,
            val_num_series=self.cfg.val_num_series,
        )
        probs = store.sampling_probs(self.cfg.sampling_alpha)

        step_idx = worker_id
        while True:
            rng = make_step_rng(self.cfg.seed, step_idx, worker_id)
            batch = pack_strip_batch(
                sampler_fn=lambda r: store.sample_train_item(r, self.cfg, probs),
                rng=rng,
                cfg=self.cfg,
            )
            yield batch
            step_idx += num_workers


def _resolve_device(choice: str) -> torch.device:
    if choice != "auto":
        return torch.device(choice)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _sync_device(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def _scale_schedule_cfg(cfg: TSFMConfig, max_steps: int | None) -> TSFMConfig:
    """Proportionally rescale WSD warmup, cooldown, and soup steps for short smoke runs."""
    if max_steps is None or max_steps == cfg.total_steps:
        return cfg
    scale = float(max_steps) / float(cfg.total_steps)
    warmup = max(1, int(round(cfg.warmup_steps * scale)))
    cooldown = max(warmup + 1, min(max_steps - 1, int(round(cfg.cooldown_start_step * scale))))
    n_snaps = len(cfg.soup_steps)
    soup_steps = tuple(
        sorted(
            {
                max(
                    cooldown + 1,
                    min(
                        max_steps,
                        int(round(cooldown + (i + 1) * (max_steps - cooldown) / n_snaps)),
                    ),
                )
                for i in range(n_snaps)
            }
            | {max_steps}
        )
    )
    return replace(
        cfg,
        total_steps=int(max_steps),
        warmup_steps=warmup,
        cooldown_start_step=cooldown,
        soup_steps=soup_steps,
        val_every_steps=min(cfg.val_every_steps, max_steps),
    )


def _wsd_lr_multiplier(step: int, cfg: TSFMConfig) -> float:
    """Warmup -> constant plateau -> cosine cooldown multiplier."""
    if step <= cfg.warmup_steps:
        return float(step) / float(max(1, cfg.warmup_steps))
    if step <= cfg.cooldown_start_step:
        return 1.0
    span = max(1, cfg.total_steps - cfg.cooldown_start_step)
    progress = min(1.0, float(step - cfg.cooldown_start_step) / float(span))
    min_ratio = float(cfg.min_lr_ratio)
    return min_ratio + 0.5 * (1.0 - min_ratio) * (1.0 + math.cos(math.pi * progress))


def _group_attn_lr_multiplier(step: int, cfg: TSFMConfig) -> float:
    """Two-timescale LR schedule for cross-variate GroupAttention residual adapters.

    Holds adapters frozen at zero during the initial 15% of training so the shared
    temporal backbone settles first, then smoothly opens to 0.20x the backbone rate.
    """
    base_mult = _wsd_lr_multiplier(step, cfg)
    start_step = int(round(0.15 * cfg.total_steps))
    ramp_end = max(start_step + 1, int(round(0.25 * cfg.total_steps)))
    if step <= start_step:
        return 0.0
    if step < ramp_end:
        ramp = float(step - start_step) / float(ramp_end - start_step)
        return 0.20 * base_mult * ramp
    return 0.20 * base_mult


def _init_wandb(cfg: TSFMConfig, mode: str, run_dir: Path):
    """Initialize W&B with graceful offline fallback if online network is unavailable."""
    if mode == "disabled":
        return None
    try:
        import wandb  # type: ignore[import-untyped]
    except ImportError:
        return None

    env_mode = mode
    os.environ["WANDB_SILENT"] = "true"
    try:
        return wandb.init(
            project="gifteval-tsfm-prototype",
            name=cfg.run_name,
            config=cfg.to_dict(),
            dir=str(run_dir),
            mode=env_mode,
        )
    except Exception:
        if env_mode != "offline":
            try:
                return wandb.init(
                    project="gifteval-tsfm-prototype",
                    name=cfg.run_name,
                    config=cfg.to_dict(),
                    dir=str(run_dir),
                    mode="offline",
                )
            except Exception:
                return None
        return None


def _upsert_val_comparison_row(csv_path: Path, row: dict[str, object]) -> None:
    """Insert or update one run's validation comparison row in ``val_soup_vs_last.csv``."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df = df[df["run_name"] != row["run_name"]]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df = df.sort_values("run_name").reset_index(drop=True)
    df.to_csv(csv_path, index=False)


def _upsert_train_history_csv(hist_path: Path, run_name: str, run_csv_path: Path) -> None:
    """Insert or update one run's step/validation CSV telemetry in ``train_metrics_history.csv``."""
    hist_path.parent.mkdir(parents=True, exist_ok=True)
    run_df = pd.read_csv(run_csv_path)
    run_df.insert(0, "run_name", run_name)
    if hist_path.exists():
        hist_df = pd.read_csv(hist_path)
        hist_df = hist_df[hist_df["run_name"] != run_name]
        merged = pd.concat([hist_df, run_df], ignore_index=True)
    else:
        merged = run_df
    merged = merged.sort_values(["run_name", "step"], kind="mergesort").reset_index(drop=True)
    merged.to_csv(hist_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one progressive TSFM prototype run.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--subset-dir",
        type=Path,
        default=DEFAULT_PROJECT_ROOT / "data" / "gifteval_pretrain_subset",
    )
    parser.add_argument("--ckpt-root", type=Path, default=DEFAULT_PROJECT_ROOT / "checkpoints")
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_PROJECT_ROOT / "runs")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_PROJECT_ROOT / "results")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--val-every", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--wandb-mode",
        choices=["online", "offline", "disabled"],
        default="online",
    )
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    cfg = TSFMConfig.from_yaml(args.config)
    orig_total_steps = cfg.total_steps
    cfg = _scale_schedule_cfg(cfg, args.max_steps)
    if args.val_every is not None:
        cfg = replace(cfg, val_every_steps=int(args.val_every))
    if args.num_workers is not None:
        cfg = replace(cfg, num_workers=int(args.num_workers))

    # Validate subset existence and build val holdouts before creating output dirs or logger state.
    store = SubsetDatasetStore(args.subset_dir, val_num_series=cfg.val_num_series)
    val_examples = store.build_validation_examples(max_examples=args.val_limit)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = _resolve_device(args.device)
    use_autocast = device.type in {"mps", "cuda"}

    ckpt_dir = args.ckpt_root / cfg.run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    cfg.to_json(ckpt_dir / "config.json")

    tb_dir = args.runs_root / cfg.run_name
    tb_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tb_dir))
    wb_run = _init_wandb(cfg, args.wandb_mode, DEFAULT_PROJECT_ROOT)

    model = TSFMModel(cfg).to(device=device, dtype=torch.float32)
    n_params = model.count_parameters()
    print(
        f"[{cfg.run_name}] device={device} params={n_params:,} c_in={cfg.c_in} "
        f"head={cfg.head_type} group_attn={cfg.use_group_attn} synth_prob={cfg.synth_prob}"
    )

    bb_decay: list[torch.nn.Parameter] = []
    bb_nodecay: list[torch.nn.Parameter] = []
    ga_decay: list[torch.nn.Parameter] = []
    ga_nodecay: list[torch.nn.Parameter] = []
    for pname, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_adapter = ("group_attn" in pname) or ("cov_role_emb" in pname)
        if param.ndim >= 2:
            (ga_decay if is_adapter else bb_decay).append(param)
        else:
            (ga_nodecay if is_adapter else bb_nodecay).append(param)

    param_groups: list[dict[str, object]] = [
        {"params": bb_decay, "weight_decay": cfg.weight_decay, "is_group_adapter": False},
        {"params": bb_nodecay, "weight_decay": 0.0, "is_group_adapter": False},
    ]
    if ga_decay:
        param_groups.append(
            {"params": ga_decay, "weight_decay": max(cfg.weight_decay, 0.05), "is_group_adapter": True}
        )
    if ga_nodecay:
        param_groups.append({"params": ga_nodecay, "weight_decay": 0.0, "is_group_adapter": True})

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.lr,
        betas=(0.9, 0.95),
    )

    dataset = _PackedStripStream(args.subset_dir, cfg)
    loader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=cfg.num_workers,
        persistent_workers=(cfg.num_workers > 0),
        prefetch_factor=(4 if cfg.num_workers > 0 else None),
        pin_memory=False,
    )
    data_iter = iter(loader)

    metrics_csv = ckpt_dir / "train_metrics.csv"
    csv_file = open(metrics_csv, "w", newline="")
    csv_writer = csv.DictWriter(
        csv_file,
        fieldnames=[
            "step",
            "split",
            "loss",
            "raw_crps",
            "cov80",
            "crossing_rate",
            "lr",
            "ms_per_step",
            "val_nmase",
            "val_nmwql",
            "val_mase",
            "val_mwql",
            "val_crps",
            "val_cov80",
        ],
    )
    csv_writer.writeheader()

    soup_set = set(cfg.soup_steps)
    snapshot_paths: list[Path] = []
    step_times_ms: list[float] = []
    running_loss = 0.0
    running_raw = 0.0
    running_cov = 0.0
    running_cross = 0.0
    log_interval = min(100, max(10, cfg.total_steps // 20))

    model.train()
    for step in range(1, cfg.total_steps + 1):
        lr_mult = _wsd_lr_multiplier(step, cfg)
        ga_mult = _group_attn_lr_multiplier(step, cfg)
        cur_lr = cfg.lr * lr_mult
        for pg in optimizer.param_groups:
            pg["lr"] = cfg.lr * (ga_mult if pg.get("is_group_adapter", False) else lr_mult)

        _sync_device(device)
        t0 = time.perf_counter()
        batch: PackedBatch = _next_collated_batch(data_iter, cfg.batch_size)

        patch_inputs = batch.patch_inputs.to(device)
        target_scaled = batch.target_scaled.to(device)
        loss_mask = batch.loss_mask.to(device)
        rel_patch_idx = batch.rel_patch_idx.to(device)
        segment_id = batch.segment_id.to(device)
        patch_obs_any = batch.patch_obs_any.to(device)
        patch_group_id = batch.patch_group_id.to(device)
        patch_is_cov = batch.patch_is_cov.to(device)
        series_id = batch.series_id.to(device)
        group_target_count = batch.group_target_count.to(device)

        optimizer.zero_grad(set_to_none=True)
        if use_autocast:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                pred_q = model(
                    patch_inputs,
                    rel_patch_idx,
                    segment_id,
                    patch_obs_any,
                    patch_group_id,
                    patch_is_cov,
                )
        else:
            pred_q = model(
                patch_inputs,
                rel_patch_idx,
                segment_id,
                patch_obs_any,
                patch_group_id,
                patch_is_cov,
            )

        stats = compute_pinball_crps_loss(
            pred_quantiles=pred_q,
            target_scaled=target_scaled,
            loss_mask=loss_mask,
            series_id=series_id,
            group_target_count=group_target_count,
            cfg=cfg,
        )
        stats.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        _sync_device(device)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        step_times_ms.append(dt_ms)

        running_loss += float(stats.loss.item())
        running_raw += float(stats.raw_crps.item())
        running_cov += float(stats.cov80.item())
        running_cross += float(stats.crossing_rate.item())

        if step % log_interval == 0 or step == cfg.total_steps:
            denom = (step % log_interval) or log_interval
            avg_loss = running_loss / denom
            avg_raw = running_raw / denom
            avg_cov = running_cov / denom
            avg_cross = running_cross / denom
            med_ms = float(np.median(step_times_ms[-denom:]))
            running_loss = running_raw = running_cov = running_cross = 0.0

            writer.add_scalar("train/loss", avg_loss, step)
            writer.add_scalar("train/raw_crps", avg_raw, step)
            writer.add_scalar("train/cov80", avg_cov, step)
            writer.add_scalar("train/crossing_rate", avg_cross, step)
            writer.add_scalar("train/lr", cur_lr, step)
            writer.add_scalar("perf/ms_per_step", med_ms, step)
            if wb_run is not None:
                wb_run.log(
                    {
                        "train/loss": avg_loss,
                        "train/raw_crps": avg_raw,
                        "train/cov80": avg_cov,
                        "train/crossing_rate": avg_cross,
                        "train/lr": cur_lr,
                        "perf/ms_per_step": med_ms,
                    },
                    step=step,
                )

            csv_writer.writerow(
                {
                    "step": step,
                    "split": "train",
                    "loss": f"{avg_loss:.6f}",
                    "raw_crps": f"{avg_raw:.6f}",
                    "cov80": f"{avg_cov:.6f}",
                    "crossing_rate": f"{avg_cross:.6f}",
                    "lr": f"{cur_lr:.6e}",
                    "ms_per_step": f"{med_ms:.2f}",
                    "val_nmase": "",
                    "val_nmwql": "",
                    "val_mase": "",
                    "val_mwql": "",
                    "val_crps": "",
                    "val_cov80": "",
                }
            )
            csv_file.flush()
            print(
                f"  step {step:5d}/{cfg.total_steps} | loss={avg_loss:.4f} raw={avg_raw:.4f} "
                f"cov80={avg_cov:.3f} lr={cur_lr:.2e} {med_ms:.1f} ms/step"
            )

        if cfg.val_every_steps > 0 and step % cfg.val_every_steps == 0 and step < cfg.total_steps:
            val_stats = evaluate_val_split(model, cfg, val_examples, device=device, batch_size=32)
            model.train()
            writer.add_scalar("val/nmase", val_stats["val_nmase"], step)
            writer.add_scalar("val/nmwql", val_stats["val_nmwql"], step)
            writer.add_scalar("val/mase", val_stats["MASE[0.5]"], step)
            writer.add_scalar("val/mwql", val_stats["mean_weighted_sum_quantile_loss"], step)
            writer.add_scalar("val/crps", val_stats["val_crps"], step)
            writer.add_scalar("val/cov80", val_stats["coverage_80"], step)
            if wb_run is not None:
                wb_run.log(
                    {
                        "val/nmase": val_stats["val_nmase"],
                        "val/nmwql": val_stats["val_nmwql"],
                        "val/mase": val_stats["MASE[0.5]"],
                        "val/mwql": val_stats["mean_weighted_sum_quantile_loss"],
                        "val/crps": val_stats["val_crps"],
                        "val/cov80": val_stats["coverage_80"],
                    },
                    step=step,
                )
            csv_writer.writerow(
                {
                    "step": step,
                    "split": "val",
                    "loss": "",
                    "raw_crps": "",
                    "cov80": "",
                    "crossing_rate": "",
                    "lr": f"{cur_lr:.6e}",
                    "ms_per_step": "",
                    "val_nmase": f"{val_stats['val_nmase']:.6f}",
                    "val_nmwql": f"{val_stats['val_nmwql']:.6f}",
                    "val_mase": f"{val_stats['MASE[0.5]']:.6f}",
                    "val_mwql": f"{val_stats['mean_weighted_sum_quantile_loss']:.6f}",
                    "val_crps": f"{val_stats['val_crps']:.6f}",
                    "val_cov80": f"{val_stats['coverage_80']:.6f}",
                }
            )
            csv_file.flush()
            print(
                f"  [val @ step {step}] nMASE={val_stats['val_nmase']:.4f} nMWQL={val_stats['val_nmwql']:.4f} | "
                f"raw MASE={val_stats['MASE[0.5]']:.4f} MWQL={val_stats['mean_weighted_sum_quantile_loss']:.4f} "
                f"CRPS={val_stats['val_crps']:.4f} Cov80={val_stats['coverage_80']:.3f}"
            )

        if step in soup_set:
            snap_name = f"step_{step}.safetensors"
            snap_path = save_checkpoint(model, cfg, ckpt_dir, filename=snap_name)
            snapshot_paths.append(snap_path)

    # Build float64 soup over cooldown snapshots; keep final step checkpoint and remove earlier snaps.
    soup_path = build_float64_soup(snapshot_paths, ckpt_dir / "model_soup.safetensors")
    last_snap = ckpt_dir / f"step_{cfg.total_steps}.safetensors"
    for p in snapshot_paths:
        if p != last_snap and p.exists():
            p.unlink()

    # Compare final step vs soup strictly on the internal 1,500-series validation split.
    last_metrics = evaluate_val_split(model, cfg, val_examples, device=device, batch_size=32)

    soup_state = load_safetensors_state(soup_path)
    model.load_state_dict(soup_state, strict=True)
    soup_metrics = evaluate_val_split(model, cfg, val_examples, device=device, batch_size=32)

    steady_ms = float(np.median(step_times_ms[min(10, len(step_times_ms) // 2) :]))
    _upsert_val_comparison_row(
        args.results_dir / "val_soup_vs_last.csv",
        {
            "run_name": cfg.run_name,
            "n_params": n_params,
            "total_steps": cfg.total_steps,
            "steady_ms_per_step": round(steady_ms, 2),
            "last_val_nmase": round(last_metrics["val_nmase"], 6),
            "soup_val_nmase": round(soup_metrics["val_nmase"], 6),
            "last_val_nmwql": round(last_metrics["val_nmwql"], 6),
            "soup_val_nmwql": round(soup_metrics["val_nmwql"], 6),
            "last_val_mase": round(last_metrics["MASE[0.5]"], 6),
            "soup_val_mase": round(soup_metrics["MASE[0.5]"], 6),
            "last_val_mwql": round(last_metrics["mean_weighted_sum_quantile_loss"], 6),
            "soup_val_mwql": round(soup_metrics["mean_weighted_sum_quantile_loss"], 6),
            "last_val_crps": round(last_metrics["val_crps"], 6),
            "soup_val_crps": round(soup_metrics["val_crps"], 6),
            "last_val_cov80": round(last_metrics["coverage_80"], 6),
            "soup_val_cov80": round(soup_metrics["coverage_80"], 6),
        },
    )

    writer.add_scalar("val_final/last_nmase", last_metrics["val_nmase"], cfg.total_steps)
    writer.add_scalar("val_final/soup_nmase", soup_metrics["val_nmase"], cfg.total_steps)
    writer.add_scalar("val_final/last_nmwql", last_metrics["val_nmwql"], cfg.total_steps)
    writer.add_scalar("val_final/soup_nmwql", soup_metrics["val_nmwql"], cfg.total_steps)
    writer.add_scalar("val_final/last_mase", last_metrics["MASE[0.5]"], cfg.total_steps)
    writer.add_scalar("val_final/soup_mase", soup_metrics["MASE[0.5]"], cfg.total_steps)
    writer.add_scalar("val_final/last_mwql", last_metrics["mean_weighted_sum_quantile_loss"], cfg.total_steps)
    writer.add_scalar("val_final/soup_mwql", soup_metrics["mean_weighted_sum_quantile_loss"], cfg.total_steps)
    writer.close()
    csv_file.close()
    if cfg.total_steps >= orig_total_steps:
        _upsert_train_history_csv(args.results_dir / "train_metrics_history.csv", cfg.run_name, metrics_csv)
    if wb_run is not None:
        wb_run.log(
            {
                "val_final/last_nmase": last_metrics["val_nmase"],
                "val_final/soup_nmase": soup_metrics["val_nmase"],
                "val_final/last_nmwql": last_metrics["val_nmwql"],
                "val_final/soup_nmwql": soup_metrics["val_nmwql"],
                "val_final/last_mase": last_metrics["MASE[0.5]"],
                "val_final/soup_mase": soup_metrics["MASE[0.5]"],
                "val_final/last_mwql": last_metrics["mean_weighted_sum_quantile_loss"],
                "val_final/soup_mwql": soup_metrics["mean_weighted_sum_quantile_loss"],
                "perf/steady_ms_per_step": steady_ms,
            },
            step=cfg.total_steps,
        )
        wb_run.finish()

    print(
        f"[{cfg.run_name}] DONE | steady {steady_ms:.1f} ms/step | "
        f"val last nMWQL={last_metrics['val_nmwql']:.4f} vs soup nMWQL={soup_metrics['val_nmwql']:.4f} "
        f"(raw MWQL last={last_metrics['mean_weighted_sum_quantile_loss']:.4f} vs "
        f"soup={soup_metrics['mean_weighted_sum_quantile_loss']:.4f})"
    )


if __name__ == "__main__":
    main()
