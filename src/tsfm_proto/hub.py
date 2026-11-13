"""Offline-first local checkpoint and Hugging Face Hub ``safetensors`` loader/saver."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from huggingface_hub import HfApi, hf_hub_download
from safetensors.torch import load_file, save_file

from tsfm_proto.config import TSFMConfig
from tsfm_proto.model import TSFMModel


def load_safetensors_state(path: Path, device: str = "cpu") -> dict[str, torch.Tensor]:
    """Load a ``safetensors`` state dictionary onto ``device``."""
    return load_file(str(path), device=device)


def save_checkpoint(
    model: TSFMModel,
    cfg: TSFMConfig,
    ckpt_dir: Path,
    filename: str = "model_soup.safetensors",
) -> Path:
    """Save model weights in ``float32`` ``safetensors`` format alongside ``config.json``."""
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    cfg.to_json(ckpt_dir / "config.json")
    state = {k: v.detach().to(device="cpu", dtype=torch.float32).contiguous() for k, v in model.state_dict().items()}
    out_path = ckpt_dir / filename
    save_file(state, str(out_path))
    return out_path


def build_float64_soup(
    snapshot_paths: list[Path],
    out_path: Path,
) -> Path:
    """Average multiple safetensors snapshots in ``float64`` and save as ``float32``."""
    if not snapshot_paths:
        raise ValueError("build_float64_soup requires at least one snapshot path")
    acc: dict[str, torch.Tensor] = {}
    for p in snapshot_paths:
        state = load_file(str(p), device="cpu")
        for k, v in state.items():
            v64 = v.to(torch.float64)
            acc[k] = v64 if k not in acc else (acc[k] + v64)
    n = float(len(snapshot_paths))
    averaged = {k: (v / n).to(torch.float32).contiguous() for k, v in acc.items()}
    save_file(averaged, str(out_path))
    return out_path


def from_pretrained(
    run_name_or_dir: str | Path,
    checkpoints_root: Path | None = None,
    hf_repo_id: str | None = None,
    filename: str = "model_soup.safetensors",
    device: str | torch.device = "cpu",
) -> tuple[TSFMModel, TSFMConfig]:
    """Load ``(TSFMModel, TSFMConfig)`` offline-first, falling back to Hugging Face Hub.

    Checks:
      1. ``run_name_or_dir`` directly if it is a directory containing ``config.json``.
      2. ``checkpoints_root / run_name_or_dir``.
      3. Remote ``hf_repo_id`` subfolder ``run_name_or_dir`` via ``hf_hub_download``.
    """
    if checkpoints_root is None:
        for root_cand in (Path("checkpoints"), Path(__file__).resolve().parents[2] / "checkpoints"):
            cand_sub = root_cand / str(run_name_or_dir)
            if (cand_sub / "config.json").exists() and (cand_sub / filename).exists():
                checkpoints_root = root_cand
                break

    cand = Path(run_name_or_dir)
    if cand.is_dir() and (cand / "config.json").exists() and (cand / filename).exists():
        ckpt_dir = cand
        cfg = TSFMConfig.from_json(ckpt_dir / "config.json")
        state = load_file(str(ckpt_dir / filename), device="cpu")
    elif (
        checkpoints_root is not None
        and (checkpoints_root / str(run_name_or_dir) / "config.json").exists()
        and (checkpoints_root / str(run_name_or_dir) / filename).exists()
    ):
        ckpt_dir = checkpoints_root / str(run_name_or_dir)
        cfg = TSFMConfig.from_json(ckpt_dir / "config.json")
        state = load_file(str(ckpt_dir / filename), device="cpu")
    elif hf_repo_id is not None:
        sub = str(run_name_or_dir)
        try:
            cfg_p = hf_hub_download(repo_id=hf_repo_id, filename=f"checkpoints/{sub}/config.json")
            wt_p = hf_hub_download(repo_id=hf_repo_id, filename=f"checkpoints/{sub}/{filename}")
        except Exception:
            cfg_p = hf_hub_download(repo_id=hf_repo_id, filename=f"{sub}/config.json")
            wt_p = hf_hub_download(repo_id=hf_repo_id, filename=f"{sub}/{filename}")
        cfg = TSFMConfig.from_json(Path(cfg_p))
        state = load_file(wt_p, device="cpu")
    else:
        raise FileNotFoundError(
            f"Could not find `{run_name_or_dir}/{filename}` locally"
            + (f" or under `{checkpoints_root}`" if checkpoints_root else "")
            + (f" or on `{hf_repo_id}`" if hf_repo_id else "")
        )

    model = TSFMModel(cfg)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model, cfg


def push_folder_to_hub(
    folder_path: Path,
    repo_id: str,
    commit_message: str = "Upload tsfm-proto checkpoints",
) -> Any:
    """Upload a pre-staged checkpoint/results folder to Hugging Face Hub."""
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    return api.upload_folder(
        folder_path=str(folder_path),
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message,
    )
