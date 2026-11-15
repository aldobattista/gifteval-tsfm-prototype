"""Stage trained checkpoints, configs, benchmark CSVs, and a Model Card for Hugging Face Hub."""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import pandas as pd

from tsfm_proto.hub import push_folder_to_hub

DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _build_model_card(repo_id: str, results_dir: Path, staged_runs: list[str]) -> str:
    summary_md = "_Run `scripts/eval_gifteval.py` to populate benchmark metrics._"
    summary_csv = results_dir / "gifteval_ablation_summary.csv"
    if summary_csv.exists():
        df = pd.read_csv(summary_csv)
        raw = df[df["postproc"] == "raw"].sort_values("run_name")
        if not raw.empty:
            cols = [c for c in ["run_name", "postproc", "geo_nmwql", "geo_nmase", "mean_cov80"] if c in df.columns]
            headers = " | ".join(cols)
            sep = " | ".join("---" for _ in cols)
            best_pp = df[df["postproc"] == "sym_avg_clamp"].sort_values("run_name")
            table_df = pd.concat([raw, best_pp], ignore_index=True) if not best_pp.empty else raw
            rows = [
                " | ".join(
                    f"{val:.6f}" if isinstance(val, float) else str(val)
                    for val in row
                )
                for row in table_df[cols].itertuples(index=False, name=None)
            ]
            summary_md = "\n".join([f"| {headers} |", f"| {sep} |", *(f"| {r} |" for r in rows)])

    runs_list = "\n".join(f"- `{r}`" for r in staged_runs)
    return f"""---
library_name: pytorch
tags:
- time-series
- forecasting
- foundation-model
- gift-eval
---

# {repo_id}

Checkpoints and `GIFT-Eval` benchmark artifacts for **`gifteval-tsfm-prototype`**, trained on a stratified `2.74 GB` compressed Parquet subset (`93` dataset directories, `96,831` series) of [`theforecastingcompany/GiftEvalPretrain`](https://huggingface.co/datasets/theforecastingcompany/GiftEvalPretrain).

## Included Runs
{runs_list}

## 97-Task GIFT-Eval Summary (Soup Checkpoints)
{summary_md}

## Artifacts Included
- `checkpoints/run*/model_soup.safetensors` & `config.json`
- `results/*.csv` (including `gifteval_ablation_summary.csv`, `gifteval_leaderboard_comparison.csv`, `train_metrics_history.csv`, and per-task CSVs)
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage and upload checkpoints + results to HF Hub.")
    parser.add_argument("--repo-id", required=True, help="Target HF Hub repo, e.g. user/gifteval-tsfm-proto")
    parser.add_argument("--ckpt-root", type=Path, default=DEFAULT_PROJECT_ROOT / "checkpoints")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_PROJECT_ROOT / "results")
    parser.add_argument("--dry-run", action="store_true", help="Validate staging directory without uploading.")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="tsfm_hf_stage_") as tmp:
        stage = Path(tmp)
        staged_runs: list[str] = []
        for run_dir in sorted(args.ckpt_root.glob("run*")):
            soup = run_dir / "model_soup.safetensors"
            cfg_json = run_dir / "config.json"
            if soup.exists() and cfg_json.exists():
                dst = stage / "checkpoints" / run_dir.name
                dst.mkdir(parents=True, exist_ok=True)
                shutil.copy2(soup, dst / "model_soup.safetensors")
                shutil.copy2(cfg_json, dst / "config.json")
                staged_runs.append(run_dir.name)

        if not staged_runs:
            raise FileNotFoundError(f"No trained run*/model_soup.safetensors found under {args.ckpt_root}")

        res_dst = stage / "results"
        res_dst.mkdir(parents=True, exist_ok=True)
        for csv_path in sorted(args.results_dir.glob("*.csv")):
            shutil.copy2(csv_path, res_dst / csv_path.name)

        card = _build_model_card(args.repo_id, args.results_dir, staged_runs)
        (stage / "README.md").write_text(card)

        if args.dry_run:
            staged_files = [str(p.relative_to(stage)) for p in sorted(stage.rglob("*")) if p.is_file()]
            print(f"[dry-run] Staged {len(staged_files)} files for {args.repo_id}: {staged_files}")
            return

        url = push_folder_to_hub(stage, repo_id=args.repo_id, commit_message="Upload tsfm-proto checkpoints")
        print(f"Uploaded to Hugging Face Hub: {url}")


if __name__ == "__main__":
    main()
