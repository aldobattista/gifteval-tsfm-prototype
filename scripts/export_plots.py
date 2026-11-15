"""Generate publication-clean PNG diagnostic and benchmark figures under results/figures/."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from PIL import Image, ImageChops

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parent.parent

_RUN_ORDER: tuple[str, ...] = (
    "run0_v0_baseline",
    "run1_data_balanced",
    "run2_missingness_suffix_cpm",
    "run3_probit_localscale_lags",
    "run4_multivariate_group_attn",
    "run5_coupled_synth_mix",
)

_RUN_LABELS: dict[str, str] = {
    "run0_v0_baseline": "Run 0: Baseline",
    "run1_data_balanced": "Run 1: Balanced",
    "run2_missingness_suffix_cpm": "Run 2: Missing+CPM",
    "run3_probit_localscale_lags": "Run 3: Probit+Lags",
    "run4_multivariate_group_attn": "Run 4: GroupAttn",
    "run5_coupled_synth_mix": "Run 5: SynthMix",
}

_RUN_COLORS: dict[str, str] = {
    "run0_v0_baseline": "#475569",
    "run1_data_balanced": "#0072B2",
    "run2_missingness_suffix_cpm": "#E69F00",
    "run3_probit_localscale_lags": "#009E73",
    "run4_multivariate_group_attn": "#7C3AED",
    "run5_coupled_synth_mix": "#B91C1C",
}

_RUN_LINESTYLES: dict[str, str] = {
    "run0_v0_baseline": "--",
    "run1_data_balanced": "-.",
    "run2_missingness_suffix_cpm": ":",
    "run3_probit_localscale_lags": "-",
    "run4_multivariate_group_attn": "--",
    "run5_coupled_synth_mix": "-",
}

_SLICE_ORDER: tuple[str, ...] = (
    "Overall (97)",
    "Term: short",
    "Term: medium",
    "Term: long",
    "Horizon <= 512",
    "Horizon > 512",
    "Univariate",
    "Multivariate",
    "Past Covariates",
    "Missing-Flagged Datasets",
    "Low-Res Freq (A/Q/M/W)",
    "High-Res Freq (D/H/Min/S)",
    "Domain: Econ/Fin",
    "Domain: Energy",
    "Domain: Healthcare",
    "Domain: Nature",
    "Domain: Sales",
    "Domain: Transport",
    "Domain: Web/CloudOps",
)


def _setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 200,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )


def plot_corpus_skew(eda_csv: Path, out_dir: Path) -> None:
    """Plot dataset series-count & byte skew alongside temperature sampling alpha curves."""
    if not eda_csv.exists():
        return
    df = pd.read_csv(eda_csv)
    counts = np.sort(df["full_num_series"].to_numpy(dtype=np.float64))[::-1]
    bytes_gb = np.sort(df["full_compressed_bytes"].to_numpy(dtype=np.float64))[::-1] / (1024**3)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4), constrained_layout=True)
    cum_series = np.cumsum(counts) / counts.sum() * 100.0
    cum_bytes = np.cumsum(bytes_gb) / bytes_gb.sum() * 100.0
    x = np.arange(1, len(counts) + 1)

    axes[0].plot(x, cum_series, lw=2.3, color="#0072B2", label="Series count (%)")
    axes[0].plot(x, cum_bytes, lw=2.3, ls="--", color="#B91C1C", label="Compressed bytes (%)")
    clim_bytes_pct = float(df.loc[df["is_climate"], "full_compressed_bytes"].sum()) / float(df["full_compressed_bytes"].sum()) * 100.0
    n_clim = int(df["is_climate"].sum())
    axes[0].annotate(
        f"Top-1 (buildings_900k)\n= {cum_series[0]:.1f}% of series",
        xy=(1, cum_series[0]),
        xytext=(13, 69),
        arrowprops={"arrowstyle": "->", "color": "#0072B2", "lw": 1.0},
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#0072B2", "alpha": 0.94},
        fontsize=8.3,
        color="#1E3A5F",
    )
    k_top_bytes = min(n_clim, len(cum_bytes) - 1)  # k=64 includes Rank-1 buildings_900k + 63 climate dirs
    axes[0].annotate(
        f"Climate dirs drive byte skew\n({n_clim} dirs = {clim_bytes_pct:.1f}% of bytes;\ntop-{k_top_bytes + 1} byte dirs = {cum_bytes[k_top_bytes]:.1f}%)",
        xy=(38, cum_bytes[37]),
        xytext=(66, 40),
        arrowprops={"arrowstyle": "->", "color": "#B91C1C", "lw": 1.1},
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": "#B91C1C", "alpha": 0.94},
        fontsize=8.0,
        color="#7F1D1D",
    )
    axes[0].set_title(f"Full GiftEvalPretrain Skew ({len(df)} Parquet Directories)")
    axes[0].set_xlabel("Top-k Directories (Sorted Independently per Metric)")
    axes[0].set_ylabel("Cumulative Share (%)")
    axes[0].set_ylim(0, 105)
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="lower right")

    alpha_styles = [
        (1.00, "#475569", "--", r"$\alpha = 1.00$ (raw proportional)"),
        (0.50, "#009E73", "-.", r"$\alpha = 0.50$"),
        (0.35, "#B91C1C", "-", r"$\alpha = 0.35$ (Runs 1–5 default)"),
        (0.00, "#0072B2", ":", r"$\alpha = 0.00$ (uniform dir)"),
    ]
    for alpha, col, ls, lbl in alpha_styles:
        w = np.power(np.maximum(counts, 1.0), alpha)
        p = w / w.sum()
        lw = 2.6 if abs(alpha - 0.35) < 1e-6 else 1.8
        axes[1].plot(x, np.cumsum(p) * 100.0, lw=lw, ls=ls, color=col, label=lbl)
    axes[1].set_title(r"Full-Corpus Temperature Mixture $p_d \propto N_d^\alpha$")
    axes[1].set_xlabel("Top-k Full-Corpus Directories (Sorted by Series Count)")
    axes[1].set_ylabel("Cumulative Sampling Probability (%)")
    axes[1].set_ylim(0, 104)
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="lower right", fontsize=8.5)

    fig.savefig(out_dir / "fig1_corpus_skew_and_sampling.png")
    plt.close(fig)


def _load_training_history(ckpt_root: Path, results_dir: Path) -> pd.DataFrame:
    """Load training history from committed CSV and overlay any local checkpoint telemetry."""
    by_run: dict[str, pd.DataFrame] = {}
    hist_csv = results_dir / "train_metrics_history.csv"
    if hist_csv.exists():
        hist_df = pd.read_csv(hist_csv)
        for rname, grp in hist_df.groupby("run_name", sort=False):
            by_run[str(rname)] = grp.copy()
    for p in sorted(ckpt_root.glob("run*/train_metrics.csv")):
        sub = pd.read_csv(p)
        sub["run_name"] = p.parent.name
        by_run[p.parent.name] = sub
    if not by_run:
        return pd.DataFrame()
    return pd.concat(list(by_run.values()), ignore_index=True)


def plot_training_curves(ckpt_root: Path, results_dir: Path, out_dir: Path) -> None:
    """Plot smoothed train CRPS with late-step inset and held-out validation nMWQL + soup stars."""
    hist_df = _load_training_history(ckpt_root, results_dir)
    if hist_df.empty:
        return

    val_cmp_path = results_dir / "val_soup_vs_last.csv"
    val_cmp = pd.read_csv(val_cmp_path).set_index("run_name") if val_cmp_path.exists() else pd.DataFrame()

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.9), constrained_layout=True)
    inset_ax = axes[0].inset_axes([0.44, 0.52, 0.53, 0.43])

    has_nmwql = False
    star_offsets = np.linspace(-1150.0, 1150.0, len(_RUN_ORDER))
    for r_idx, rname in enumerate(_RUN_ORDER):
        sub = hist_df[hist_df["run_name"] == rname]
        if sub.empty:
            continue
        col = _RUN_COLORS.get(rname, "#0072B2")
        ls = _RUN_LINESTYLES.get(rname, "-")
        lbl = _RUN_LABELS.get(rname, rname)
        tr = sub[sub["split"] == "train"].copy()
        va = sub[sub["split"] == "val"].copy()

        if not tr.empty:
            loss_smooth = tr["raw_crps"].astype(float).rolling(5, min_periods=1).mean()
            steps = tr["step"].astype(float)
            keep_main = steps >= 600
            axes[0].plot(steps[keep_main], loss_smooth[keep_main], lw=1.7, ls=ls, color=col, label=lbl)
            late = steps >= 8000
            inset_ax.plot(steps[late], loss_smooth[late], lw=1.4, ls=ls, color=col)

        if not va.empty and "val_nmwql" in va.columns and va["val_nmwql"].notna().any():
            has_nmwql = True
            v_steps = va["step"].astype(float).tolist()
            v_vals = va["val_nmwql"].astype(float).tolist()
            if not val_cmp.empty and rname in val_cmp.index:
                last_val = float(val_cmp.loc[rname, "last_val_nmwql"])
                soup_val = float(val_cmp.loc[rname, "soup_val_nmwql"])
                x_star = 22800.0 + float(star_offsets[r_idx])
                v_steps.append(20000.0)
                v_vals.append(last_val)
                axes[1].plot([20000.0, x_star], [last_val, soup_val], ls=":", lw=1.2, color=col, alpha=0.85, zorder=4)
                axes[1].scatter(
                    [x_star],
                    [soup_val],
                    marker="*",
                    s=95,
                    color=col,
                    edgecolors="#1A202C",
                    linewidths=0.6,
                    zorder=6,
                )
            axes[1].plot(v_steps, v_vals, marker="o", ms=3.8, lw=1.7, ls=ls, color=col, label=lbl)

    axes[0].set_title("Training 2x Pinball CRPS (On Each Run's Own Mixture)")
    axes[0].set_xlabel("Training Step (Steps 600–20k)")
    axes[0].set_ylabel("Raw 2x Pinball Loss")
    axes[0].set_ylim(0.28, 0.78)
    axes[0].grid(alpha=0.25)
    axes[0].text(
        0.02,
        0.95,
        "Step-100 CRPS:\nR0–2 Softplus ~3.55\nR3–5 Probit ~0.67",
        transform=axes[0].transAxes,
        va="top",
        fontsize=7.3,
        color="#334155",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#CBD5E1", "alpha": 0.90},
    )
    axes[0].text(
        0.98,
        0.03,
        "Higher level = harder mixture (α=0.35, +30% blk, H≤1024, +15% synth)",
        transform=axes[0].transAxes,
        ha="right",
        va="bottom",
        fontsize=7.4,
        color="#475569",
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "none", "alpha": 0.82},
    )
    inset_ax.set_title("Steps 8k–20k (Zoom)", fontsize=8)
    inset_ax.set_ylim(0.32, 0.50)
    inset_ax.tick_params(labelsize=7.5)
    inset_ax.grid(alpha=0.25)

    if has_nmwql:
        axes[1].axhline(1.0, color="#64748B", ls="--", lw=1.1, label="Seasonal Naive = 1.0")
        axes[1].axvline(15000.0, color="#94A3B8", ls=":", lw=1.2)
        axes[1].text(15150.0, 0.96, "WSD Cooldown (15k)", fontsize=7.6, color="#475569", va="top")
        axes[1].scatter(
            [],
            [],
            marker="*",
            s=95,
            color="#F8FAFC",
            edgecolors="#0F172A",
            linewidths=1.0,
            label="5-Snap Soup (16k–20k)",
        )
        axes[1].set_title("Held-Out Internal Validation (66 Datasets, Geometric Mean nMWQL)")
        axes[1].set_ylabel("Validation nMWQL (Lower Better)")
        axes[1].set_ylim(0.49, 1.04)
        axes[1].set_xlim(1500, 24800)
        axes[1].set_xticks([2000, 6000, 10000, 15000, 20000, 22800])
        axes[1].set_xticklabels(["2k", "6k", "10k", "15k", "20k", "Soup"])
    else:
        axes[1].set_title("Held-Out Internal Validation")
    axes[1].set_xlabel("Training Step (Circles = Checkpoint Steps; Stars = 5-Snap Soup)")
    axes[1].grid(alpha=0.25)

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=4,
        fontsize=8.5,
    )
    fig.savefig(out_dir / "fig2_training_and_val_curves.png")
    plt.close(fig)


def plot_ablation_waterfall(summary_csv: Path, out_dir: Path) -> None:
    """Plot 97-task normalized geometric-mean MWQL and MASE progression across Runs 0..5 + SymAvg."""
    if not summary_csv.exists():
        return
    df = pd.read_csv(summary_csv)
    raw = df[df["postproc"] == "raw"].copy()
    if raw.empty:
        return
    order_idx = {r: i for i, r in enumerate(_RUN_ORDER)}
    raw["_ord"] = raw["run_name"].map(lambda r: order_idx.get(str(r), 99))
    raw = raw.sort_values("_ord").drop(columns=["_ord"]).reset_index(drop=True)

    sym_r5 = df[(df["run_name"] == "run5_coupled_synth_mix") & (df["postproc"] == "sym_avg_clamp")]
    plot_rows = raw[["run_name", "geo_nmwql", "geo_nmase"]].copy()
    plot_rows["label"] = [
        _RUN_LABELS.get(str(r), str(r)).replace(": ", "\n") for r in plot_rows["run_name"]
    ]
    plot_rows["color"] = [_RUN_COLORS.get(str(r), "#2B6CB0") for r in plot_rows["run_name"]]
    if not sym_r5.empty:
        r5_row = sym_r5.iloc[0]
        plot_rows = pd.concat(
            [
                plot_rows,
                pd.DataFrame(
                    [
                        {
                            "run_name": "run5_sym_avg_clamp",
                            "geo_nmwql": float(r5_row["geo_nmwql"]),
                            "geo_nmase": float(r5_row["geo_nmase"]),
                            "label": "Run 5\n+SymAvg+Clamp",
                            "color": "#991B1B",
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 4.9), constrained_layout=True)
    x = np.arange(len(plot_rows))

    configs = [
        (0, "nmwql", "geo_nmwql", (0.53, 0.69), "97-Task GIFT-Eval nMWQL (Bars = Run Geometric Means)", "Geometric Mean nMWQL (SN = 1.0)"),
        (1, "nmase", "geo_nmase", (0.80, 0.955), "97-Task GIFT-Eval nMASE (Bars = Run Geometric Means)", "Geometric Mean nMASE (SN = 1.0)"),
    ]
    for ax_idx, metric_col, geo_col, ylims, title, ylabel in configs:
        ax = axes[ax_idx]
        bars = ax.bar(
            x,
            plot_rows[geo_col],
            color=plot_rows["color"],
            alpha=0.90,
            width=0.68,
        )
        if len(bars) > 6:
            bars[-1].set_hatch("//")
            bars[-1].set_edgecolor("#450A0A")

        if metric_col == "nmwql":
            ref_lines = [0.6521, 0.5989]
            ax.axhline(0.6521, color="#64748B", ls=":", lw=1.2, label="Chronos-Base (200M) = 0.652")
            ax.axhline(0.5989, color="#1F2937", ls="--", lw=1.2, label="Moirai-Large (311M) = 0.599")
            ax.legend(loc="upper right", fontsize=8.2)
        else:
            ref_lines = [0.8932, 0.8749]
            ax.axhline(0.8932, color="#64748B", ls=":", lw=1.2, label="iTransformer = 0.893")
            ax.axhline(0.8749, color="#1F2937", ls="--", lw=1.2, label="Moirai-Large (311M) = 0.875")
            ax.legend(loc="upper right", fontsize=8.2)

        for idx, val in enumerate(plot_rows[geo_col]):
            val_f = float(val)
            tight_under_ref = any(0.0 <= (ref_y - val_f) < 0.0065 for ref_y in ref_lines)
            if tight_under_ref:
                ax.text(
                    x[idx],
                    val_f - 0.0022,
                    f"{val_f:.4f}",
                    ha="center",
                    va="top",
                    fontsize=8.2,
                    fontweight="bold",
                    color="white",
                    zorder=5,
                )
            else:
                ax.text(
                    x[idx],
                    val_f + 0.0016,
                    f"{val_f:.4f}",
                    ha="center",
                    va="bottom",
                    fontsize=8.2,
                    fontweight="bold" if idx in (0, len(plot_rows) - 1) else "normal",
                    color="#1A202C",
                    bbox={"facecolor": "white", "alpha": 0.92, "edgecolor": "none", "pad": 0.35},
                    zorder=5,
                )
        ax.set_xticks(x)
        ax.set_xticklabels(plot_rows["label"], fontsize=8.2)
        ax.set_ylim(*ylims)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)

    fig.savefig(out_dir / "fig3_ablation_waterfall.png")
    plt.close(fig)


def plot_slices_and_blackout(slices_csv: Path, blackout_csv: Path, out_dir: Path) -> None:
    """Plot disaggregated GIFT-Eval capability slices and pre-origin blackout stress deltas."""
    if slices_csv.exists():
        sdf = pd.read_csv(slices_csv)
        sdf = sdf[sdf["postproc"] == "raw"]
        if not sdf.empty:
            n_map = sdf.groupby("slice")["n_tasks"].first().to_dict()
            piv = sdf.pivot(index="slice", columns="run_name", values="geo_nmwql")
            ordered_idx = [s for s in _SLICE_ORDER if s in piv.index] + [
                s for s in piv.index if s not in _SLICE_ORDER
            ]
            cols = [r for r in _RUN_ORDER if r in piv.columns]
            piv = piv.reindex(index=ordered_idx, columns=cols)

            fig, ax = plt.subplots(figsize=(14.4, 5.6), constrained_layout=True)
            bar_colors = [_RUN_COLORS.get(r, "#2B6CB0") for r in cols]
            piv.plot(kind="bar", ax=ax, width=0.80, color=bar_colors)
            ax.axhline(1.0, color="#718096", ls="--", lw=1.1, label="Seasonal Naive = 1.0")
            for xv in (0.5, 3.5, 5.5, 9.5, 11.5):
                ax.axvline(xv, color="#CBD5E1", lw=0.8, ls=":", zorder=1)
            tick_labels = [
                "Overall (n=97)"
                if str(s) == "Overall (97)"
                else f"{s} (n={int(n_map[s])})"
                for s in piv.index
            ]
            ax.set_xticklabels(tick_labels, rotation=32, ha="right", fontsize=8.5)
            ax.set_title("Disaggregated GIFT-Eval Slices (Geometric Mean nMWQL, Raw Neural)", pad=28)
            ax.set_ylabel("Geometric Mean nMWQL (Lower Better)")
            ax.set_xlabel("")
            ax.grid(axis="y", alpha=0.25)
            handles, labels = ax.get_legend_handles_labels()
            clean_labels = [_RUN_LABELS.get(lbl, lbl) for lbl in labels]
            ax.legend(
                handles,
                clean_labels,
                loc="lower center",
                bbox_to_anchor=(0.5, 1.02),
                ncol=7,
                fontsize=8.2,
            )
            fig.savefig(out_dir / "fig4_disaggregated_slices.png")
            plt.close(fig)

    if blackout_csv.exists():
        bdf = pd.read_csv(blackout_csv)
        if not bdf.empty:
            piv_b = bdf.pivot(index="run_name", columns="blackout_steps", values="geo_nmwql")
            rows = [r for r in _RUN_ORDER if r in piv_b.index]
            piv_b = piv_b.reindex(index=rows)

            fig, ax = plt.subplots(figsize=(10.6, 4.6), constrained_layout=True)
            b_colors = ["#0072B2", "#E69F00", "#B91C1C"][: piv_b.shape[1]]
            bars = piv_b.plot(kind="bar", ax=ax, width=0.72, color=b_colors)
            ax.axhline(1.0, color="#64748B", ls="--", lw=1.2, label="Point Seasonal Naive = 1.0")
            ax.set_xticklabels([_RUN_LABELS.get(r, r).replace(": ", "\n") for r in piv_b.index], rotation=0, fontsize=8.8)
            ax.set_title("90-Task Pre-Origin Blackout Stress Test (0 vs 20 vs 64 Missing Steps Before Origin)", pad=26)
            ax.set_ylabel("Point-SN Normalized MWQL (Lower Better)")
            ax.set_xlabel("")
            ax.set_ylim(0.0, max(1.21, float(piv_b.to_numpy().max()) * 1.18))
            ax.grid(axis="y", alpha=0.25)

            for c_idx, container in enumerate(bars.containers):
                for rect in container:
                    h = float(rect.get_height())
                    if np.isfinite(h):
                        if 0.92 <= h < 1.015:
                            ax.text(
                                rect.get_x() + rect.get_width() / 2.0,
                                h - 0.032,
                                f"{h:.2f}",
                                ha="center",
                                va="top",
                                fontsize=7.4,
                                fontweight="bold",
                                color="#1A202C" if c_idx == 1 else "white",
                                zorder=5,
                            )
                        else:
                            ax.text(
                                rect.get_x() + rect.get_width() / 2.0,
                                h + 0.015,
                                f"{h:.2f}",
                                ha="center",
                                va="bottom",
                                fontsize=7.4,
                                color="#1A202C",
                                bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none", "pad": 0.6},
                                zorder=5,
                            )

            ax.annotate(
                "0.6P intra-patch\nzero-fill cliff",
                xy=(0.0, 1.086),
                xytext=(0.56, 1.13),
                arrowprops={"arrowstyle": "->", "color": "#B45309", "lw": 0.9},
                fontsize=7.6,
                color="#78350F",
                ha="left",
                va="center",
            )

            handles, labels = ax.get_legend_handles_labels()
            step_map = {"0": "0 missing steps", "20": "20 steps (0.6P)", "64": "64 steps (2.0P)"}
            clean_labels = [step_map.get(str(lbl), str(lbl)) for lbl in labels]
            ax.legend(
                handles,
                clean_labels,
                loc="lower center",
                bbox_to_anchor=(0.5, 1.02),
                ncol=4,
                fontsize=8.5,
            )
            fig.savefig(out_dir / "fig5_blackout_stress.png")
            plt.close(fig)


_ARCHITECTURE_TIKZ_TEX = r"""\documentclass[border=8pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage[scaled=0.95]{helvet}
\renewcommand{\familydefault}{\sfdefault}
\usepackage{amsmath,amssymb,etoolbox,xcolor,tikz}
\usetikzlibrary{arrows.meta,calc,decorations.pathreplacing,patterns}

\definecolor{tfcink}{HTML}{171717}
\definecolor{tfcbody}{HTML}{3D3D3D}
\definecolor{tfcmuted}{HTML}{757575}
\definecolor{tfchair}{HTML}{E3E3E3}
\definecolor{tfcsurface}{HTML}{F5F5F5}
\definecolor{tfcgreen}{HTML}{268248}
\definecolor{tfcgreenstroke}{HTML}{259D51}
\definecolor{tfcband}{HTML}{ECF8F0}
\definecolor{roley}{HTML}{346CB2}
\definecolor{roleyy}{HTML}{B4512D}
\definecolor{rolex}{HTML}{525B66}

\tikzset{
  archblock/.style={draw=tfchair, fill=white, rounded corners=2pt,
    line width=0.5pt, inner sep=2pt, align=center, font=\scriptsize,
    text=tfcink},
  attnblock/.style={archblock, fill=tfcband, draw=tfcgreenstroke},
  archpanel/.style={draw=tfchair, fill=tfcsurface, rounded corners=3pt,
    line width=0.5pt},
  smallblock/.style={archblock, font=\tiny, inner sep=1.2pt,
    minimum height=0.28cm},
  flow/.style={-{Stealth[length=1.4mm, width=1.3mm]}, line width=0.7pt,
    tfcbody},
  paneltitle/.style={font=\scriptsize\bfseries, text=tfcink},
  sublabel/.style={font=\scriptsize, text=tfcbody, inner sep=1pt},
  rowlabel/.style={font=\tiny, anchor=east, inner sep=1pt, align=right},
  note/.style={font=\tiny, text=tfcmuted, align=left},
  tok/.style={circle, draw, line width=0.5pt, minimum size=2.4mm,
    inner sep=0pt},
  blanktok/.style={tok, draw=tfcmuted, densely dotted, fill=white},
  patchbox/.style={draw=tfchair, line width=0.4pt, fill=white},
  blankpatch/.style={draw=tfcmuted, densely dotted, line width=0.5pt,
    fill=white},
  attnarrow/.style={-{Stealth[length=1.1mm, width=1mm]}, line width=0.5pt},
  bothways/.style={{Stealth[length=1.1mm, width=1mm]}-{Stealth[length=1.1mm,
    width=1mm]}, line width=0.5pt},
  seriesbox/.style={draw=tfcmuted, dashed, rounded corners=2pt,
    line width=0.4pt},
  seriesbrace/.style={decorate, decoration={brace, amplitude=2.5pt},
    tfcmuted, line width=0.4pt},
}

\newcommand{\wave}[3]{0.09*sin((#3)*#2 + #1) + 0.05*sin((#3)*#2*2.3 + #1*1.7)}

\newcommand{\rowwidth}{2.4}
\newcommand{\ctxwidth}{1.68}

\newcommand{\miniseries}[6]{%
  \begin{scope}
    \draw[tfchair, line width=0.4pt, fill=white]
      (#1, #2-0.18) rectangle (#1+\rowwidth, #2+0.18);
    \clip (#1, #2-0.18) rectangle (#1+\rowwidth, #2+0.18);
    \draw[#3, line width=0.6pt, smooth, samples=36, domain=0:\ctxwidth]
      plot ({\x+#1}, {#2 + \wave{#5}{#6}{\x}});
    \ifstrequal{#4}{blank}{%
      \draw[tfcmuted, densely dotted, line width=0.5pt]
        (#1+\ctxwidth, #2) -- (#1+\rowwidth, #2);}{}
  \end{scope}
}

\newcommand{\forecastrow}[5]{%
  \begin{scope}
    \draw[tfchair, line width=0.4pt, fill=white]
      (#1, #2-0.18) rectangle (#1+\rowwidth, #2+0.18);
    \clip (#1, #2-0.18) rectangle (#1+\rowwidth, #2+0.18);
    \draw[#3!35, line width=0.5pt, smooth, samples=36, domain=0:\ctxwidth]
      plot ({\x+#1}, {#2 + \wave{#4}{#5}{\x}});
    \fill[#3, fill opacity=0.22]
      plot[smooth, samples=18, domain=\ctxwidth:\rowwidth]
        ({\x+#1}, {#2 + \wave{#4}{#5}{\x} + 0.04 + 0.14*(\x-\ctxwidth)})
      -- plot[smooth, samples=18, domain=\rowwidth:\ctxwidth]
        ({\x+#1}, {#2 + \wave{#4}{#5}{\x} - 0.04 - 0.14*(\x-\ctxwidth)})
      -- cycle;
    \draw[#3, dashed, line width=0.6pt, smooth, samples=18,
          domain=\ctxwidth:\rowwidth]
      plot ({\x+#1}, {#2 + \wave{#4}{#5}{\x}});
  \end{scope}
}

\newcommand{\emptyrow}[2]{%
  \draw[tfchair, line width=0.4pt, fill=white]
    (#1, #2-0.18) rectangle (#1+\rowwidth, #2+0.18);
}

\newcommand{\packx}[1]{0.6 + (#1)*0.275}
\newcommand{\packy}[1]{-4.20 - (#1)*0.357}
\newcommand{\prow}[4]{%
  \fill[#4!22] ({\packx{#2}}, {\packy{#1}}) rectangle
    ({\packx{#3}}, {\packy{#1+1}});
}
\newcommand{\pbox}[4]{%
  \draw[tfcbody, line width=0.5pt] ({\packx{#3}}, {\packy{#1}}) rectangle
    ({\packx{#4}}, {\packy{#2+1}});
}

\newcommand{\tokn}[4]{\node[tok, draw=#4, fill=#4!22] (#1) at (#2, #3) {};}

\newcommand{\rolevec}[4]{%
  \fill[#3!#4] (#1-0.25, #2-0.055) rectangle (#1+0.25, #2+0.055);
  \draw[#3, line width=0.4pt] (#1-0.25, #2-0.055) rectangle (#1+0.25, #2+0.055);
  \foreach \k in {1,...,4}
    {\draw[#3, line width=0.3pt] (#1-0.25+\k*0.1, #2-0.055) -- (#1-0.25+\k*0.1, #2+0.055);}
}

\newcommand{\cpmpatch}[5]{%
  \begin{scope}
    \edef\cpmkind{#3}%
    \ifdefstring{\cpmkind}{blank}{%
      \draw[blankpatch] (#1, #2-0.18) rectangle (#1+0.64, #2+0.18);}{%
      \draw[patchbox] (#1, #2-0.18) rectangle (#1+0.64, #2+0.18);}
    \clip (#1, #2-0.18) rectangle (#1+0.64, #2+0.18);
    \ifdefstring{\cpmkind}{obs}{%
      \draw[#5, line width=0.6pt, smooth, samples=14, domain=0:0.64]
        plot ({\x+#1}, {#2 + \wave{30}{120}{\x+#4}});}{}
    \ifdefstring{\cpmkind}{pred}{%
      \fill[#5, fill opacity=0.22]
        plot[smooth, samples=10, domain=0:0.64]
          ({\x+#1}, {#2 + \wave{30}{120}{\x+#4} + 0.05 + 0.08*\x})
        -- plot[smooth, samples=10, domain=0.64:0]
          ({\x+#1}, {#2 + \wave{30}{120}{\x+#4} - 0.05 - 0.08*\x})
        -- cycle;
      \draw[#5, dashed, line width=0.6pt, smooth, samples=14, domain=0:0.64]
        plot ({\x+#1}, {#2 + \wave{30}{120}{\x+#4}});}{}
  \end{scope}
}

\begin{document}
\begin{tikzpicture}[x=1cm, y=1cm]

% --- (a) MODEL ARCHITECTURE ---
\node[paneltitle, anchor=south] at (7.80, 3.60) {(a) \texttt{TSFMModel} ARCHITECTURE (16.32M PARAMS, RUN 5)};
\node[paneltitle, anchor=south] at (2.00, 3.05) {INPUT};

\node[rowlabel, text=roley]  at (0.74, 2.52) {$y$};
\node[rowlabel, text=rolex]  at (0.74, 1.88) {$x$};
\node[rowlabel, text=roley]  at (0.74, 1.22) {$y'_1$};
\node[rowlabel, text=roleyy] at (0.74, 0.58) {$y'_2$};

\draw[seriesbrace] (0.30, 1.68) -- (0.30, 2.72);
\node[note, rotate=90, anchor=south] at (0.16, 2.20) {group $g$};
\draw[seriesbrace] (0.30, 0.38) -- (0.30, 1.42);
\node[note, rotate=90, anchor=south] at (0.16, 0.90) {group $g'$};

\miniseries{0.80}{2.52}{roley}{blank}{40}{210}
\miniseries{0.80}{1.88}{rolex, dashed}{none}{160}{250}
\miniseries{0.80}{1.22}{roley}{blank}{95}{275}
\miniseries{0.80}{0.58}{roleyy}{blank}{200}{235}

\draw[tfcgreenstroke, line width=0.55pt] (2.48, 0.35) -- (2.48, 2.75);
\node[font=\tiny, text=tfcgreen, anchor=south, inner sep=1pt] at (2.48, 2.77) {$T_{\text{orig}}$};
\node[note, anchor=north, inner sep=1pt] at (1.64, 0.32) {context};
\node[note, anchor=north, inner sep=1pt] at (2.84, 0.32) {horizon};

\draw[flow] (3.20, 1.55) -- (3.45, 1.55);
\node[archblock, minimum width=2.4cm, minimum height=0.65cm, rotate=90]
  at (3.775, 1.55) {Obs. \texttt{asinh}\\scale};

\draw[flow] (4.10, 1.55) -- (4.35, 1.55);
\draw[archpanel] (4.35, 0.35) rectangle (6.35, 2.75);
\node[paneltitle, anchor=south] at (5.35, 3.05) {EMBEDDING};
\node[archblock, minimum width=2.16cm, minimum height=0.8cm, rotate=90]
  at (4.87, 1.55) {Mod-32\\patching};
\node[archblock, minimum width=2.16cm, minimum height=0.8cm, rotate=90]
  at (5.83, 1.55) {17-ch\\ResStem};

\draw[flow] (6.35, 1.55) -- (6.60, 1.55);
\draw[archpanel] (6.60, 0.35) rectangle (9.80, 2.75);
\node[paneltitle, anchor=south] at (8.20, 3.05) {TRANSFORMER STACK};
\node[attnblock, minimum width=1.6cm, minimum height=0.92cm, rotate=90]
  at (7.18, 1.78) {Temporal\\attn (RoPE)};
\node[attnblock, minimum width=1.6cm, minimum height=0.92cm, rotate=90]
  at (8.20, 1.78) {Group attn\\(L3, L7)};
\node[archblock, minimum width=1.6cm, minimum height=0.92cm, rotate=90]
  at (9.22, 1.78) {SwiGLU\\FFN};
\draw[decorate, decoration={brace, amplitude=3pt, mirror}, tfcbody,
  line width=0.5pt] (6.72, 0.88) -- (9.68, 0.88);
\node[font=\tiny, text=tfcink, anchor=north, inner sep=1pt]
  at (8.20, 0.74) {$\times\,8$ layers ($d{=}384$, 6h)};

\draw[flow] (9.80, 1.55) -- (10.05, 1.55);
\draw[archpanel] (10.05, 0.35) rectangle (11.99, 2.75);
\node[paneltitle, anchor=south] at (11.02, 3.05) {DECODER};
\node[archblock, minimum width=2.16cm, minimum height=0.8cm, rotate=90]
  at (10.57, 1.55) {SiLU\\ResMLP};
\node[archblock, minimum width=2.16cm, minimum height=0.8cm, rotate=90]
  at (11.47, 1.55) {Probit Q21\\head};

\draw[flow] (11.99, 1.55) -- (12.24, 1.55);
\node[archblock, minimum width=2.4cm, minimum height=0.65cm, rotate=90]
  at (12.565, 1.55) {Exact \texttt{sinh}\\+ clamp};

\draw[flow] (12.89, 1.55) -- (13.14, 1.55);
\node[paneltitle, anchor=south] at (14.34, 3.05) {FORECAST};
\forecastrow{13.14}{2.52}{roley}{40}{210}
\emptyrow{13.14}{1.88}
\forecastrow{13.14}{1.22}{roley}{95}{275}
\forecastrow{13.14}{0.58}{roleyy}{200}{235}
\draw[tfcgreenstroke, line width=0.55pt] (14.82, 0.35) -- (14.82, 2.75);
\node[font=\tiny, text=tfcgreen, anchor=south, inner sep=1pt] at (14.82, 2.77) {$T_{\text{orig}}$};
\node[note, anchor=north, inner sep=1pt] at (15.18, 0.32) {$H{\le}1{,}024$};

\draw[tfcmuted, densely dotted, line width=0.4pt] (0, -0.25) -- (15.54, -0.25);

% --- (b) 17-CH PATCH EMBEDDING ---
\node[paneltitle, anchor=north] at (2.80, -0.45) {(b) 17-CH PATCH EMBEDDING};
\begin{scope}[yshift=-0.25cm]
\node[rowlabel, text=tfcbody] at (0.88, -1.25) {val + loc};
\node[rowlabel, text=tfcbody] at (0.88, -1.85) {mask + pos};
\node[rowlabel, text=tfcbody] at (0.88, -2.45) {6 lags+ok};
\foreach \yc in {-1.25, -1.85, -2.45} {
  \draw[tfcbody, line width=0.4pt]
    (0.99, \yc+0.15) -| (0.92, \yc-0.15) -- (0.99, \yc-0.15);
  \draw[tfcbody, line width=0.4pt]
    (1.69, \yc+0.15) -| (1.76, \yc-0.15) -- (1.69, \yc-0.15);
}
\draw[roley, line width=0.6pt, smooth, samples=22, domain=0:0.68]
  plot ({\x+1.0}, {-1.25 + 0.7*(\wave{40}{600}{\x})});
\draw[tfcbody, line width=0.5pt] (1.0, -1.96) -- (1.68, -1.74);
\draw[tfcbody, line width=0.5pt]
  (1.0, -2.37) -- (1.46, -2.37) -- (1.46, -2.53) -- (1.68, -2.53);
\node[note, anchor=north, inner sep=1pt] at (1.34, -2.68) {$17{\times}32$ ($P{=}32$)};

\fill[tfcbody] (2.02, -1.85) circle (0.9pt);
\draw[tfcbody, line width=0.5pt] (1.79, -1.25) -- (2.02, -1.85);
\draw[tfcbody, line width=0.5pt] (1.79, -1.85) -- (2.02, -1.85);
\draw[tfcbody, line width=0.5pt] (1.79, -2.45) -- (2.02, -1.85);

\draw[seriesbox] (2.14, -1.60) rectangle (4.02, -2.48);
\node[note, anchor=south west, inner sep=1pt] at (2.14, -1.58) {residual stem};
\node[smallblock, minimum width=0.44cm] (lin1) at (2.42, -1.85) {lin};
\node[smallblock, minimum width=0.38cm] (silu) at (2.98, -1.85) {SiLU};
\node[smallblock, minimum width=0.44cm] (lin2) at (3.54, -1.85) {lin};
\node[smallblock, minimum width=0.48cm] (skip) at (2.98, -2.27) {skip};
\node[circle, draw=tfcbody, line width=0.5pt, inner sep=0pt,
  minimum size=2.1mm, font=\tiny] (plus1) at (4.18, -1.85) {$+$};
\node[smallblock, minimum width=0.56cm] (rmsn) at (4.68, -1.85) {Norm};
\node[circle, draw=tfcbody, line width=0.5pt, inner sep=0pt,
  minimum size=2.1mm, font=\tiny] (plus2) at (5.18, -1.85) {$+$};

\draw[attnarrow, tfcbody] (2.02, -1.85) -- (lin1.west);
\draw[attnarrow, tfcbody] (lin1.east) -- (silu.west);
\draw[attnarrow, tfcbody] (silu.east) -- (lin2.west);
\draw[attnarrow, tfcbody] (lin2.east) -- (plus1.west);
\draw[tfcbody, line width=0.5pt] (2.02, -1.85) |- (skip.west);
\draw[attnarrow, tfcbody] (skip.east) -| (plus1.south);
\draw[attnarrow, tfcbody] (plus1.east) -- (rmsn.west);
\draw[attnarrow, tfcbody] (rmsn.east) -- (plus2.west);

\node[note, anchor=south, inner sep=1pt] at (5.18, -1.06) {role emb};
\node[rowlabel, text=roley] at (4.88, -1.24) {$y$};
\node[rowlabel, text=rolex] at (4.88, -1.42) {$x$};
\rolevec{5.18}{-1.24}{roley}{12}
\rolevec{5.18}{-1.42}{rolex}{48}
\node[font=\tiny, text=roley] at (5.54, -1.24) {$0$};
\draw[attnarrow, tfcbody] (5.18, -1.49) -- (plus2.north);

\node[tok, draw=roley, fill=roley!22] (token) at (5.62, -1.85) {};
\draw[attnarrow, tfcbody] (plus2.east) -- (token.west);
\node[note, anchor=north, inner sep=1pt] at (5.62, -2.02) {$384$};
\end{scope}

% --- (c) ORIGIN-ANCHORED TIME ATTENTION ---
\node[paneltitle, anchor=north] at (8.35, -0.45) {(c) TIME ATTENTION (CAUSAL + OBS-KEY)};
\begin{scope}[yshift=-0.25cm]
\fill[tfcsurface] (8.76, -1.02) rectangle (9.80, -2.56);
\draw[tfcgreenstroke, line width=0.45pt] (8.76, -1.02) -- (8.76, -2.56);
\node[note, anchor=north, inner sep=1pt] at (9.27, -2.60) {horizon ($>0$)};
\node[note, anchor=north, inner sep=1pt] at (7.47, -2.60) {context ($\le 0$)};

\node[rowlabel, text=roley] at (6.46, -1.35) {$y$};
\node[rowlabel, text=rolex] at (6.46, -2.10) {$x$};

\tokn{ty0}{6.72}{-1.35}{roley}
\node[blanktok] (ty1) at (7.22, -1.35) {};
\tokn{ty2}{7.72}{-1.35}{roley}
\tokn{ty3}{8.22}{-1.35}{roley}
\node[blanktok, draw=roley, line width=0.9pt] (ty4) at (9.02, -1.35) {};
\node[blanktok] (ty5) at (9.52, -1.35) {};

\tokn{tx0}{6.72}{-2.10}{rolex}
\tokn{tx1}{7.22}{-2.10}{rolex}
\tokn{tx2}{7.72}{-2.10}{rolex}
\node[tok, draw=rolex, fill=rolex!22, line width=0.9pt] (tx3) at (8.22, -2.10) {};
\node[blanktok] (tx4) at (9.02, -2.10) {};
\node[blanktok] (tx5) at (9.52, -2.10) {};

\draw[attnarrow, roley] (ty0.north) to[bend left=42] (ty4.north);
\draw[attnarrow, roley] (ty2.north) to[bend left=38] (ty4.north);
\draw[attnarrow, roley] (ty3.north) to[bend left=34] (ty4.north);

\draw[attnarrow, rolex] (tx0.north) to[bend left=42] (tx3.north);
\draw[attnarrow, rolex] (tx1.north) to[bend left=38] (tx3.north);
\draw[attnarrow, rolex] (tx2.north) to[bend left=34] (tx3.north);

\node[note, anchor=north, align=center] at (8.25, -2.86)
  {skip blanked/NaN keys; $\texttt{eye\_l}$ 100\%-NaN guard};
\end{scope}

% --- (d) GROUP ATTENTION (BLOCKS 3 & 7) ---
\node[paneltitle, anchor=north] at (13.15, -0.45) {(d) GROUP ATTENTION (L3, L7)};
\begin{scope}[yshift=-0.25cm]
\fill[tfcsurface] (13.06, -1.02) rectangle (13.46, -2.92);
\node[note, anchor=north, inner sep=1pt] at (13.26, -2.95) {same \texttt{rel\_patch\_idx}};
\draw[seriesbox] (12.16, -1.03) rectangle (14.36, -1.93);
\draw[seriesbox] (12.16, -2.01) rectangle (14.36, -2.91);
\draw[seriesbrace] (11.68, -1.93) -- (11.68, -1.03);
\node[note, rotate=90, anchor=south] at (11.56, -1.48) {group $g$};
\draw[seriesbrace] (11.68, -2.91) -- (11.68, -2.01);
\node[note, rotate=90, anchor=south] at (11.56, -2.46) {group $g'$};
\node[rowlabel, text=roley]  at (12.06, -1.25) {$y$};
\node[rowlabel, text=rolex]  at (12.06, -1.70) {$x$};
\node[rowlabel, text=roley]  at (12.06, -2.23) {$y'_1$};
\node[rowlabel, text=roleyy] at (12.06, -2.68) {$y'_2$};

\foreach \j in {0,1,3,4} {
  \begin{scope}[opacity=0.3]
    \tokn{vy\j}{12.42+\j*0.42}{-1.25}{roley}
    \tokn{vx\j}{12.42+\j*0.42}{-1.70}{rolex}
    \tokn{va\j}{12.42+\j*0.42}{-2.23}{roley}
    \tokn{vb\j}{12.42+\j*0.42}{-2.68}{roleyy}
  \end{scope}
}
\tokn{vy2}{13.26}{-1.25}{roley}
\tokn{vx2}{13.26}{-1.70}{rolex}
\tokn{va2}{13.26}{-2.23}{roley}
\tokn{vb2}{13.26}{-2.68}{roleyy}

\draw[attnarrow, tfcbody] (vx2.west) to[bend left=55] (vy2.west);
\draw[bothways, tfcbody] (va2.west) to[bend right=55] (vb2.west);

\node[note, anchor=west, align=left] at (14.48, -1.48) {$x{\to}y$ only\\($y{\not\to}x$ firewall)};
\node[note, anchor=west, align=left] at (14.48, -2.46) {$y'_1{\leftrightarrow}y'_2$\\(univar $=0.0$)};
\end{scope}

\draw[tfcmuted, densely dotted, line width=0.4pt] (0, -3.72) -- (15.54, -3.72);

% --- (e) MOD-32 STRIP PACKING ---
\node[paneltitle, anchor=north] at (2.80, -3.92) {(e) MOD-32 STRIP PACKING};
\begin{scope}[yshift=-0.65cm]
\node[sublabel, rotate=90, anchor=south] at (0.24, -5.45) {training};
\fill[pattern=north east lines, pattern color=tfchair]
  ({\packx{0}}, {\packy{0}}) rectangle ({\packx{16}}, {\packy{7}});
\prow{0}{0}{7}{roley} \prow{1}{0}{7}{rolex}
\prow{0}{7}{12}{roley}
\prow{0}{12}{15}{roley}
\prow{1}{7}{13}{roley} \prow{2}{7}{13}{roleyy}
\prow{2}{0}{7}{roley}
\prow{3}{0}{10}{roley} \prow{4}{0}{10}{rolex}
\prow{3}{10}{16}{roley}
\prow{4}{10}{14}{roley}
\prow{5}{0}{4}{roley} \prow{6}{0}{4}{roleyy}
\prow{5}{4}{16}{roley}
\prow{6}{4}{9}{roley}
\prow{6}{9}{15}{roley}
\foreach \c in {1,...,15}
  {\draw[white, line width=0.3pt]
    ({\packx{\c}}, {\packy{0}}) -- ({\packx{\c}}, {\packy{7}});}
\pbox{0}{1}{0}{7} \pbox{2}{2}{0}{7} \pbox{0}{0}{7}{12} \pbox{0}{0}{12}{15} \pbox{1}{2}{7}{13}
\pbox{3}{4}{0}{10} \pbox{3}{3}{10}{16} \pbox{4}{4}{10}{14}
\pbox{5}{6}{0}{4} \pbox{5}{5}{4}{16} \pbox{6}{6}{4}{9} \pbox{6}{6}{9}{15}
\draw[tfcmuted, line width=0.4pt]
  ({\packx{0}}, {\packy{0}}) rectangle ({\packx{16}}, {\packy{7}});
\node[note, rotate=90, anchor=south] at (0.50, -5.45) {32 variate rows};
\node[note, anchor=north, inner sep=1pt] at (2.80, -6.78)
  {4,096 steps, 128 patches (99.5\% dense)};
\end{scope}

% --- (f) SUFFIX HORIZON & BLACKOUT MASKING ---
\node[paneltitle, anchor=north] at (10.75, -3.92)
  {(f) SUFFIX HORIZON \& BLACKOUT MASKING};
\begin{scope}[yshift=-0.65cm]
\node[sublabel, rotate=90, anchor=south] at (5.85, -4.90) {training};
\node[note, anchor=east, inner sep=1pt] at (6.58, -4.45) {input};
\node[note, anchor=east, inner sep=1pt] at (6.58, -5.35) {output};
\fill[tfcsurface] (6.68+5*0.64-0.04, -4.24) rectangle (6.68+8*0.64+0.04, -5.56);
\foreach \p/\k in {0/obs,1/obs,2/obs,3/blank,4/obs,5/blank,6/blank,7/blank}
  {\cpmpatch{6.68+\p*0.64}{-4.45}{\k}{\p*0.64}{roley}}
\node[note, anchor=south, inner sep=0.5pt] at (6.68+3.5*0.64, -4.23) {30\% blk};
\node[smallblock, minimum width=1.15cm] (cpmtrain) at (9.24, -4.90) {\texttt{TSFMModel}};
\draw[attnarrow, tfcbody] (9.24, -4.65) -- (cpmtrain.north);
\draw[attnarrow, tfcbody] (cpmtrain.south) -- (9.24, -5.15);
\begin{scope}[opacity=0.28]
  \foreach \p in {0,...,4} {\cpmpatch{6.68+\p*0.64}{-5.35}{none}{\p*0.64}{roley}}
\end{scope}
\foreach \p in {5,6,7}
  {\cpmpatch{6.68+\p*0.64}{-5.35}{pred}{\p*0.64}{roley}}
\node[note, anchor=west, text width=3.20cm] at (12.00, -4.90)
  {random suffix horizon ($K_{\text{hor}}{\le}32$p); $1/V_g$ $\phi_{12}$-damped Q21 loss on valid targets};

\node[sublabel, rotate=90, anchor=south] at (5.85, -6.40) {inference};
\node[note, anchor=east, inner sep=1pt] at (6.58, -5.95) {input};
\node[note, anchor=east, inner sep=1pt] at (6.58, -6.85) {output};
\fill[tfcsurface] (6.68+5*0.64-0.04, -5.74) rectangle (6.68+8*0.64+0.04, -7.06);
\foreach \p/\k in {0/obs,1/obs,2/obs,3/obs,4/obs,5/blank,6/blank,7/blank}
  {\cpmpatch{6.68+\p*0.64}{-5.95}{\k}{\p*0.64}{roley}}
\node[smallblock, minimum width=1.15cm] (cpminfer) at (9.24, -6.40) {\texttt{TSFMModel}};
\draw[attnarrow, tfcbody] (9.24, -6.15) -- (cpminfer.north);
\draw[attnarrow, tfcbody] (cpminfer.south) -- (9.24, -6.65);
\begin{scope}[opacity=0.28]
  \foreach \p in {0,...,4} {\cpmpatch{6.68+\p*0.64}{-6.85}{none}{\p*0.64}{roley}}
\end{scope}
\foreach \p in {5,6,7} {\cpmpatch{6.68+\p*0.64}{-6.85}{pred}{\p*0.64}{roley}}
\node[note, anchor=west, text width=3.20cm] at (12.00, -6.40)
  {blanked horizon appended and decoded in one pass ($H{\le}1{,}024$)};
\end{scope}

\end{tikzpicture}
\end{document}
"""


def _trim_png_whitespace(png_path: Path, margin_px: int = 24) -> None:
    """Crop excessive white borders around a rendered diagram while preserving uniform margins."""
    with Image.open(png_path) as img:
        rgb = img.convert("RGB")
        bg = Image.new("RGB", rgb.size, (255, 255, 255))
        diff = ImageChops.difference(rgb, bg)
        bbox = diff.getbbox()
        if bbox is None:
            return
        w, h = rgb.size
        left = max(0, bbox[0] - margin_px)
        top = max(0, bbox[1] - margin_px)
        right = min(w, bbox[2] + margin_px)
        bottom = min(h, bbox[3] + margin_px)
        rgb.crop((left, top, right, bottom)).save(png_path, format="PNG")


def export_architecture_schematic(out_dir: Path) -> None:
    """Render standalone TikZ TSFMModel architecture schematic to results/figures/fig0_model_architecture.png."""
    pdflatex_bin = shutil.which("pdflatex") or (
        "/Library/TeX/texbin/pdflatex" if Path("/Library/TeX/texbin/pdflatex").exists() else None
    )
    gs_bin = shutil.which("gs")
    if gs_bin is None:
        for cand in ("/usr/local/bin/gs", "/opt/homebrew/bin/gs"):
            if Path(cand).exists():
                gs_bin = cand
                break
    out_png = out_dir / "fig0_model_architecture.png"
    if not pdflatex_bin or not gs_bin:
        print(f"Skipping fig0 schematic compile (pdflatex={pdflatex_bin}, gs={gs_bin}); keeping {out_png}.")
        return

    with tempfile.TemporaryDirectory(prefix="tsfm_arch_tikz_") as tmp_dir:
        tmp = Path(tmp_dir)
        tex_path = tmp / "arch.tex"
        pdf_path = tmp / "arch.pdf"
        raw_png = tmp / "arch_raw.png"
        tex_path.write_text(_ARCHITECTURE_TIKZ_TEX)

        try:
            subprocess.run(
                [pdflatex_bin, "-interaction=nonstopmode", "-halt-on-error", "arch.tex"],
                cwd=tmp,
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                [
                    gs_bin,
                    "-dSAFER",
                    "-dBATCH",
                    "-dNOPAUSE",
                    "-sDEVICE=png16m",
                    "-r350",
                    "-dTextAlphaBits=4",
                    "-dGraphicsAlphaBits=4",
                    f"-sOutputFile={raw_png}",
                    str(pdf_path),
                ],
                cwd=tmp,
                check=True,
                capture_output=True,
                text=True,
            )
            _trim_png_whitespace(raw_png, margin_px=24)
            shutil.copy2(raw_png, out_png)
        except (subprocess.CalledProcessError, OSError) as exc:
            stderr_msg = getattr(exc, "stderr", "") or getattr(exc, "stdout", "") or str(exc)
            print(f"Warning: fig0 schematic compile failed ({stderr_msg[-400:].strip()}); keeping {out_png}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export summary matplotlib PNGs to results/figures/.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_PROJECT_ROOT / "results")
    parser.add_argument("--ckpt-root", type=Path, default=DEFAULT_PROJECT_ROOT / "checkpoints")
    args = parser.parse_args()

    _setup_style()
    fig_dir = args.results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    export_architecture_schematic(fig_dir)
    plot_corpus_skew(args.results_dir / "eda_dataset_catalog.csv", fig_dir)
    plot_training_curves(args.ckpt_root, args.results_dir, fig_dir)
    plot_ablation_waterfall(
        args.results_dir / "gifteval_ablation_summary.csv",
        fig_dir,
    )
    plot_slices_and_blackout(
        args.results_dir / "gifteval_slices_summary.csv",
        args.results_dir / "gifteval_blackout_stress.csv",
        fig_dir,
    )
    print(f"Exported figures to {fig_dir}")


if __name__ == "__main__":
    main()
