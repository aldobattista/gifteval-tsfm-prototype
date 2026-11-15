"""Generate publication-clean PNG diagnostic and benchmark figures under results/figures/."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Export summary matplotlib PNGs to results/figures/.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_PROJECT_ROOT / "results")
    parser.add_argument("--ckpt-root", type=Path, default=DEFAULT_PROJECT_ROOT / "checkpoints")
    args = parser.parse_args()

    _setup_style()
    fig_dir = args.results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

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
