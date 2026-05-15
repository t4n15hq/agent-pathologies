"""Generate publication-quality figures for the paper.

Reads from data/*.jsonl, writes to paper/figures/. Deterministic
(bootstrap seed fixed), colorblind-safe, serif typography, no chart
junk. Each figure is a function so individual plots can be regenerated
without rebuilding everything.

USAGE:
    python scripts/make_paper_figures.py            # all figures
    python scripts/make_paper_figures.py fig1 fig3  # just specified ones
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mpl.use("Agg")

from agent_pathologies.analysis.metrics import (
    filter_analyzable,
    load_jsonl,
    extracted_divergence,
)
from agent_pathologies.analysis.stats import (
    benjamini_hochberg,
    bootstrap_ci,
    cohens_h,
    paired_did_bootstrap,
    paired_test,
)
from agent_pathologies.tasks.scoring import extract_last_integer

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

# Okabe-Ito colorblind-safe palette + neutrals
COLOR_INSTRUCT = "#0072B2"   # blue
COLOR_REASONING = "#D55E00"  # vermillion
COLOR_DEEPSEEK = "#009E73"   # bluish green
COLOR_QWEN = "#CC79A7"       # reddish purple
COLOR_NULL = "#999999"       # grey
COLOR_FILLER_KIND = {
    "irrelevant": "#0072B2",
    "related": "#D55E00",
    "token_matched": "#009E73",
    "collapsed": "#CC79A7",
}

FAMILY_ORDER = ["qwen3-30b", "qwen3-235b", "deepseek-v4-flash", "deepseek-v4-pro"]
FAMILY_LABEL = {
    "qwen3-30b": "Qwen3-30B",
    "qwen3-235b": "Qwen3-235B",
    "deepseek-v4-flash": "DeepSeek V4-Flash",
    "deepseek-v4-pro": "DeepSeek V4-Pro",
}
ROLE_LABEL = {"instruct": "Instruct", "reasoning": "Reasoning"}

PAPER_FIGS = Path("paper/figures")
DATA = Path("data")


def setup_style():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.4,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
        "figure.dpi": 100,
    })


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_axis(exp: str) -> pd.DataFrame:
    path = DATA / f"{exp}.jsonl"
    if not path.exists():
        return pd.DataFrame()
    df = load_jsonl(path)
    return filter_analyzable(df)


def families_present(df: pd.DataFrame) -> list[str]:
    if df.empty:
        return []
    have = set(df["model_family"].dropna().unique())
    return [f for f in FAMILY_ORDER if f in have]


# ---------------------------------------------------------------------------
# Figure 1 — Headline forest plot across all axes
# ---------------------------------------------------------------------------

def fig1_headline_forest():
    sc = load_axis("self_consistency")
    sy = load_axis("sycophancy")
    ct = load_axis("context_rot")

    rows = []

    # Self-consistency: per-family accuracy delta (instruct - reasoning),
    # paired by task_id.
    if not sc.empty:
        for fam in families_present(sc):
            sub = sc[sc["model_family"] == fam]
            per_task = sub.groupby(["model_role", "task_id"])["is_correct"].mean().reset_index()
            instr = per_task[per_task["model_role"] == "instruct"].set_index("task_id")["is_correct"]
            reas = per_task[per_task["model_role"] == "reasoning"].set_index("task_id")["is_correct"]
            common = instr.index.intersection(reas.index)
            if len(common) < 5:
                continue
            diff = (reas.loc[common] - instr.loc[common]).values
            lo, hi = bootstrap_ci(diff.tolist(), n_iters=5000)
            rows.append({
                "axis": "Self-consistency (acc)",
                "family": fam,
                "delta": float(diff.mean()),
                "ci_lo": lo, "ci_hi": hi,
                "n": int(len(common)),
            })

    # Sycophancy: per-family DiD pooled across gaps (paired by task_id)
    if not sy.empty:
        syc = sy.copy()
        syc["condition"] = syc["sweep_value"].apply(lambda x: x["condition"])
        for fam in families_present(syc):
            sub = syc[syc["model_family"] == fam]
            try:
                ic = sub[(sub["model_role"] == "instruct") & (sub["condition"] == "correct")
                        ].set_index("task_id")["is_correct"].groupby(level=0).mean()
                iw = sub[(sub["model_role"] == "instruct") & (sub["condition"] == "wrong")
                        ].set_index("task_id")["is_correct"].groupby(level=0).mean()
                rc = sub[(sub["model_role"] == "reasoning") & (sub["condition"] == "correct")
                        ].set_index("task_id")["is_correct"].groupby(level=0).mean()
                rw = sub[(sub["model_role"] == "reasoning") & (sub["condition"] == "wrong")
                        ].set_index("task_id")["is_correct"].groupby(level=0).mean()
            except KeyError:
                continue
            common = ic.index.intersection(iw.index).intersection(rc.index).intersection(rw.index)
            if len(common) < 5:
                continue
            res = paired_did_bootstrap(
                instr_correct=ic.loc[common].astype(float).tolist(),
                instr_wrong=iw.loc[common].astype(float).tolist(),
                reas_correct=rc.loc[common].astype(float).tolist(),
                reas_wrong=rw.loc[common].astype(float).tolist(),
                n_iters=5000,
            )
            # Use the `reasoning_gain` convention everywhere: positive means
            # reasoning is LESS sycophantic. (See stats.DiDResult docstring.)
            rows.append({
                "axis": "Sycophancy (reasoning gain)",
                "family": fam,
                "delta": float(res.reasoning_gain),
                "ci_lo": float(res.gain_ci_lo),
                "ci_hi": float(res.gain_ci_hi),
                "n": int(res.n_paired),
            })

    # Context-rot: per-family pooled accuracy delta (reasoning - instruct)
    # across filler conditions, paired by task_id at the deepest filler.
    if not ct.empty:
        ctx = ct.copy()
        ctx["n_filler"] = ctx["sweep_value"].apply(lambda x: x["n_filler"])
        ctx["kind"] = ctx["sweep_value"].apply(lambda x: x["kind"])
        deep = ctx[(ctx["kind"] == "irrelevant") & (ctx["n_filler"] == ctx["n_filler"].max())]
        for fam in families_present(deep):
            sub = deep[deep["model_family"] == fam]
            instr = sub[sub["model_role"] == "instruct"].set_index("task_id")["is_correct"]
            reas = sub[sub["model_role"] == "reasoning"].set_index("task_id")["is_correct"]
            common = instr.index.intersection(reas.index)
            if len(common) < 5:
                continue
            diff = (reas.loc[common].astype(float) - instr.loc[common].astype(float)).values
            lo, hi = bootstrap_ci(diff.tolist(), n_iters=5000)
            rows.append({
                "axis": "Context-rot (acc @ max filler)",
                "family": fam,
                "delta": float(diff.mean()),
                "ci_lo": lo, "ci_hi": hi,
                "n": int(len(common)),
            })

    if not rows:
        print("fig1_headline_forest: no data yet — skipping")
        return

    df = pd.DataFrame(rows)
    # Order: by axis then by family
    axis_order = ["Self-consistency (acc)", "Sycophancy (reasoning gain)", "Context-rot (acc @ max filler)"]
    df["axis"] = pd.Categorical(df["axis"], axis_order, ordered=True)
    df["family"] = pd.Categorical(df["family"], FAMILY_ORDER, ordered=True)
    df = df.sort_values(["axis", "family"]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(7, 0.32 * len(df) + 1.5))
    y = np.arange(len(df))
    colors = [COLOR_REASONING if v >= 0 else COLOR_INSTRUCT for v in df["delta"]]
    ax.errorbar(
        df["delta"], y,
        xerr=[df["delta"] - df["ci_lo"], df["ci_hi"] - df["delta"]],
        fmt="o", color="black", ecolor="grey", elinewidth=1.0,
        markersize=5, capsize=2,
    )
    # Color each marker
    for i, c in enumerate(colors):
        ax.plot(df["delta"].iloc[i], y[i], "o", color=c, markersize=5, zorder=3)

    ax.axvline(0, color="black", linewidth=0.8, linestyle="-")
    ax.set_yticks(y)
    yticklabels = [f"{FAMILY_LABEL[r['family']]}  (n={r['n']})" for _, r in df.iterrows()]
    ax.set_yticklabels(yticklabels)
    ax.invert_yaxis()
    ax.set_xlabel("Within-pair effect (positive = reasoning advantage)")

    # Group separators between axes
    prev_axis = None
    for i, (_, row) in enumerate(df.iterrows()):
        if prev_axis is not None and row["axis"] != prev_axis:
            ax.axhline(i - 0.5, color="black", linewidth=0.4, alpha=0.3)
        prev_axis = row["axis"]

    # Annotate axis groups on the right
    axis_first = df.groupby("axis", observed=True).apply(lambda g: g.index.min(), include_groups=False)
    axis_last = df.groupby("axis", observed=True).apply(lambda g: g.index.max(), include_groups=False)
    xmax = df["ci_hi"].max() + 0.05
    for axis, first in axis_first.items():
        last = axis_last[axis]
        ax.text(xmax, (first + last) / 2, axis, ha="left", va="center",
                fontsize=8, fontstyle="italic", color="#555")

    ax.set_xlim(df["ci_lo"].min() - 0.05, xmax + 0.35)
    ax.set_title("Within-family paired effects across pathology axes",
                 loc="left", weight="bold")
    fig.savefig(PAPER_FIGS / "fig1_headline_forest.pdf")
    fig.savefig(PAPER_FIGS / "fig1_headline_forest.png")
    plt.close(fig)
    print("fig1 written.")


# ---------------------------------------------------------------------------
# Figure 2 — Self-consistency paired bars (accuracy + divergence)
# ---------------------------------------------------------------------------

def fig2_selfconsistency_paired():
    df = load_axis("self_consistency")
    if df.empty:
        print("fig2: no self_consistency data — skipping")
        return

    fams = families_present(df)

    # Per-family per-role: pooled accuracy and integer-divergence per task,
    # then aggregated to mean ± bootstrap 95% CI.
    rows = []
    for fam in fams:
        for role in ["instruct", "reasoning"]:
            sub = df[(df["model_family"] == fam) & (df["model_role"] == role)]
            if sub.empty:
                continue
            per_task = sub.groupby("task_id")
            accs = per_task["is_correct"].mean().tolist()
            divs = per_task["probe_answer"].apply(extracted_divergence).tolist()
            acc_mean = float(np.mean(accs)) if accs else float("nan")
            div_mean = float(np.mean(divs)) if divs else float("nan")
            acc_lo, acc_hi = bootstrap_ci(accs, n_iters=5000) if accs else (np.nan, np.nan)
            div_lo, div_hi = bootstrap_ci(divs, n_iters=5000) if divs else (np.nan, np.nan)
            rows.append({
                "family": fam, "role": role,
                "acc": acc_mean, "acc_lo": acc_lo, "acc_hi": acc_hi,
                "div": div_mean, "div_lo": div_lo, "div_hi": div_hi,
            })
    if not rows:
        return
    plot_df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4), sharey=False)

    width = 0.36
    x = np.arange(len(fams))

    for i, role in enumerate(["instruct", "reasoning"]):
        sub = plot_df[plot_df["role"] == role].set_index("family").reindex(fams)
        color = COLOR_INSTRUCT if role == "instruct" else COLOR_REASONING
        # accuracy
        axes[0].bar(
            x + (i - 0.5) * width, sub["acc"], width,
            label=ROLE_LABEL[role], color=color, edgecolor="white", linewidth=0.5,
            yerr=[sub["acc"] - sub["acc_lo"], sub["acc_hi"] - sub["acc"]],
            error_kw={"linewidth": 0.7, "ecolor": "#333"},
        )
        # divergence
        axes[1].bar(
            x + (i - 0.5) * width, sub["div"], width,
            label=ROLE_LABEL[role], color=color, edgecolor="white", linewidth=0.5,
            yerr=[sub["div"] - sub["div_lo"], sub["div_hi"] - sub["div"]],
            error_kw={"linewidth": 0.7, "ecolor": "#333"},
        )

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([FAMILY_LABEL[f] for f in fams], rotation=15, ha="right")
        ax.grid(axis="x", visible=False)

    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Accuracy on hardness-5 arithmetic", loc="left", fontsize=9)
    axes[0].legend(loc="upper left", frameon=False)

    axes[1].set_ylabel("Integer-divergence")
    axes[1].set_ylim(0, max(0.05, plot_df["div_hi"].max() * 1.1))
    axes[1].set_title("Answer divergence across 25 replays", loc="left", fontsize=9)
    axes[1].legend(loc="upper left", frameon=False)

    fig.suptitle("Self-consistency: paired comparison per family",
                 fontsize=10, weight="bold", x=0.01, y=1.00, ha="left")
    fig.savefig(PAPER_FIGS / "fig2_selfconsistency_paired.pdf")
    fig.savefig(PAPER_FIGS / "fig2_selfconsistency_paired.png")
    plt.close(fig)
    print("fig2 written.")


# ---------------------------------------------------------------------------
# Figure 3 — Sycophancy DiD per family
# ---------------------------------------------------------------------------

def fig3_sycophancy_did():
    df = load_axis("sycophancy")
    if df.empty:
        print("fig3: no sycophancy data — skipping")
        return
    df = df.copy()
    df["condition"] = df["sweep_value"].apply(lambda x: x["condition"])

    rows = []
    for fam in families_present(df):
        sub = df[df["model_family"] == fam]
        try:
            ic = sub[(sub["model_role"] == "instruct") & (sub["condition"] == "correct")
                    ].set_index("task_id")["is_correct"].groupby(level=0).mean()
            iw = sub[(sub["model_role"] == "instruct") & (sub["condition"] == "wrong")
                    ].set_index("task_id")["is_correct"].groupby(level=0).mean()
            rc = sub[(sub["model_role"] == "reasoning") & (sub["condition"] == "correct")
                    ].set_index("task_id")["is_correct"].groupby(level=0).mean()
            rw = sub[(sub["model_role"] == "reasoning") & (sub["condition"] == "wrong")
                    ].set_index("task_id")["is_correct"].groupby(level=0).mean()
        except KeyError:
            continue
        common = ic.index.intersection(iw.index).intersection(rc.index).intersection(rw.index)
        if len(common) < 5:
            continue
        res = paired_did_bootstrap(
            instr_correct=ic.loc[common].astype(float).tolist(),
            instr_wrong=iw.loc[common].astype(float).tolist(),
            reas_correct=rc.loc[common].astype(float).tolist(),
            reas_wrong=rw.loc[common].astype(float).tolist(),
            n_iters=8000,
        )
        # Use the `reasoning_gain` convention (positive = reasoning less
        # sycophantic). See stats.DiDResult docstring for the relationship
        # between `did` (raw) and `reasoning_gain` (user-facing).
        rows.append({
            "family": fam,
            "did": float(res.reasoning_gain),
            "ci_lo": float(res.gain_ci_lo),
            "ci_hi": float(res.gain_ci_hi),
            "instr_gap": float(res.instr_gap),
            "reas_gap": float(res.reas_gap),
            "n": int(res.n_paired),
        })
    if not rows:
        return

    plot_df = pd.DataFrame(rows)
    plot_df["family"] = pd.Categorical(plot_df["family"], FAMILY_ORDER, ordered=True)
    plot_df = plot_df.sort_values("family")

    fig, ax = plt.subplots(figsize=(7, 3.6))
    x = np.arange(len(plot_df))
    colors = [COLOR_DEEPSEEK if "deepseek" in f else COLOR_QWEN for f in plot_df["family"]]

    ax.bar(
        x, plot_df["did"], color=colors, edgecolor="white", linewidth=0.6,
        yerr=[plot_df["did"] - plot_df["ci_lo"], plot_df["ci_hi"] - plot_df["did"]],
        error_kw={"linewidth": 0.7, "ecolor": "#333"},
    )
    # Symmetric preregistered effect-size threshold |DiD|≥0.10. With the
    # flipped sign convention, positive crosses the +0.10 line for a
    # reasoning advantage; negative crosses -0.10 for an instruct advantage.
    ax.axhline( 0.10, color="black", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.axhline(-0.10, color="black", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.text(len(plot_df) - 0.5, 0.105, "preregistered |DiD|≥0.10",
            ha="right", va="bottom", fontsize=7, style="italic", color="#555")

    ax.set_xticks(x)
    ax.set_xticklabels([FAMILY_LABEL[f] for f in plot_df["family"]], rotation=15, ha="right")
    ax.set_ylabel(r"Reasoning advantage: instruct(c$-$w) $-$ reasoning(c$-$w)")
    ax.grid(axis="x", visible=False)
    ax.set_title("Sycophancy paired DiD, by family (positive = reasoning more resistant)",
                 loc="left", weight="bold")

    # Annotate n per bar
    for xi, row in zip(x, plot_df.itertuples()):
        ax.text(xi, max(row.ci_hi, 0.01) + 0.03,
                f"n={row.n}", ha="center", va="bottom", fontsize=7, color="#555")

    # Legend for color
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=COLOR_DEEPSEEK, label="DeepSeek (within-model toggle)"),
               Patch(facecolor=COLOR_QWEN, label="Qwen (cross-SKU)")]
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=8)

    fig.savefig(PAPER_FIGS / "fig3_sycophancy_did.pdf")
    fig.savefig(PAPER_FIGS / "fig3_sycophancy_did.png")
    plt.close(fig)
    print("fig3 written.")


# ---------------------------------------------------------------------------
# Figure 4 — Sycophancy accuracy by (family, role, condition) heatmap
# ---------------------------------------------------------------------------

def fig4_sycophancy_conditions():
    df = load_axis("sycophancy")
    if df.empty:
        print("fig4: no sycophancy data — skipping")
        return
    df = df.copy()
    df["condition"] = df["sweep_value"].apply(lambda x: x["condition"])

    fams = families_present(df)
    if not fams:
        return

    matrix = np.full((len(fams) * 2, 3), np.nan)
    row_labels = []
    for i, fam in enumerate(fams):
        for j, role in enumerate(["instruct", "reasoning"]):
            row_labels.append(f"{FAMILY_LABEL[fam]} — {ROLE_LABEL[role]}")
            for k, cond in enumerate(["wrong", "neutral", "correct"]):
                sub = df[(df["model_family"] == fam) & (df["model_role"] == role)
                         & (df["condition"] == cond)]
                if len(sub) > 0:
                    matrix[i * 2 + j, k] = sub["is_correct"].mean()

    fig, ax = plt.subplots(figsize=(5.5, 0.55 * len(row_labels) + 1.0))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(3))
    ax.set_xticklabels(["Wrong pushback", "Neutral pushback", "Correct pushback"], rotation=20, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)

    # Annotate cells
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            if np.isnan(v):
                continue
            txt_color = "white" if (v < 0.35 or v > 0.85) else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=8, color=txt_color)

    # Separator between families
    for i in range(2, len(row_labels), 2):
        ax.axhline(i - 0.5, color="white", linewidth=2)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.04)
    cbar.set_label("Accuracy at re-probe", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    ax.set_title("Sycophancy: accuracy after pushback,\nby (family, role, condition)",
                 loc="left", weight="bold")
    fig.savefig(PAPER_FIGS / "fig4_sycophancy_conditions.pdf")
    fig.savefig(PAPER_FIGS / "fig4_sycophancy_conditions.png")
    plt.close(fig)
    print("fig4 written.")


# ---------------------------------------------------------------------------
# Figure 5 — Context-rot decay curves, faceted by family
# ---------------------------------------------------------------------------

def fig5_contextrot_curves():
    """Simplified main figure: irrelevant-filler decay per family with paired
    instruct vs reasoning lines and bootstrap CI bands. The full four-kind
    plot is in fig5b_contextrot_full (supplementary)."""
    df = load_axis("context_rot")
    if df.empty:
        print("fig5: no context_rot data — skipping")
        return
    df = df.copy()
    df["n_filler"] = df["sweep_value"].apply(lambda x: x["n_filler"])
    df["kind"] = df["sweep_value"].apply(lambda x: x["kind"])

    # Main figure shows only the irrelevant filler kind — the cleanest
    # decay signal and the one used for the §6 results table. Other kinds
    # go to fig5b_contextrot_full for the supplementary.
    df = df[df["kind"] == "irrelevant"]

    fams = families_present(df)
    if not fams:
        return

    n_fam = len(fams)
    ncols = 2
    nrows = (n_fam + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.5, 3.0 * nrows),
                              sharey=True, sharex=True)
    if nrows == 1:
        axes = np.atleast_2d(axes)

    for idx, fam in enumerate(fams):
        ax = axes[idx // ncols, idx % ncols]
        sub_fam = df[df["model_family"] == fam]
        for role, color, ls in [("instruct", COLOR_INSTRUCT, "-"),
                                 ("reasoning", COLOR_REASONING, "--")]:
            sub_role = sub_fam[sub_fam["model_role"] == role]
            if sub_role.empty:
                continue
            xs, means, los, his = [], [], [], []
            for k_val, group in sub_role.groupby("n_filler"):
                accs = group["is_correct"].astype(float).tolist()
                if len(accs) < 2:
                    continue
                xs.append(k_val)
                means.append(float(np.mean(accs)))
                lo, hi = bootstrap_ci(accs, n_iters=2000)
                los.append(lo); his.append(hi)
            if not xs:
                continue
            order = np.argsort(xs)
            xs_a = np.array(xs)[order]
            means_a = np.array(means)[order]
            los_a = np.array(los)[order]
            his_a = np.array(his)[order]
            ax.fill_between(xs_a, los_a, his_a, color=color, alpha=0.15, linewidth=0)
            ax.plot(xs_a, means_a, color=color, linestyle=ls, marker="o",
                    markersize=4, linewidth=1.3, label=ROLE_LABEL[role])
        ax.set_title(FAMILY_LABEL[fam], fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        if idx // ncols == nrows - 1:
            ax.set_xlabel("# irrelevant filler turn-pairs")
        if idx % ncols == 0:
            ax.set_ylabel("Accuracy")

    for k in range(n_fam, nrows * ncols):
        axes[k // ncols, k % ncols].set_visible(False)

    axes[0, 0].legend(loc="lower left", frameon=False, fontsize=8)

    fig.suptitle("Context rot: accuracy decay under irrelevant filler\n"
                 "(bands = bootstrap 95% CI; full multi-kind comparison in supplementary fig5b)",
                 fontsize=10, weight="bold", x=0.01, y=1.0, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(PAPER_FIGS / "fig5_contextrot_curves.pdf")
    fig.savefig(PAPER_FIGS / "fig5_contextrot_curves.png")
    plt.close(fig)
    print("fig5 written (simplified, irrelevant only).")


def fig5b_contextrot_full():
    """Supplementary: full four-filler-kind decay-curve grid. Useful for
    reviewers who want to see token_matched / collapsed / related kinds
    alongside the headline irrelevant curve. Considered too busy for the
    main paper."""
    df = load_axis("context_rot")
    if df.empty:
        print("fig5b: no context_rot data — skipping")
        return
    df = df.copy()
    df["n_filler"] = df["sweep_value"].apply(lambda x: x["n_filler"])
    df["kind"] = df["sweep_value"].apply(lambda x: x["kind"])

    fams = families_present(df)
    if not fams:
        return

    n_fam = len(fams)
    ncols = 2
    nrows = (n_fam + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(8.5, 3.0 * nrows),
                              sharey=True, sharex=True)
    if nrows == 1:
        axes = np.atleast_2d(axes)

    kinds = sorted(df["kind"].dropna().unique())

    for idx, fam in enumerate(fams):
        ax = axes[idx // ncols, idx % ncols]
        sub_fam = df[df["model_family"] == fam]
        for kind in kinds:
            sub_kind = sub_fam[sub_fam["kind"] == kind]
            if sub_kind.empty:
                continue
            grouped = sub_kind.groupby(["model_role", "n_filler"]).agg(
                acc=("is_correct", "mean"),
                n=("is_correct", "count"),
            ).reset_index()
            for role, marker, ls in [("instruct", "o", "-"), ("reasoning", "s", "--")]:
                g = grouped[grouped["model_role"] == role].sort_values("n_filler")
                if len(g) < 2:
                    continue
                color = COLOR_FILLER_KIND.get(kind, "#888")
                alpha = 1.0 if role == "instruct" else 0.6
                ax.plot(g["n_filler"], g["acc"], marker=marker, linestyle=ls,
                        color=color, alpha=alpha, markersize=4, linewidth=1.0,
                        label=f"{kind} / {ROLE_LABEL[role]}" if idx == 0 else None)
        ax.set_title(FAMILY_LABEL[fam], fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("# filler turn-pairs")
        if idx % ncols == 0:
            ax.set_ylabel("Accuracy")

    for k in range(n_fam, nrows * ncols):
        axes[k // ncols, k % ncols].set_visible(False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center",
                   ncol=4, fontsize=7, frameon=False,
                   bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Supplementary: context rot across all four filler kinds, per family",
                 fontsize=10, weight="bold", x=0.01, y=1.0, ha="left")
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(PAPER_FIGS / "fig5b_contextrot_full.pdf")
    fig.savefig(PAPER_FIGS / "fig5b_contextrot_full.png")
    plt.close(fig)
    print("fig5b written (supplementary, all kinds).")


# ---------------------------------------------------------------------------
# Figure 6 — Qwen mode-collapse (top answers per task on Qwen instruct)
# ---------------------------------------------------------------------------

def fig6_qwen_mode_collapse():
    df = load_axis("self_consistency")
    if df.empty:
        print("fig6: no self_consistency data — skipping")
        return

    qwen_pairs = [("qwen3-30b", "Qwen3-30B Instruct"),
                  ("qwen3-235b", "Qwen3-235B Instruct")]
    qwen_pairs = [(f, l) for f, l in qwen_pairs if f in df["model_family"].unique()]
    if not qwen_pairs:
        print("fig6: no Qwen data — skipping")
        return

    fig, axes = plt.subplots(1, len(qwen_pairs), figsize=(4.4 * len(qwen_pairs), 3.2))
    if len(qwen_pairs) == 1:
        axes = [axes]

    for ax, (fam, label) in zip(axes, qwen_pairs):
        sub = df[(df["model_family"] == fam) & (df["model_role"] == "instruct")]
        if sub.empty:
            continue
        extracted = [extract_last_integer(r) for r in sub["probe_answer"]]
        extracted = [str(e) for e in extracted if e is not None]
        if not extracted:
            continue
        counts = Counter(extracted)
        top5 = counts.most_common(5)
        total = sum(counts.values())
        # Aggregate the rest
        rest = total - sum(c for _, c in top5)
        bars = top5 + [("(other)", rest)] if rest else top5
        labels_, vals = zip(*bars)
        colors = [COLOR_QWEN if l != "(other)" else "#cccccc" for l in labels_]
        y = np.arange(len(labels_))
        ax.barh(y, [v / total for v in vals], color=colors, edgecolor="white", linewidth=0.6)
        for yi, (lab, v) in enumerate(bars):
            ax.text((v / total) + 0.01, yi, f"{v} / {total} ({100*v/total:.0f}%)",
                    va="center", fontsize=7)
        ax.set_yticks(y)
        ax.set_yticklabels(labels_)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_xlabel("Fraction of replays")
        ax.set_title(label, fontsize=9)
        ax.grid(axis="y", visible=False)

    fig.suptitle("Qwen instruct mode-collapse on hardness-5 arithmetic\n"
                 "(top-5 most-frequent extracted answers across all task replays)",
                 fontsize=10, weight="bold", x=0.01, y=1.0, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(PAPER_FIGS / "fig6_qwen_mode_collapse.pdf")
    fig.savefig(PAPER_FIGS / "fig6_qwen_mode_collapse.png")
    plt.close(fig)
    print("fig6 written.")


# ---------------------------------------------------------------------------
# Figure 7 — Cross-axis pass/fail summary heatmap
# ---------------------------------------------------------------------------

def fig7_cross_axis_summary():
    """Per-family per-axis effect-direction summary. Positive = reasoning
    advantage. Cell color = signed effect magnitude (clipped to [-1, 1])."""
    sc = load_axis("self_consistency")
    sy = load_axis("sycophancy")
    ct = load_axis("context_rot")

    axes_def = []
    if not sc.empty:
        axes_def.append(("Self-consistency\n(accuracy Δ)", sc, "accuracy"))
        axes_def.append(("Self-consistency\n(divergence Δ, sign-flipped)", sc, "divergence"))
    if not sy.empty:
        sy_c = sy.copy(); sy_c["condition"] = sy_c["sweep_value"].apply(lambda x: x["condition"])
        axes_def.append(("Sycophancy\n(reasoning gain)", sy_c, "did"))
    if not ct.empty:
        ct_c = ct.copy()
        ct_c["kind"] = ct_c["sweep_value"].apply(lambda x: x["kind"])
        ct_c["n_filler"] = ct_c["sweep_value"].apply(lambda x: x["n_filler"])
        axes_def.append(("Context-rot\n(accuracy Δ @ max filler)", ct_c, "ctx_acc"))

    fams = sorted(set().union(*(families_present(d) for _, d, _ in axes_def)),
                   key=lambda f: FAMILY_ORDER.index(f) if f in FAMILY_ORDER else 99)
    if not fams or not axes_def:
        print("fig7: no data — skipping")
        return

    matrix = np.full((len(fams), len(axes_def)), np.nan)
    annot = np.full(matrix.shape, "", dtype=object)

    for ji, (label, df_axis, metric) in enumerate(axes_def):
        for ii, fam in enumerate(fams):
            sub = df_axis[df_axis["model_family"] == fam]
            try:
                if metric == "accuracy":
                    instr = sub[sub["model_role"] == "instruct"].groupby("task_id")["is_correct"].mean()
                    reas = sub[sub["model_role"] == "reasoning"].groupby("task_id")["is_correct"].mean()
                    common = instr.index.intersection(reas.index)
                    if len(common) < 5: continue
                    delta = float((reas.loc[common] - instr.loc[common]).mean())
                elif metric == "divergence":
                    instr = sub[sub["model_role"] == "instruct"].groupby("task_id")["probe_answer"].apply(extracted_divergence)
                    reas = sub[sub["model_role"] == "reasoning"].groupby("task_id")["probe_answer"].apply(extracted_divergence)
                    common = instr.index.intersection(reas.index)
                    if len(common) < 5: continue
                    # Lower divergence = better → sign-flip so positive = reasoning advantage
                    delta = float((instr.loc[common] - reas.loc[common]).mean())
                elif metric == "did":
                    ic = sub[(sub["model_role"] == "instruct") & (sub["condition"] == "correct")
                            ].set_index("task_id")["is_correct"].groupby(level=0).mean()
                    iw = sub[(sub["model_role"] == "instruct") & (sub["condition"] == "wrong")
                            ].set_index("task_id")["is_correct"].groupby(level=0).mean()
                    rc = sub[(sub["model_role"] == "reasoning") & (sub["condition"] == "correct")
                            ].set_index("task_id")["is_correct"].groupby(level=0).mean()
                    rw = sub[(sub["model_role"] == "reasoning") & (sub["condition"] == "wrong")
                            ].set_index("task_id")["is_correct"].groupby(level=0).mean()
                    common = ic.index.intersection(iw.index).intersection(rc.index).intersection(rw.index)
                    if len(common) < 5: continue
                    # Sign convention: positive = reasoning advantage. Reasoning
                    # is BETTER when its correct-wrong gap is SMALLER, so we
                    # take instruct_gap − reasoning_gap.
                    delta = float(((ic.loc[common] - iw.loc[common]) - (rc.loc[common] - rw.loc[common])).mean())
                elif metric == "ctx_acc":
                    deep = sub[(sub["kind"] == "irrelevant") & (sub["n_filler"] == sub["n_filler"].max())]
                    instr = deep[deep["model_role"] == "instruct"].set_index("task_id")["is_correct"]
                    reas = deep[deep["model_role"] == "reasoning"].set_index("task_id")["is_correct"]
                    common = instr.index.intersection(reas.index)
                    if len(common) < 5: continue
                    delta = float((reas.loc[common].astype(float) - instr.loc[common].astype(float)).mean())
                else:
                    continue
            except KeyError:
                continue
            matrix[ii, ji] = delta
            annot[ii, ji] = f"{delta:+.2f}"

    fig, ax = plt.subplots(figsize=(1.7 * len(axes_def) + 1, 0.65 * len(fams) + 1.5))
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(axes_def)))
    ax.set_xticklabels([label for label, _, _ in axes_def], fontsize=8)
    ax.set_yticks(range(len(fams)))
    ax.set_yticklabels([FAMILY_LABEL[f] for f in fams])

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if not annot[i, j]:
                continue
            v = matrix[i, j]
            txt_color = "white" if abs(v) > 0.55 else "black"
            ax.text(j, i, annot[i, j], ha="center", va="center",
                    fontsize=8.5, color=txt_color, weight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    cbar.set_label("Reasoning − instruct (positive = reasoning advantage)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    ax.set_title("Cross-axis effect-direction summary, per family",
                 loc="left", weight="bold")
    fig.savefig(PAPER_FIGS / "fig7_cross_axis_summary.pdf")
    fig.savefig(PAPER_FIGS / "fig7_cross_axis_summary.png")
    plt.close(fig)
    print("fig7 written.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

FIGURES = {
    "fig1": fig1_headline_forest,
    "fig2": fig2_selfconsistency_paired,
    "fig3": fig3_sycophancy_did,
    "fig4": fig4_sycophancy_conditions,
    "fig5": fig5_contextrot_curves,
    "fig5b": fig5b_contextrot_full,
    "fig6": fig6_qwen_mode_collapse,
    "fig7": fig7_cross_axis_summary,
}


def main():
    setup_style()
    PAPER_FIGS.mkdir(parents=True, exist_ok=True)

    p = argparse.ArgumentParser()
    p.add_argument("which", nargs="*", default=list(FIGURES),
                   help=f"Figures to render. Default = all. Options: {list(FIGURES)}")
    args = p.parse_args()

    for name in args.which:
        if name not in FIGURES:
            print(f"unknown figure: {name}", file=sys.stderr)
            continue
        try:
            FIGURES[name]()
        except Exception as e:
            print(f"{name}: failed — {type(e).__name__}: {e}", file=sys.stderr)
            import traceback; traceback.print_exc()

    print(f"\nfigures written to {PAPER_FIGS.resolve()}")


if __name__ == "__main__":
    main()
