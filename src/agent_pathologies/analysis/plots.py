from __future__ import annotations

import os
from pathlib import Path

Path(os.environ.setdefault("MPLCONFIGDIR", "/tmp/agent-pathologies-mpl")).mkdir(
    parents=True, exist_ok=True
)
Path(os.environ.setdefault("XDG_CACHE_HOME", "/tmp/agent-pathologies-cache")).mkdir(
    parents=True, exist_ok=True
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_accuracy_curve(
    df: pd.DataFrame,
    x_col: str,
    out: Path,
    title: str = "",
    hue_col: str | None = None,
) -> None:
    """df must have columns: <x_col>, accuracy, ci_lo, ci_hi[, hue_col]."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if hue_col and hue_col in df.columns:
        for level, sub in df.groupby(hue_col):
            sub = sub.sort_values(x_col)
            yerr = np.vstack([sub["accuracy"] - sub["ci_lo"], sub["ci_hi"] - sub["accuracy"]])
            ax.errorbar(sub[x_col], sub["accuracy"], yerr=yerr,
                        marker="o", capsize=3, label=str(level))
        ax.legend(loc="best", fontsize=9)
    else:
        df = df.sort_values(x_col)
        yerr = np.vstack([df["accuracy"] - df["ci_lo"], df["ci_hi"] - df["accuracy"]])
        ax.errorbar(df[x_col], df["accuracy"], yerr=yerr, marker="o", capsize=3)
    ax.set_xlabel(x_col)
    ax.set_ylabel("accuracy (bootstrap 95% CI)")
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_paired_bars(
    df: pd.DataFrame,
    pair_col: str = "model_family",
    group_col: str = "model_role",
    out: Path | None = None,
    title: str = "",
) -> None:
    """Side-by-side instruct vs reasoning bars per family, with CIs."""
    pairs = list(df[pair_col].unique())
    roles = list(df[group_col].unique())
    width = 0.35
    x = np.arange(len(pairs))

    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(pairs) + 2), 4.5))
    for i, role in enumerate(roles):
        sub = df[df[group_col] == role].set_index(pair_col).reindex(pairs)
        acc = sub["accuracy"].values
        err_lo = (sub["accuracy"] - sub["ci_lo"]).values
        err_hi = (sub["ci_hi"] - sub["accuracy"]).values
        ax.bar(
            x + (i - (len(roles) - 1) / 2) * width,
            acc,
            width,
            yerr=np.vstack([err_lo, err_hi]),
            label=str(role),
            capsize=4,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(pairs, rotation=15, ha="right")
    ax.set_ylabel("accuracy (bootstrap 95% CI)")
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3, axis="y")
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
    return None
