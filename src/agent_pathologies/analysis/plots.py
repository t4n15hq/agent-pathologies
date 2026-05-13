from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_accuracy_curve(
    df: pd.DataFrame, x_col: str, out: Path, title: str = ""
) -> None:
    """df must have columns: <x_col>, accuracy, sem."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(df[x_col], df["accuracy"], yerr=df["sem"], marker="o", capsize=3)
    ax.set_xlabel(x_col)
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
