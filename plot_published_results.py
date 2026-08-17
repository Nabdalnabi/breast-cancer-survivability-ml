"""Render a camera-ready comparison from aggregate published metrics."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).parents[1]
metrics = pd.read_csv(ROOT / "results" / "published_model_performance.csv")
metrics = metrics.sort_values("roc_auc", ascending=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), constrained_layout=True)
colors = ["#D8E6F3"] * len(metrics)
colors[metrics.index.tolist().index(metrics["roc_auc"].idxmax())] = "#0072B2"

axes[0].barh(metrics["model"], metrics["roc_auc"], color=colors, edgecolor="#243447", linewidth=0.7)
axes[0].set_xlim(0.75, 1.0)
axes[0].set_xlabel("ROC-AUC")
axes[0].set_title("(A) Discrimination", loc="left", fontweight="bold")

colors_pr = ["#F7D9C4"] * len(metrics)
colors_pr[metrics.index.tolist().index(metrics["auc_pr"].idxmax())] = "#D55E00"
axes[1].barh(metrics["model"], metrics["auc_pr"], color=colors_pr, edgecolor="#243447", linewidth=0.7)
axes[1].set_xlim(0.80, 1.0)
axes[1].set_xlabel("Area under precision-recall curve")
axes[1].set_title("(B) Precision-recall performance", loc="left", fontweight="bold")

for axis, values in [(axes[0], metrics["roc_auc"]), (axes[1], metrics["auc_pr"])]:
    axis.grid(axis="x", alpha=0.22, linewidth=0.7)
    axis.set_axisbelow(True)
    for row_index, value in enumerate(values):
        axis.text(value - 0.005, row_index, f"{value:.3f}", va="center", ha="right", color="#15202B", fontweight="bold", fontsize=8)

figure.suptitle("Published aggregate model performance", fontweight="bold", fontsize=13)
figure.text(0.5, -0.02, "Aggregate research results only; no patient-level data are shown.", ha="center", fontsize=9, color="#4C566A")
figure.savefig(ROOT / "results" / "published_model_performance.png", dpi=300, bbox_inches="tight")
plt.close(figure)
