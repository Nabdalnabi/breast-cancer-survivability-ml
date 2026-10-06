"""Aggregate plots from synthetic observations only."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay


def save_curve_figures(models, x_test, y_test, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, display in [("roc", RocCurveDisplay), ("precision_recall", PrecisionRecallDisplay)]:
        fig, ax = plt.subplots(figsize=(8, 6))
        for label, model in models.items():
            display.from_estimator(model, x_test, y_test, name=label, ax=ax)
        ax.set_title("Synthetic demonstration: " + name.replace("_", " "))
        fig.tight_layout()
        fig.savefig(output_dir / f"synthetic_{name}.png", dpi=150)
        plt.close(fig)


def save_shap_importance(pipeline, x, output_dir):
    import shap
    transformed = pipeline.named_steps["scaler"].transform(
        pipeline.named_steps["imputer"].transform(x))
    values = shap.TreeExplainer(pipeline.named_steps["model"])(transformed)
    importance = np.abs(values.values).mean(axis=0)
    order = np.argsort(importance)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(np.asarray(x.columns)[order], importance[order])
    ax.set_xlabel("Mean absolute SHAP value (model raw output)")
    ax.set_title("Synthetic demo only: XGBoost explanations")
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "synthetic_shap_importance.png", dpi=150)
    plt.close(fig)
