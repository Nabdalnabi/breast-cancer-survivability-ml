"""Build the public notebook without copying cells or outputs from the source notebook."""

from __future__ import annotations

import json
from pathlib import Path


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


cells = [
    markdown("""# Privacy-Safe Breast Cancer Survivability Modeling Demo

This notebook demonstrates the public modeling workflow with **fully synthetic data**. It contains no clinical records, dates, identifiers, record-level outputs, or serialized models trained on the governed cohort.
"""),
    markdown("""## tl;dr

- Six classifiers are compared with a locked, stratified holdout set.
- SMOTE is applied only inside training folds through an imbalanced-learn pipeline.
- Metrics produced here are illustrative synthetic-demo results, not clinical-study results.
- Published aggregate results are documented separately in `results/published_model_performance.csv`.
"""),
    markdown("""## Context & Methods

The workflow separates data generation, splitting, training, evaluation, and visualization. Fixed seeds make the demonstration reproducible.

### Key Assumptions

The synthetic generator tests software behavior only. It does not reproduce the clinical cohort's distributions, dependencies, or treatment processes, and its outputs have no clinical interpretation.
"""),
    code("""from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

from breast_cancer_ml.data import FEATURE_NAMES, TARGET, make_synthetic_cohort
from breast_cancer_ml.evaluation import evaluate_models
from breast_cancer_ml.modeling import build_models
from breast_cancer_ml.plotting import save_curve_figures

RANDOM_STATE = 42
OUTPUT_DIR = Path(\"../artifacts\")
OUTPUT_DIR.mkdir(exist_ok=True)
"""),
    markdown("""## Data

### 1. Generate an identifier-free cohort

Only schema-level information and aggregate checks are displayed. Row-level previews are intentionally omitted.
"""),
    code("""cohort = make_synthetic_cohort(n_samples=401, random_state=RANDOM_STATE)
pd.Series({
    \"observations\": len(cohort),
    \"predictors\": len(FEATURE_NAMES),
    \"positive_class_rate\": cohort[TARGET].mean(),
    \"missing_values\": int(cohort.isna().sum().sum()),
}, name=\"synthetic_cohort_check\")
"""),
    markdown("""### 2. Lock the holdout set before resampling

Stratification preserves the outcome ratio. Resampling occurs later, inside each model pipeline, and therefore never modifies the holdout observations.
"""),
    code("""x_train, x_test, y_train, y_test = train_test_split(
    cohort[FEATURE_NAMES],
    cohort[TARGET],
    test_size=0.20,
    stratify=cohort[TARGET],
    random_state=RANDOM_STATE,
)

pd.DataFrame({
    \"partition\": [\"training\", \"holdout\"],
    \"n\": [len(x_train), len(x_test)],
    \"positive_class_rate\": [y_train.mean(), y_test.mean()],
})
"""),
    markdown("""## Results

### 3. Compare leakage-safe pipelines

Cross-validation is limited to the training partition. The holdout set is used once for final aggregate evaluation.
"""),
    code("""metrics, fitted_models = evaluate_models(
    build_models(RANDOM_STATE),
    x_train,
    y_train,
    x_test,
    y_test,
    random_state=RANDOM_STATE,
)

metrics.round(3)
"""),
    markdown("""### 4. Save aggregate evaluation figures

The curves reveal discrimination and class-imbalance behavior without exposing record-level predictions.
"""),
    code("""save_curve_figures(fitted_models, x_test, y_test, OUTPUT_DIR)
metrics.to_csv(OUTPUT_DIR / \"synthetic_demo_metrics.csv\", index=False)
sorted(path.name for path in OUTPUT_DIR.iterdir())
"""),
    markdown("""## Takeaways

This notebook demonstrates reproducible model comparison and explainability-ready design while protecting the governed dataset. Synthetic-demo performance is expected to differ from the publication and must not be interpreted clinically. For the research findings, consult the associated peer-reviewed article and the aggregate table in this repository.
"""),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

destination = Path(__file__).parents[1] / "notebooks" / "01_privacy_safe_modeling_demo.ipynb"
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
