# Explainable Machine Learning for Breast Cancer Survivability

Privacy-preserving, recruiter-facing implementation accompanying the study **“Impact of Tumor Location on Predicting Breast Cancer Patient Survivability Using Explainable Machine Learning Models.”**

## Project overview

This repository demonstrates an end-to-end binary-classification workflow for five-year survivability modeling. It emphasizes reproducible model comparison, leakage-safe class-imbalance handling, clinically readable evaluation, and explainability.

The original clinical dataset is **not included**. The runnable example generates fully synthetic observations that do not correspond to real patients. Published aggregate results are retained only as a documented research outcome and are not regenerated from the synthetic sample.

## Highlights

- Compares XGBoost, random forest, decision tree, logistic regression, support vector classification, and AdaBoost.
- Uses a stratified holdout set and cross-validation on training data only.
- Applies SMOTE inside each training fold through an imbalanced-learn pipeline, preventing train-test contamination.
- Reports ROC-AUC, average precision, precision, recall, F1 score, and accuracy.
- Supports global feature importance and SHAP explanations for the XGBoost model.
- Uses fixed random seeds, command-line configuration, validation checks, and automated tests.

## Repository structure

```text
.
├── notebooks/01_privacy_safe_modeling_demo.ipynb
├── results/published_model_performance.csv
├── src/breast_cancer_ml/
│   ├── data.py
│   ├── evaluation.py
│   ├── modeling.py
│   └── plotting.py
├── tests/
├── CITATION.cff
├── LICENSE
└── requirements.txt
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
python -m breast_cancer_ml.run_demo --output-dir artifacts
pytest
```

For an optional **synthetic-only** SHAP importance plot, run
`python -m breast_cancer_ml.run_demo --output-dir artifacts --shap`.
The 20 generic synthetic predictors are not clinical variables; their importance
must not be interpreted as a finding about breast cancer. Model ranking uses
training-only cross-validation macro-F1, not holdout performance. The holdout
comparison is descriptive and must not be reused for tuning.

The demonstration creates a synthetic cohort, trains all six models, writes aggregate metrics, and saves ROC and precision-recall figures. Synthetic-demo metrics are illustrative and should not be compared with the published clinical results.

On macOS, XGBoost requires an OpenMP runtime (normally `brew install libomp`).
The restored package passed two tests and a complete six-model synthetic run,
including ROC, precision-recall, and SHAP figures, using Python 3.12 and XGBoost
2.1.4. No clinical training or reproduction was performed. Legacy root-level
notebook/result files are retained; the folders above are the canonical layout.

## Published aggregate results

The following values are aggregate model-level results reported by the research workflow. They contain no row-level data.

| Model | Accuracy | ROC-AUC | AUC-PR |
|---|---:|---:|---:|
| XGBoost | 0.951 | 0.981 | 0.975 |
| Random forest | 0.910 | 0.972 | 0.961 |
| Decision tree | 0.812 | 0.808 | 0.849 |
| Logistic regression | 0.896 | 0.953 | 0.922 |
| Support vector classifier | 0.917 | 0.950 | 0.916 |
| AdaBoost | 0.931 | 0.960 | 0.915 |

These results are presented for research documentation. The cleaned public implementation intentionally uses a stricter leakage-safe workflow, so it does not claim to reproduce these values without the governed source data.

## Privacy and responsible use

- No clinical records, dates, identifiers, record-level predictions, trained models, or source-data extracts are included.
- The synthetic generator is designed for software demonstration, not clinical simulation.
- This code is not a medical device and must not be used for diagnosis, prognosis, or treatment decisions.
- Authorized researchers should keep governed data outside the repository and provide it through a local, access-controlled path.

## Publication

N. Abdalnabi et al., “Impact of Tumor Location on Predicting Breast Cancer Patient Survivability Using Explainable Machine Learning Models,” *JCO Clinical Cancer Informatics*, 2025. [Article](https://ascopubs.org/doi/full/10.1200/CCI-24-00178)

## License

Code is released under the MIT License. Publication text, figures, and clinical data remain subject to their respective rights and governance requirements.
