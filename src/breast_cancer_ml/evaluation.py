"""Rank models using training-only CV; holdout results are descriptive."""
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, average_precision_score,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score


def evaluate_models(models, x_train, y_train, x_test, y_test, random_state=42):
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
    rows, fitted = [], {}
    for name, pipeline in models.items():
        scores = cross_val_score(pipeline, x_train, y_train, cv=cv,
                                 scoring="f1_macro", error_score="raise")
        model = clone(pipeline).fit(x_train, y_train)
        pred = model.predict(x_test)
        prob = model.predict_proba(x_test)[:, list(model.classes_).index(1)]
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, pred, average="binary", zero_division=0)
        rows.append({"model": name, "cv_macro_f1": scores.mean(),
                     "cv_macro_f1_std": scores.std(), "accuracy": accuracy_score(y_test, pred),
                     "precision": precision, "recall": recall, "f1": f1,
                     "roc_auc": roc_auc_score(y_test, prob),
                     "average_precision": average_precision_score(y_test, prob)})
        fitted[name] = model
    return pd.DataFrame(rows).sort_values("cv_macro_f1", ascending=False), fitted
