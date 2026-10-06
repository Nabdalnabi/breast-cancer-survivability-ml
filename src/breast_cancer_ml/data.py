"""Generate software-test data, never clinical or patient-derived records."""
import pandas as pd
from sklearn.datasets import make_classification

FEATURE_NAMES = [f"synthetic_feature_{i:02d}" for i in range(1, 21)]
TARGET = "synthetic_outcome"


def make_synthetic_cohort(n_samples=401, random_state=42):
    if n_samples < 100:
        raise ValueError("Use at least 100 observations for stratified evaluation.")
    x, y = make_classification(
        n_samples=n_samples, n_features=20, n_informative=8,
        n_redundant=4, weights=[0.35, 0.65], flip_y=0.05,
        random_state=random_state,
    )
    frame = pd.DataFrame(x, columns=FEATURE_NAMES)
    frame[TARGET] = y
    return frame
