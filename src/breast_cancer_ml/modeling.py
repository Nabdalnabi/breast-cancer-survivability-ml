"""Preprocessing and resampling are refitted within every CV fold."""
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


def build_models(random_state=42):
    estimators = {
        "XGBoost": XGBClassifier(n_estimators=100, max_depth=3, n_jobs=1,
                                 eval_metric="logloss", random_state=random_state),
        "Random forest": RandomForestClassifier(n_estimators=100, n_jobs=1,
                                                 random_state=random_state),
        "Decision tree": DecisionTreeClassifier(max_depth=5, random_state=random_state),
        "Logistic regression": LogisticRegression(max_iter=1000, random_state=random_state),
        "Support vector classifier": SVC(probability=True, random_state=random_state),
        "AdaBoost": AdaBoostClassifier(random_state=random_state),
    }
    return {name: Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=random_state, k_neighbors=3)),
        ("model", estimator),
    ]) for name, estimator in estimators.items()}
