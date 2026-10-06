"""Run the self-contained synthetic example without accessing clinical data."""
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split
from .data import FEATURE_NAMES, TARGET, make_synthetic_cohort
from .evaluation import evaluate_models
from .modeling import build_models
from .plotting import save_curve_figures, save_shap_importance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--samples", type=int, default=401)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shap", action="store_true")
    args = parser.parse_args()
    frame = make_synthetic_cohort(args.samples, args.seed)
    x_train, x_test, y_train, y_test = train_test_split(
        frame[FEATURE_NAMES], frame[TARGET], stratify=frame[TARGET],
        test_size=0.2, random_state=args.seed)
    metrics, fitted = evaluate_models(build_models(args.seed), x_train, y_train,
                                      x_test, y_test, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output_dir / "synthetic_demo_metrics.csv", index=False)
    save_curve_figures(fitted, x_test, y_test, args.output_dir)
    if args.shap:
        save_shap_importance(fitted["XGBoost"], x_test, args.output_dir)
    print("Synthetic software demonstration, not clinical performance.")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
