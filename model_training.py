"""
Model Training Pipeline
Trains baseline models (Linear Regression, Ridge, Random Forest)
and XGBoost with hyperparameter tuning using RandomizedSearchCV.
"""

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ─── Paths ───────────────────────────────────────────────────────────────────

DATA_DIR = Path("data")
MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")


# ─── Data Loading ────────────────────────────────────────────────────────────


def load_splits():
    """Load train/val/test splits."""
    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    X_val = pd.read_csv(DATA_DIR / "X_val.csv")
    X_test = pd.read_csv(DATA_DIR / "X_test.csv")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze()
    y_val = pd.read_csv(DATA_DIR / "y_val.csv").squeeze()
    y_test = pd.read_csv(DATA_DIR / "y_test.csv").squeeze()

    feature_names = joblib.load(MODELS_DIR / "feature_names.joblib")

    print(f"Data loaded:")
    print(f"  Train: {X_train.shape}")
    print(f"  Val:   {X_val.shape}")
    print(f"  Test:  {X_test.shape}")
    print(f"  Features: {feature_names}")

    return X_train, X_val, X_test, y_train, y_val, y_test, feature_names


# ─── Metrics ─────────────────────────────────────────────────────────────────


def evaluate_model(model, X, y_true_log, split_name=""):
    """Evaluate model on both log and original price scale."""
    y_pred_log = model.predict(X)

    # Log-scale metrics
    rmse_log = np.sqrt(mean_squared_error(y_true_log, y_pred_log))
    mae_log = mean_absolute_error(y_true_log, y_pred_log)
    r2_log = r2_score(y_true_log, y_pred_log)

    # Original scale metrics (LKR)
    y_true_lkr = np.expm1(y_true_log)
    y_pred_lkr = np.expm1(y_pred_log)

    rmse_lkr = np.sqrt(mean_squared_error(y_true_lkr, y_pred_lkr))
    mae_lkr = mean_absolute_error(y_true_lkr, y_pred_lkr)
    r2_lkr = r2_score(y_true_lkr, y_pred_lkr)

    # MAPE (avoid division by zero)
    nonzero = y_true_lkr > 0
    mape = np.mean(np.abs(
        (y_true_lkr[nonzero] - y_pred_lkr[nonzero]) / y_true_lkr[nonzero]
    )) * 100

    return {
        f"{split_name}_rmse_log": round(rmse_log, 4),
        f"{split_name}_mae_log": round(mae_log, 4),
        f"{split_name}_r2_log": round(r2_log, 4),
        f"{split_name}_rmse_lkr": round(rmse_lkr, 0),
        f"{split_name}_mae_lkr": round(mae_lkr, 0),
        f"{split_name}_r2_lkr": round(r2_lkr, 4),
        f"{split_name}_mape": round(mape, 2),
    }


# ─── Baseline Models ────────────────────────────────────────────────────────


def train_baselines(X_train, y_train, X_val, y_val):
    """Train baseline models for comparison."""
    # Linear models need imputation (can't handle NaN)
    # Tree-based models handle NaN natively
    baselines = {
        "Linear Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", LinearRegression()),
        ]),
        "Ridge Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", Ridge(alpha=1.0)),
        ]),
        "Random Forest": HistGradientBoostingRegressor(
            max_iter=200,
            max_depth=12,
            min_samples_leaf=5,
            random_state=42,
        ),
    }

    results = {}

    for name, model in baselines.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)

        # Evaluate on train and validation
        train_metrics = evaluate_model(model, X_train, y_train, "train")
        val_metrics = evaluate_model(model, X_val, y_val, "val")

        results[name] = {**train_metrics, **val_metrics}

        # Save model
        safe_name = name.lower().replace(" ", "_")
        joblib.dump(model, MODELS_DIR / f"baseline_{safe_name}.joblib")

        print(f"  Train R2: {train_metrics['train_r2_log']:.4f} | "
              f"Val R2: {val_metrics['val_r2_log']:.4f}")
        print(f"  Val RMSE (LKR): Rs {val_metrics['val_rmse_lkr']:,.0f}")
        print(f"  Val MAPE: {val_metrics['val_mape']:.1f}%")

    return results


# ─── XGBoost Hyperparameter Tuning ───────────────────────────────────────────


def tune_xgboost(X_train, y_train):
    """Tune XGBoost using RandomizedSearchCV."""
    print("\n" + "=" * 60)
    print("XGBoost Hyperparameter Tuning (RandomizedSearchCV)")
    print("=" * 60)

    param_distributions = {
        "n_estimators": [100, 200, 300, 500, 800],
        "max_depth": [3, 4, 5, 6, 7, 8],
        "learning_rate": [0.01, 0.05, 0.1, 0.15, 0.2],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "min_child_weight": [1, 3, 5, 7],
        "reg_alpha": [0, 0.01, 0.1, 1.0],
        "reg_lambda": [0, 0.01, 0.1, 1.0],
        "gamma": [0, 0.1, 0.3, 0.5],
    }

    xgb_base = XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )

    search = RandomizedSearchCV(
        xgb_base,
        param_distributions=param_distributions,
        n_iter=50,
        scoring="neg_root_mean_squared_error",
        cv=5,
        random_state=42,
        verbose=1,
        n_jobs=-1,
    )

    print("Searching 50 random hyperparameter combinations with 5-fold CV...")
    search.fit(X_train, y_train)

    best_params = search.best_params_
    best_score = -search.best_score_

    print(f"\nBest CV RMSE (log-scale): {best_score:.4f}")
    print(f"Best parameters:")
    for k, v in sorted(best_params.items()):
        print(f"  {k}: {v}")

    # Save best params
    with open(MODELS_DIR / "best_params.json", "w") as f:
        json.dump(best_params, f, indent=2)

    return best_params


def train_final_xgboost(X_train, y_train, X_val, y_val, best_params):
    """Train final XGBoost model with best params and early stopping."""
    print("\n" + "=" * 60)
    print("Training Final XGBoost Model")
    print("=" * 60)

    final_model = XGBRegressor(
        **best_params,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )

    final_model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=50,
    )

    # Save model
    joblib.dump(final_model, MODELS_DIR / "xgboost_model.joblib")
    print(f"\nModel saved: {MODELS_DIR / 'xgboost_model.joblib'}")

    return final_model


# ─── Model Comparison ────────────────────────────────────────────────────────


def compare_all_models(baseline_results, xgb_model, X_train, y_train, X_val, y_val, X_test, y_test):
    """Create final comparison table across all models."""
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    # XGBoost metrics
    xgb_train = evaluate_model(xgb_model, X_train, y_train, "train")
    xgb_val = evaluate_model(xgb_model, X_val, y_val, "val")
    xgb_test = evaluate_model(xgb_model, X_test, y_test, "test")
    baseline_results["XGBoost"] = {**xgb_train, **xgb_val, **xgb_test}

    # Also evaluate baselines on test set
    for name in ["Linear Regression", "Ridge Regression", "Random Forest"]:
        safe_name = name.lower().replace(" ", "_")
        model = joblib.load(MODELS_DIR / f"baseline_{safe_name}.joblib")
        test_metrics = evaluate_model(model, X_test, y_test, "test")
        baseline_results[name].update(test_metrics)

    # Create comparison DataFrame
    rows = []
    for model_name, metrics in baseline_results.items():
        rows.append({
            "Model": model_name,
            "Val R2": metrics.get("val_r2_log", ""),
            "Val RMSE (log)": metrics.get("val_rmse_log", ""),
            "Val MAE (LKR)": metrics.get("val_mae_lkr", ""),
            "Val MAPE (%)": metrics.get("val_mape", ""),
            "Test R2": metrics.get("test_r2_log", ""),
            "Test RMSE (log)": metrics.get("test_rmse_log", ""),
            "Test MAE (LKR)": metrics.get("test_mae_lkr", ""),
            "Test MAPE (%)": metrics.get("test_mape", ""),
        })

    comparison = pd.DataFrame(rows)
    comparison = comparison.sort_values("Test R2", ascending=False)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)

    print("\n" + comparison.to_string(index=False))
    print(f"\nComparison saved to {RESULTS_DIR / 'model_comparison.csv'}")

    return comparison


# ─── Main ────────────────────────────────────────────────────────────────────


def main():
    """Run the full training pipeline."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MODEL TRAINING PIPELINE")
    print("=" * 60)

    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names = load_splits()

    # Train baselines
    print("\n--- Baseline Models ---")
    baseline_results = train_baselines(X_train, y_train, X_val, y_val)

    # Tune XGBoost
    best_params = tune_xgboost(X_train, y_train)

    # Train final model
    xgb_model = train_final_xgboost(X_train, y_train, X_val, y_val, best_params)

    # Compare
    comparison = compare_all_models(
        baseline_results, xgb_model,
        X_train, y_train, X_val, y_val, X_test, y_test,
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best model: XGBoost")
    xgb_test = evaluate_model(xgb_model, X_test, y_test, "test")
    print(f"Test R2: {xgb_test['test_r2_log']:.4f}")
    print(f"Test RMSE (LKR): Rs {xgb_test['test_rmse_lkr']:,.0f}")
    print(f"Test MAPE: {xgb_test['test_mape']:.1f}%")


if __name__ == "__main__":
    main()
