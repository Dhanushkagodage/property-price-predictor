"""
Model Evaluation Module
Computes detailed metrics, generates evaluation plots (actual vs predicted,
residuals, learning curves, model comparison, error by price range).
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import learning_curve
import joblib
from pathlib import Path

# ─── Paths ───────────────────────────────────────────────────────────────────

DATA_DIR = Path("data")
MODELS_DIR = Path("models")
PLOTS_DIR = Path("plots")
RESULTS_DIR = Path("results")

# ─── Style ───────────────────────────────────────────────────────────────────

sns.set_theme(style="whitegrid", font_scale=1.1)
COLORS = {
    "primary": "#2196F3",
    "secondary": "#FF5722",
    "accent": "#4CAF50",
    "warning": "#FFC107",
}


# ─── Metrics ─────────────────────────────────────────────────────────────────


def calculate_metrics(y_true_log, y_pred_log):
    """Calculate comprehensive metrics on both scales."""
    # Log-scale
    rmse_log = np.sqrt(mean_squared_error(y_true_log, y_pred_log))
    mae_log = mean_absolute_error(y_true_log, y_pred_log)
    r2_log = r2_score(y_true_log, y_pred_log)

    # Original scale (LKR)
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)

    rmse_lkr = np.sqrt(mean_squared_error(y_true, y_pred))
    mae_lkr = mean_absolute_error(y_true, y_pred)
    r2_lkr = r2_score(y_true, y_pred)

    nonzero = y_true > 0
    mape = np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100

    return {
        "RMSE (log)": rmse_log,
        "MAE (log)": mae_log,
        "R2 (log)": r2_log,
        "RMSE (LKR)": rmse_lkr,
        "MAE (LKR)": mae_lkr,
        "R2 (LKR)": r2_lkr,
        "MAPE (%)": mape,
    }


# ─── Visualization Functions ────────────────────────────────────────────────


def plot_actual_vs_predicted(y_true_log, y_pred_log, split_name, save_path):
    """Scatter plot of actual vs predicted prices."""
    y_true = np.expm1(y_true_log) / 1e6  # to millions
    y_pred = np.expm1(y_pred_log) / 1e6

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_true, y_pred, alpha=0.3, s=10, color=COLORS["primary"])

    # Perfect prediction line
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([0, max_val], [0, max_val], "r--", linewidth=2, label="Perfect Prediction")

    r2 = r2_score(y_true, y_pred)
    ax.text(0.05, 0.92, f"R² = {r2:.4f}",
            transform=ax.transAxes, fontsize=14,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    ax.set_xlabel("Actual Price (Million LKR)", fontsize=12)
    ax.set_ylabel("Predicted Price (Million LKR)", fontsize=12)
    ax.set_title(f"Actual vs Predicted Property Prices ({split_name})", fontsize=14)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_residuals(y_true_log, y_pred_log, save_path):
    """Residual scatter plot and histogram."""
    y_true = np.expm1(y_true_log) / 1e6
    y_pred = np.expm1(y_pred_log) / 1e6
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Residuals vs Predicted
    axes[0].scatter(y_pred, residuals, alpha=0.3, s=10, color=COLORS["primary"])
    axes[0].axhline(y=0, color="red", linestyle="--", linewidth=2)
    axes[0].set_xlabel("Predicted Price (Million LKR)", fontsize=12)
    axes[0].set_ylabel("Residual (Million LKR)", fontsize=12)
    axes[0].set_title("Residuals vs Predicted Values", fontsize=13)

    # Residual histogram
    axes[1].hist(residuals, bins=50, color=COLORS["primary"],
                 alpha=0.7, edgecolor="white")
    axes[1].axvline(x=0, color="red", linestyle="--", linewidth=2)
    axes[1].set_xlabel("Residual (Million LKR)", fontsize=12)
    axes[1].set_ylabel("Frequency", fontsize=12)
    axes[1].set_title("Residual Distribution", fontsize=13)

    mean_res = residuals.mean()
    std_res = residuals.std()
    axes[1].text(0.65, 0.85,
                 f"Mean: {mean_res:.2f}M\nStd: {std_res:.2f}M",
                 transform=axes[1].transAxes, fontsize=11,
                 bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_learning_curves(model, X_train, y_train, save_path):
    """Plot training and validation learning curves."""
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train, y_train,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=5,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        random_state=42,
    )

    train_mean = -train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = -val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                     alpha=0.1, color=COLORS["primary"])
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                     alpha=0.1, color=COLORS["secondary"])
    ax.plot(train_sizes, train_mean, "o-", color=COLORS["primary"],
            label="Training Score", linewidth=2)
    ax.plot(train_sizes, val_mean, "o-", color=COLORS["secondary"],
            label="Cross-Validation Score", linewidth=2)

    ax.set_xlabel("Training Set Size", fontsize=12)
    ax.set_ylabel("RMSE (log-scale)", fontsize=12)
    ax.set_title("Learning Curves - XGBoost", fontsize=14)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_error_by_price_range(y_true_log, y_pred_log, save_path):
    """Box plot of prediction errors by price range."""
    y_true = np.expm1(y_true_log) / 1e6
    y_pred = np.expm1(y_pred_log) / 1e6
    pct_error = ((y_pred - y_true) / y_true) * 100

    # Define price bins
    bins = [0, 5, 10, 25, 50, 100, 500]
    labels = ["<5M", "5-10M", "10-25M", "25-50M", "50-100M", ">100M"]
    price_range = pd.cut(y_true, bins=bins, labels=labels)

    df_err = pd.DataFrame({"Price Range": price_range, "Error (%)": pct_error})
    df_err = df_err.dropna()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df_err, x="Price Range", y="Error (%)",
                palette="viridis", ax=ax, showfliers=False)
    ax.axhline(y=0, color="red", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Price Range (LKR)", fontsize=12)
    ax.set_ylabel("Prediction Error (%)", fontsize=12)
    ax.set_title("Prediction Error by Price Range", fontsize=14)
    ax.set_ylim(-100, 100)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_model_comparison(save_path):
    """Grouped bar chart comparing all models."""
    comparison = pd.read_csv(RESULTS_DIR / "model_comparison.csv")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    models = comparison["Model"].values
    x = np.arange(len(models))
    width = 0.35

    # R2 comparison
    axes[0].bar(x - width / 2, comparison["Val R2"], width,
                label="Validation", color=COLORS["primary"])
    axes[0].bar(x + width / 2, comparison["Test R2"], width,
                label="Test", color=COLORS["secondary"])
    axes[0].set_ylabel("R² Score")
    axes[0].set_title("R² Score Comparison")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=30, ha="right")
    axes[0].legend()

    # RMSE comparison
    axes[1].bar(x - width / 2, comparison["Val RMSE (log)"], width,
                label="Validation", color=COLORS["primary"])
    axes[1].bar(x + width / 2, comparison["Test RMSE (log)"], width,
                label="Test", color=COLORS["secondary"])
    axes[1].set_ylabel("RMSE (log-scale)")
    axes[1].set_title("RMSE Comparison")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=30, ha="right")
    axes[1].legend()

    # MAPE comparison
    axes[2].bar(x - width / 2, comparison["Val MAPE (%)"], width,
                label="Validation", color=COLORS["primary"])
    axes[2].bar(x + width / 2, comparison["Test MAPE (%)"], width,
                label="Test", color=COLORS["secondary"])
    axes[2].set_ylabel("MAPE (%)")
    axes[2].set_title("MAPE Comparison")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(models, rotation=30, ha="right")
    axes[2].legend()

    plt.suptitle("Model Performance Comparison", fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def generate_evaluation_report(metrics_train, metrics_val, metrics_test):
    """Generate a text evaluation report."""
    lines = [
        "=" * 60,
        "MODEL EVALUATION REPORT",
        "=" * 60,
        f"Model: XGBoost Regressor",
        f"Target: log(1 + price)",
        "",
        "--- Training Set ---",
    ]
    for k, v in metrics_train.items():
        if "LKR" in k:
            lines.append(f"  {k}: Rs {v:,.0f}")
        else:
            lines.append(f"  {k}: {v:.4f}")

    lines.append("\n--- Validation Set ---")
    for k, v in metrics_val.items():
        if "LKR" in k:
            lines.append(f"  {k}: Rs {v:,.0f}")
        else:
            lines.append(f"  {k}: {v:.4f}")

    lines.append("\n--- Test Set ---")
    for k, v in metrics_test.items():
        if "LKR" in k:
            lines.append(f"  {k}: Rs {v:,.0f}")
        else:
            lines.append(f"  {k}: {v:.4f}")

    lines.append("\n" + "=" * 60)

    report = "\n".join(lines)
    report_path = RESULTS_DIR / "evaluation_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nReport saved: {report_path}")
    print(report)


# ─── Main ────────────────────────────────────────────────────────────────────


def main():
    """Run the full evaluation pipeline."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MODEL EVALUATION PIPELINE")
    print("=" * 60)

    # Load model and data
    model = joblib.load(MODELS_DIR / "xgboost_model.joblib")
    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    X_val = pd.read_csv(DATA_DIR / "X_val.csv")
    X_test = pd.read_csv(DATA_DIR / "X_test.csv")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze()
    y_val = pd.read_csv(DATA_DIR / "y_val.csv").squeeze()
    y_test = pd.read_csv(DATA_DIR / "y_test.csv").squeeze()

    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)

    # Metrics
    print("\nCalculating metrics...")
    metrics_train = calculate_metrics(y_train, y_pred_train)
    metrics_val = calculate_metrics(y_val, y_pred_val)
    metrics_test = calculate_metrics(y_test, y_pred_test)

    # Plots
    print("\nGenerating plots...")

    plot_actual_vs_predicted(
        y_train, y_pred_train, "Training",
        PLOTS_DIR / "actual_vs_predicted_train.png"
    )
    plot_actual_vs_predicted(
        y_test, y_pred_test, "Test",
        PLOTS_DIR / "actual_vs_predicted_test.png"
    )
    plot_residuals(
        y_test, y_pred_test,
        PLOTS_DIR / "residuals_test.png"
    )

    print("\n  Generating learning curves (this may take a few minutes)...")
    plot_learning_curves(
        model, X_train, y_train,
        PLOTS_DIR / "learning_curves.png"
    )

    plot_error_by_price_range(
        y_test, y_pred_test,
        PLOTS_DIR / "prediction_error_by_range.png"
    )

    plot_model_comparison(PLOTS_DIR / "model_comparison.png")

    # Report
    generate_evaluation_report(metrics_train, metrics_val, metrics_test)

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"All plots saved to {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
