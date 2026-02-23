"""
Explainability & Interpretation Module
Applies 4 XAI techniques:
  1. SHAP (TreeExplainer) - Global and local explanations
  2. Feature Importance (XGBoost built-in: weight, gain, cover)
  3. Partial Dependence Plots (PDP + ICE)
  4. LIME - Local interpretable model-agnostic explanations
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import lime
import lime.lime_tabular
from sklearn.inspection import PartialDependenceDisplay
import joblib
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", font_scale=1.1)

# ─── Paths ───────────────────────────────────────────────────────────────────

DATA_DIR = Path("data")
MODELS_DIR = Path("models")
PLOTS_DIR = Path("plots")

# ─── Data Loading ────────────────────────────────────────────────────────────


def load_data():
    """Load model, data, and feature names."""
    model = joblib.load(MODELS_DIR / "xgboost_model.joblib")
    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    X_test = pd.read_csv(DATA_DIR / "X_test.csv")
    y_test = pd.read_csv(DATA_DIR / "y_test.csv").squeeze()
    feature_names = joblib.load(MODELS_DIR / "feature_names.joblib")
    encoders = joblib.load(MODELS_DIR / "encoders.joblib")

    print(f"Model loaded. Features: {feature_names}")
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    return model, X_train, X_test, y_test, feature_names, encoders


def get_example_indices(y_test):
    """Get indices for low, mid, and high price examples."""
    y_sorted = y_test.sort_values()
    n = len(y_sorted)
    low_idx = y_sorted.index[n // 10]       # 10th percentile
    mid_idx = y_sorted.index[n // 2]        # median
    high_idx = y_sorted.index[9 * n // 10]  # 90th percentile
    return {"low": low_idx, "mid": mid_idx, "high": high_idx}


# ═══════════════════════════════════════════════════════════════════════════
# 1. SHAP ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════


def compute_shap_values(model, X_train, X_test):
    """Compute SHAP values using TreeExplainer."""
    print("\n  Computing SHAP values (TreeExplainer)...")
    explainer = shap.TreeExplainer(model, data=X_train)
    shap_values = explainer(X_test)

    # Save for reuse
    joblib.dump(shap_values, MODELS_DIR / "shap_values.joblib")
    print(f"  SHAP values computed for {X_test.shape[0]} test samples")
    return shap_values


def plot_shap_summary_beeswarm(shap_values, save_path):
    """SHAP beeswarm summary plot (global feature importance with direction)."""
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, show=False, plot_size=(12, 8))
    plt.title("SHAP Summary Plot (Beeswarm)", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_shap_bar(shap_values, save_path):
    """SHAP bar plot (mean absolute SHAP values)."""
    plt.figure(figsize=(10, 6))
    shap.plots.bar(shap_values, show=False)
    plt.title("SHAP Feature Importance (Mean |SHAP Value|)", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_shap_waterfall(shap_values, index, label, save_path):
    """SHAP waterfall plot for a single prediction."""
    plt.figure(figsize=(10, 6))
    # Find position in shap_values array
    test_idx = list(shap_values.data[:, 0]).index(shap_values.data[index, 0]) if index >= len(shap_values) else index
    shap.plots.waterfall(shap_values[index], show=False)
    plt.title(f"SHAP Waterfall - {label} Price Property", fontsize=13, pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_shap_force(shap_values, index, label, save_path):
    """SHAP force plot for a single prediction."""
    shap.force_plot(
        shap_values.base_values[index],
        shap_values.values[index],
        shap_values.data[index],
        feature_names=shap_values.feature_names,
        matplotlib=True,
        show=False,
    )
    plt.title(f"SHAP Force Plot - {label} Price Property", fontsize=13, pad=40)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_shap_dependence(shap_values, feature_idx, feature_name, save_path):
    """SHAP dependence plot for a specific feature."""
    plt.figure(figsize=(8, 6))
    shap.dependence_plot(
        feature_idx,
        shap_values.values,
        shap_values.data,
        feature_names=shap_values.feature_names,
        show=False,
    )
    plt.title(f"SHAP Dependence: {feature_name}", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def run_shap_analysis(model, X_train, X_test, y_test, feature_names):
    """Run all SHAP analyses."""
    print("\n" + "=" * 50)
    print("1. SHAP ANALYSIS")
    print("=" * 50)

    shap_values = compute_shap_values(model, X_train, X_test)

    # Global plots
    plot_shap_summary_beeswarm(shap_values, PLOTS_DIR / "shap_summary_beeswarm.png")
    plot_shap_bar(shap_values, PLOTS_DIR / "shap_bar_importance.png")

    # Local plots for 3 examples
    examples = get_example_indices(y_test)
    for label, orig_idx in examples.items():
        # Map original index to position in test set
        test_positions = list(y_test.index)
        if orig_idx in test_positions:
            pos = test_positions.index(orig_idx)
        else:
            pos = {"low": 0, "mid": len(y_test) // 2, "high": len(y_test) - 1}[label]

        plot_shap_waterfall(
            shap_values, pos, label.capitalize(),
            PLOTS_DIR / f"shap_waterfall_{label}_price.png"
        )
        plot_shap_force(
            shap_values, pos, label.capitalize(),
            PLOTS_DIR / f"shap_force_{label}_price.png"
        )

    # Dependence plots for top 5 features
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[-5:][::-1]
    for idx in top_indices:
        fname = feature_names[idx]
        plot_shap_dependence(
            shap_values, idx, fname,
            PLOTS_DIR / f"shap_dependence_{fname}.png"
        )

    return shap_values


# ═══════════════════════════════════════════════════════════════════════════
# 2. XGBOOST BUILT-IN FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════════════════════


def plot_xgboost_importance(model, feature_names, save_path):
    """Plot XGBoost built-in feature importance (weight, gain, cover)."""
    print("\n" + "=" * 50)
    print("2. XGBOOST FEATURE IMPORTANCE")
    print("=" * 50)

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    importance_types = ["weight", "gain", "cover"]
    titles = [
        "Feature Importance (Weight)\n# of splits using feature",
        "Feature Importance (Gain)\nAvg. gain per split",
        "Feature Importance (Cover)\nAvg. coverage per split",
    ]

    for i, (imp_type, title) in enumerate(zip(importance_types, titles)):
        booster = model.get_booster()
        scores = booster.get_score(importance_type=imp_type)

        # Map feature indices (f0, f1, ...) to actual names
        mapped = {}
        for k, v in scores.items():
            if k.startswith("f"):
                idx = int(k[1:])
                if idx < len(feature_names):
                    mapped[feature_names[idx]] = v
            else:
                mapped[k] = v

        # Sort
        sorted_items = sorted(mapped.items(), key=lambda x: x[1])
        names = [item[0] for item in sorted_items]
        values = [item[1] for item in sorted_items]

        axes[i].barh(names, values, color=plt.cm.viridis(np.linspace(0.3, 0.9, len(names))))
        axes[i].set_title(title, fontsize=12)
        axes[i].set_xlabel(imp_type.capitalize())

    plt.suptitle("XGBoost Built-in Feature Importance", fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ═══════════════════════════════════════════════════════════════════════════
# 3. PARTIAL DEPENDENCE PLOTS (PDP)
# ═══════════════════════════════════════════════════════════════════════════


def plot_pdp(model, X_train, feature_names, save_path):
    """Partial Dependence Plots with ICE lines."""
    print("\n" + "=" * 50)
    print("3. PARTIAL DEPENDENCE PLOTS")
    print("=" * 50)

    # Select features for PDP
    pdp_features = []
    for fname in ["bedrooms", "bathrooms", "log_house_size",
                   "log_land_size", "district_encoded", "total_rooms"]:
        if fname in feature_names:
            pdp_features.append(feature_names.index(fname))

    if len(pdp_features) < 2:
        print("  Not enough features for PDP. Skipping.")
        return

    n_features = len(pdp_features)
    n_cols = min(3, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else list(axes)
    else:
        axes = axes.flatten().tolist()

    # Use subsample for speed
    sample_size = min(500, len(X_train))

    PartialDependenceDisplay.from_estimator(
        model, X_train,
        features=pdp_features,
        kind="both",  # PDP + ICE
        subsample=sample_size,
        n_jobs=-1,
        ax=axes[:n_features],
        random_state=42,
    )

    # Remove extra axes
    for i in range(n_features, len(axes)):
        fig.delaxes(axes[i])

    plt.suptitle("Partial Dependence Plots (PDP + ICE)", fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_pdp_interactions(model, X_train, feature_names, save_path):
    """2D PDP interaction plots for key feature pairs."""
    # Define interaction pairs
    pairs = []
    pair_names = [
        ("bedrooms", "bathrooms"),
        ("log_house_size", "log_land_size"),
        ("bedrooms", "district_encoded"),
    ]

    for f1, f2 in pair_names:
        if f1 in feature_names and f2 in feature_names:
            pairs.append((feature_names.index(f1), feature_names.index(f2)))

    if not pairs:
        print("  No valid feature pairs for interaction PDP. Skipping.")
        return

    fig, axes = plt.subplots(1, len(pairs), figsize=(7 * len(pairs), 6))
    if len(pairs) == 1:
        axes = [axes]

    PartialDependenceDisplay.from_estimator(
        model, X_train,
        features=pairs,
        kind="average",
        subsample=min(300, len(X_train)),
        n_jobs=-1,
        ax=axes,
        random_state=42,
    )

    plt.suptitle("2D Partial Dependence Interactions", fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ═══════════════════════════════════════════════════════════════════════════
# 4. LIME EXPLANATIONS
# ═══════════════════════════════════════════════════════════════════════════


def run_lime_analysis(model, X_train, X_test, y_test, feature_names):
    """Generate LIME explanations for example instances."""
    print("\n" + "=" * 50)
    print("4. LIME EXPLANATIONS")
    print("=" * 50)

    # LIME can't handle NaN — fill with median for LIME only
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy="median")
    X_train_filled = pd.DataFrame(
        imputer.fit_transform(X_train), columns=feature_names
    )
    X_test_filled = pd.DataFrame(
        imputer.transform(X_test), columns=feature_names
    )

    # Wrap model to predict on imputed data
    def predict_fn(X):
        df = pd.DataFrame(X, columns=feature_names)
        return model.predict(df)

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train_filled.values,
        feature_names=feature_names,
        mode="regression",
        discretize_continuous=False,
        random_state=42,
    )

    examples = get_example_indices(y_test)
    test_positions = list(y_test.index)

    for label, orig_idx in examples.items():
        if orig_idx in test_positions:
            pos = test_positions.index(orig_idx)
        else:
            pos = {"low": 0, "mid": len(y_test) // 2, "high": len(y_test) - 1}[label]

        explanation = explainer.explain_instance(
            X_test_filled.iloc[pos].values,
            predict_fn,
            num_features=len(feature_names),
        )

        fig = explanation.as_pyplot_figure()
        fig.set_size_inches(10, 6)
        plt.title(f"LIME Explanation - {label.capitalize()} Price Property", fontsize=13)
        plt.tight_layout()
        save_path = PLOTS_DIR / f"lime_explanation_{label}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {save_path}")


# ─── Main ────────────────────────────────────────────────────────────────────


def main():
    """Run all explainability analyses."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("EXPLAINABILITY & INTERPRETATION PIPELINE")
    print("=" * 60)

    # Load
    model, X_train, X_test, y_test, feature_names, encoders = load_data()

    # 1. SHAP
    shap_values = run_shap_analysis(model, X_train, X_test, y_test, feature_names)

    # 2. XGBoost Feature Importance
    plot_xgboost_importance(model, feature_names, PLOTS_DIR / "xgboost_feature_importance.png")

    # 3. Partial Dependence Plots
    plot_pdp(model, X_train, feature_names, PLOTS_DIR / "pdp_individual.png")
    plot_pdp_interactions(model, X_train, feature_names, PLOTS_DIR / "pdp_interactions.png")

    # 4. LIME
    run_lime_analysis(model, X_train, X_test, y_test, feature_names)

    print("\n" + "=" * 60)
    print("EXPLAINABILITY ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"All plots saved to {PLOTS_DIR}/")
    print("\nXAI Techniques Applied:")
    print("  1. SHAP (TreeExplainer) - Global summary, bar, waterfall, force, dependence")
    print("  2. XGBoost Feature Importance - Weight, Gain, Cover")
    print("  3. Partial Dependence Plots - Individual PDP+ICE, 2D Interactions")
    print("  4. LIME - Local explanations for low/mid/high price examples")


if __name__ == "__main__":
    main()
