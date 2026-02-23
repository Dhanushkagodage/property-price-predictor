"""
Streamlit Web Application
Sri Lanka Property Price Predictor
Allows users to input property details and get price predictions
with SHAP-based explanations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ─── Paths ───────────────────────────────────────────────────────────────────

MODELS_DIR = Path("models")
PLOTS_DIR = Path("plots")
RESULTS_DIR = Path("results")
DATA_DIR = Path("data")

# ─── Sri Lanka Districts ─────────────────────────────────────────────────────

SRI_LANKA_DISTRICTS = [
    "Colombo", "Gampaha", "Kalutara",
    "Kandy", "Matale", "Nuwara Eliya",
    "Galle", "Matara", "Hambantota",
    "Jaffna", "Kilinochchi", "Mannar",
    "Vavuniya", "Mullaitivu",
    "Batticaloa", "Ampara", "Trincomalee",
    "Kurunegala", "Puttalam",
    "Anuradhapura", "Polonnaruwa",
    "Badulla", "Moneragala",
    "Ratnapura", "Kegalle",
]

# ─── Page Config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Sri Lanka Property Price Predictor",
    page_icon="\U0001f3e0",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Model Loading ───────────────────────────────────────────────────────────


@st.cache_resource
def load_model():
    """Load trained model and encoders."""
    model = joblib.load(MODELS_DIR / "xgboost_model.joblib")
    encoders = joblib.load(MODELS_DIR / "encoders.joblib")
    feature_names = joblib.load(MODELS_DIR / "feature_names.joblib")
    return model, encoders, feature_names


@st.cache_resource
def load_shap_explainer(_model, _X_train):
    """Create SHAP TreeExplainer (cached)."""
    return shap.TreeExplainer(_model, data=_X_train)


@st.cache_data
def load_training_data():
    """Load training data for SHAP background."""
    return pd.read_csv(DATA_DIR / "X_train.csv")


# ─── Helper Functions ────────────────────────────────────────────────────────


def encode_district(district, encoders):
    """Encode a district name using the saved encoder."""
    le = encoders["district_encoder"]
    classes = list(le.classes_)
    if district in classes:
        return le.transform([district])[0]
    return 0  # fallback to first district


def encode_property_type(ptype, encoders):
    """Encode property type using saved encoder."""
    le = encoders["property_type_encoder"]
    classes = list(le.classes_)
    # Map display name to stored name
    for cls in classes:
        if ptype.lower() in cls.lower():
            return le.transform([cls])[0]
    return 0


def build_features(district, property_type, bedrooms, bathrooms,
                   house_size, land_size, encoders, feature_names):
    """Build feature vector from user inputs."""
    district_enc = encode_district(district, encoders)
    ptype_enc = encode_property_type(property_type, encoders)

    is_apartment = 1 if property_type == "Apartment" else 0
    bed_bath_ratio = min(bedrooms / max(bathrooms, 1), 5)
    total_rooms = bedrooms + bathrooms

    # Handle optional sizes
    has_house = 1 if house_size > 0 else 0
    has_land = 1 if land_size > 0 else 0
    hs = house_size if house_size > 0 else np.nan
    ls = land_size if land_size > 0 else np.nan
    log_hs = np.log1p(hs) if has_house else np.nan
    log_ls = np.log1p(ls) if has_land else np.nan

    features = {
        "district_encoded": district_enc,
        "property_type_encoded": ptype_enc,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "house_size": hs,
        "land_size": ls,
        "bed_bath_ratio": bed_bath_ratio,
        "total_rooms": total_rooms,
        "has_house_size": has_house,
        "has_land_size": has_land,
        "is_apartment": is_apartment,
        "log_house_size": log_hs,
        "log_land_size": log_ls,
    }

    # Build DataFrame with correct column order
    df = pd.DataFrame([features])[feature_names]
    return df


def format_price(price_lkr):
    """Format price in LKR with readable units."""
    if price_lkr >= 1e6:
        return f"Rs {price_lkr/1e6:,.2f} Million"
    elif price_lkr >= 1e3:
        return f"Rs {price_lkr/1e3:,.0f}K"
    else:
        return f"Rs {price_lkr:,.0f}"


# ─── Main App ────────────────────────────────────────────────────────────────


def main():
    # Check model exists
    if not (MODELS_DIR / "xgboost_model.joblib").exists():
        st.error("Model not found! Please run the training pipeline first.")
        st.code("""
# Run these commands in order:
python scraper.py
python data_preprocessing.py
python model_training.py
python model_evaluation.py
python explainability.py
        """)
        return

    model, encoders, feature_names = load_model()

    # ─── Header ──────────────────────────────────────────────────────────
    st.title("\U0001f3e0 Sri Lanka Property Price Predictor")
    st.markdown(
        "Predict property prices across Sri Lanka using Machine Learning. "
        "Powered by **XGBoost** with **SHAP** explainability."
    )

    # ─── Sidebar: User Inputs ────────────────────────────────────────────
    st.sidebar.header("Property Details")
    st.sidebar.markdown("Enter the property specifications below:")

    # Get available districts from encoder
    available_districts = sorted(encoders.get("district_classes", SRI_LANKA_DISTRICTS))

    district = st.sidebar.selectbox(
        "District",
        options=available_districts,
        index=available_districts.index("Colombo") if "Colombo" in available_districts else 0,
    )

    property_type = st.sidebar.selectbox(
        "Property Type",
        options=["House", "Apartment"],
    )

    bedrooms = st.sidebar.slider("Bedrooms", min_value=1, max_value=10, value=3)
    bathrooms = st.sidebar.slider("Bathrooms", min_value=1, max_value=10, value=2)

    house_size = st.sidebar.number_input(
        "House Size (sqft)", min_value=0, max_value=20000,
        value=1500, step=100,
        help="Set to 0 if unknown"
    )

    land_size = st.sidebar.number_input(
        "Land Size (perches)", min_value=0.0, max_value=500.0,
        value=10.0, step=0.5,
        help="Set to 0 if unknown"
    )

    predict_btn = st.sidebar.button("\U0001f50d Predict Price", type="primary", use_container_width=True)

    # ─── Tabs ────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs([
        "\U0001f4b0 Price Prediction",
        "\U0001f4ca Model Performance",
        "\U0001f9e0 Explainability",
    ])

    # ═══ Tab 1: Price Prediction ═════════════════════════════════════════
    with tab1:
        if predict_btn:
            features_df = build_features(
                district, property_type, bedrooms, bathrooms,
                house_size, land_size, encoders, feature_names
            )

            # Predict
            log_price = model.predict(features_df)[0]
            predicted_price = np.expm1(log_price)

            # Display prediction
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.metric(
                    label="Estimated Property Price",
                    value=format_price(predicted_price),
                )
            with col2:
                st.metric("District", district)
            with col3:
                st.metric("Type", property_type)

            st.markdown("---")

            # Property summary
            st.subheader("Property Summary")
            summary_cols = st.columns(4)
            with summary_cols[0]:
                st.metric("Bedrooms", bedrooms)
            with summary_cols[1]:
                st.metric("Bathrooms", bathrooms)
            with summary_cols[2]:
                st.metric("House Size", f"{house_size} sqft" if house_size > 0 else "N/A")
            with summary_cols[3]:
                st.metric("Land Size", f"{land_size} perches" if land_size > 0 else "N/A")

            st.markdown("---")

            # SHAP explanation for this prediction
            st.subheader("Why This Price? (SHAP Explanation)")
            try:
                X_train = load_training_data()
                explainer = load_shap_explainer(model, X_train)
                shap_value = explainer(features_df)

                fig, ax = plt.subplots(figsize=(10, 5))
                shap.plots.waterfall(shap_value[0], show=False)
                plt.title("Feature Contributions to This Prediction", fontsize=13)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

                st.caption(
                    "The waterfall chart shows how each feature pushes the prediction "
                    "higher (red) or lower (blue) from the average predicted price."
                )
            except Exception as e:
                st.warning(f"Could not generate SHAP explanation: {e}")

        else:
            st.info(
                "\U0001f449 Enter property details in the sidebar and click "
                "**Predict Price** to get started."
            )

    # ═══ Tab 2: Model Performance ════════════════════════════════════════
    with tab2:
        st.subheader("Model Comparison")

        comparison_path = RESULTS_DIR / "model_comparison.csv"
        if comparison_path.exists():
            comparison = pd.read_csv(comparison_path)
            st.dataframe(comparison, use_container_width=True, hide_index=True)
        else:
            st.warning("Model comparison data not found. Run model_evaluation.py first.")

        # Display evaluation plots
        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Actual vs Predicted (Test Set)")
            img_path = PLOTS_DIR / "actual_vs_predicted_test.png"
            if img_path.exists():
                st.image(str(img_path), use_container_width=True)
            else:
                st.info("Run model_evaluation.py to generate this plot.")

        with col2:
            st.subheader("Model Comparison Chart")
            img_path = PLOTS_DIR / "model_comparison.png"
            if img_path.exists():
                st.image(str(img_path), use_container_width=True)
            else:
                st.info("Run model_evaluation.py to generate this plot.")

        st.markdown("---")
        col3, col4 = st.columns(2)

        with col3:
            st.subheader("Residual Analysis")
            img_path = PLOTS_DIR / "residuals_test.png"
            if img_path.exists():
                st.image(str(img_path), use_container_width=True)

        with col4:
            st.subheader("Learning Curves")
            img_path = PLOTS_DIR / "learning_curves.png"
            if img_path.exists():
                st.image(str(img_path), use_container_width=True)

        # Error by price range
        img_path = PLOTS_DIR / "prediction_error_by_range.png"
        if img_path.exists():
            st.subheader("Prediction Error by Price Range")
            st.image(str(img_path), use_container_width=True)

        # Evaluation report
        report_path = RESULTS_DIR / "evaluation_report.txt"
        if report_path.exists():
            st.subheader("Detailed Evaluation Report")
            with open(report_path, "r") as f:
                st.code(f.read())

    # ═══ Tab 3: Explainability ═══════════════════════════════════════════
    with tab3:
        st.subheader("Model Explainability (XAI)")
        st.markdown(
            "Understanding **what** the model learned and **why** it makes "
            "specific predictions using four XAI techniques."
        )

        # SHAP section
        st.markdown("### 1. SHAP Analysis")
        st.markdown(
            "**SHAP (SHapley Additive exPlanations)** assigns each feature "
            "a contribution value for every prediction, based on game theory."
        )

        shap_cols = st.columns(2)
        with shap_cols[0]:
            img = PLOTS_DIR / "shap_summary_beeswarm.png"
            if img.exists():
                st.image(str(img), caption="SHAP Beeswarm (Global Feature Impact)", use_container_width=True)
        with shap_cols[1]:
            img = PLOTS_DIR / "shap_bar_importance.png"
            if img.exists():
                st.image(str(img), caption="SHAP Bar (Mean |SHAP Value|)", use_container_width=True)

        # SHAP waterfall examples
        st.markdown("#### SHAP Waterfall Examples (Local Explanations)")
        wf_cols = st.columns(3)
        for i, label in enumerate(["low", "mid", "high"]):
            with wf_cols[i]:
                img = PLOTS_DIR / f"shap_waterfall_{label}_price.png"
                if img.exists():
                    st.image(str(img), caption=f"{label.capitalize()} Price", use_container_width=True)

        st.markdown("---")

        # Feature Importance
        st.markdown("### 2. XGBoost Feature Importance")
        st.markdown(
            "Built-in importance metrics: **Weight** (# splits), "
            "**Gain** (avg improvement), **Cover** (avg samples affected)."
        )
        img = PLOTS_DIR / "xgboost_feature_importance.png"
        if img.exists():
            st.image(str(img), use_container_width=True)

        st.markdown("---")

        # PDP
        st.markdown("### 3. Partial Dependence Plots (PDP)")
        st.markdown(
            "PDP shows the marginal effect of each feature on predictions, "
            "with ICE (Individual Conditional Expectation) lines showing "
            "variation across individual samples."
        )
        pdp_cols = st.columns(2)
        with pdp_cols[0]:
            img = PLOTS_DIR / "pdp_individual.png"
            if img.exists():
                st.image(str(img), caption="Individual PDP + ICE", use_container_width=True)
        with pdp_cols[1]:
            img = PLOTS_DIR / "pdp_interactions.png"
            if img.exists():
                st.image(str(img), caption="2D PDP Interactions", use_container_width=True)

        st.markdown("---")

        # LIME
        st.markdown("### 4. LIME Explanations")
        st.markdown(
            "**LIME (Local Interpretable Model-Agnostic Explanations)** "
            "creates a simple linear model around each prediction to explain it."
        )
        lime_cols = st.columns(3)
        for i, label in enumerate(["low", "mid", "high"]):
            with lime_cols[i]:
                img = PLOTS_DIR / f"lime_explanation_{label}.png"
                if img.exists():
                    st.image(str(img), caption=f"{label.capitalize()} Price", use_container_width=True)

    # ─── Footer ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        "*Sri Lanka Property Price Predictor | "
        "Data source: ikman.lk | "
        "Model: XGBoost Regressor | "
        "XAI: SHAP, LIME, PDP, Feature Importance*"
    )


if __name__ == "__main__":
    main()
