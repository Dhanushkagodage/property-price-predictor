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

# Hide Deploy button and hamburger menu
st.markdown(
    """
    <style>
        .stAppDeployButton {display: none;}
        [data-testid="stToolbar"] {display: none;}
    </style>
    """,
    unsafe_allow_html=True,
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
            st.subheader("Why This Price?")
            try:
                X_train = load_training_data()
                explainer = load_shap_explainer(model, X_train)
                shap_value = explainer(features_df)

                # Extract SHAP values
                shap_vals = shap_value[0].values
                base_log = shap_value[0].base_values
                feat_names = list(features_df.columns)
                base_price = np.expm1(base_log)

                # ─── Group related features ───────────────────────
                group_map = {
                    "house_size": "House Size",
                    "log_house_size": "House Size",
                    "has_house_size": "House Size",
                    "land_size": "Land Size",
                    "log_land_size": "Land Size",
                    "has_land_size": "Land Size",
                    "bedrooms": "Bedrooms",
                    "total_rooms": "Bedrooms",
                    "bathrooms": "Bathrooms",
                    "bed_bath_ratio": "Bathrooms",
                    "district_encoded": "District",
                    "property_type_encoded": "Property Type",
                    "is_apartment": "Property Type",
                }

                # User-friendly labels with actual values
                group_labels = {
                    "House Size": f"House Size ({house_size} sqft)" if house_size > 0 else "House Size (N/A)",
                    "Land Size": f"Land Size ({land_size} perch)" if land_size > 0 else "Land Size (N/A)",
                    "Bedrooms": f"Bedrooms ({bedrooms})",
                    "Bathrooms": f"Bathrooms ({bathrooms})",
                    "District": f"District ({district})",
                    "Property Type": f"Property Type ({property_type})",
                }

                # Sum SHAP values in log-space per group
                grouped_log = {}
                for i, name in enumerate(feat_names):
                    grp = group_map.get(name, name)
                    grouped_log[grp] = grouped_log.get(grp, 0) + shap_vals[i]

                # Convert grouped log-space values to LKR impact
                grouped = {}
                for grp, log_impact in grouped_log.items():
                    lkr_impact = (np.expm1(base_log + log_impact) - base_price) / 1e6
                    grouped[grp] = lkr_impact

                # Sort by absolute impact (largest first)
                sorted_groups = sorted(grouped.items(), key=lambda x: abs(x[1]), reverse=True)

                # ─── Info box: What is SHAP? ──────────────────────
                st.info(
                    "**How to read this chart:** The model starts from the **average price** "
                    f"of all properties in the dataset (**Rs {base_price/1e6:.1f}M**). "
                    "Then each feature **pushes the price up (green)** or **pulls it down (red)** "
                    "based on your input values compared to the dataset average. "
                    "The final bar shows **your predicted price**."
                )

                # ─── Waterfall Bar Chart ──────────────────────────
                labels = ["Average\nPrice"]
                values = [base_price / 1e6]
                colors = ["#90A4AE"]
                annotations = [f"Rs {base_price/1e6:.1f}M"]

                running = base_price / 1e6
                for grp, impact in sorted_groups:
                    lbl = group_labels.get(grp, grp)
                    labels.append(lbl.replace(" (", "\n("))
                    running += impact
                    values.append(impact)
                    if impact >= 0:
                        colors.append("#4CAF50")
                        annotations.append(f"+{impact:.1f}M")
                    else:
                        colors.append("#F44336")
                        annotations.append(f"{impact:.1f}M")

                labels.append("Your\nPrice")
                values.append(predicted_price / 1e6)
                colors.append("#1A237E")
                annotations.append(f"Rs {predicted_price/1e6:.1f}M")

                # Build waterfall coordinates
                n = len(labels)
                bottoms = [0.0] * n
                heights = [0.0] * n

                # First bar: Average Price
                bottoms[0] = 0
                heights[0] = values[0]
                cumulative = values[0]

                # Middle bars: feature impacts
                for i in range(1, n - 1):
                    if values[i] >= 0:
                        bottoms[i] = cumulative
                        heights[i] = values[i]
                    else:
                        bottoms[i] = cumulative + values[i]
                        heights[i] = abs(values[i])
                    cumulative += values[i]

                # Last bar: Your Price
                bottoms[-1] = 0
                heights[-1] = values[-1]

                fig, ax = plt.subplots(figsize=(max(12, n * 1.5), 6))
                x = np.arange(n)
                bars = ax.bar(x, heights, bottom=bottoms, color=colors,
                              width=0.6, edgecolor="white", linewidth=1.5)

                # Add connecting lines between bars
                for i in range(n - 2):
                    top_i = bottoms[i] + heights[i] if values[i] >= 0 else bottoms[i]
                    if i == 0:
                        top_i = heights[0]
                    conn_y = bottoms[i] + heights[i]
                    ax.plot([x[i] + 0.3, x[i + 1] - 0.3], [conn_y, conn_y],
                            color="#BDBDBD", linewidth=1.2, linestyle="--")

                # Add value annotations above/below bars
                for i, bar in enumerate(bars):
                    y_pos = bottoms[i] + heights[i] + (max(heights) * 0.02)
                    fontw = "bold" if i == 0 or i == n - 1 else "normal"
                    fontsz = 11 if i == 0 or i == n - 1 else 10
                    color = colors[i] if i > 0 and i < n - 1 else "#1A237E"
                    ax.text(x[i], y_pos, annotations[i],
                            ha="center", va="bottom", fontsize=fontsz,
                            fontweight=fontw, color=color)

                ax.set_xticks(x)
                ax.set_xticklabels(labels, fontsize=9, ha="center")
                ax.set_ylabel("Price (Million LKR)", fontsize=12)
                ax.set_title("How Each Feature Affects the Predicted Price",
                             fontsize=14, fontweight="bold", pad=15)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.set_ylim(bottom=0)

                # Add legend
                from matplotlib.patches import Patch
                legend_elements = [
                    Patch(facecolor="#90A4AE", label="Average / Final price"),
                    Patch(facecolor="#4CAF50", label="Increases price"),
                    Patch(facecolor="#F44336", label="Decreases price"),
                ]
                ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

                # ─── Detailed Feature Breakdown Table ─────────────
                st.markdown("---")
                st.markdown("#### Feature Contribution Breakdown")

                total_positive = sum(v for _, v in sorted_groups if v > 0)
                total_negative = sum(v for _, v in sorted_groups if v < 0)
                total_impact = total_positive + total_negative

                inc_col, dec_col = st.columns(2)

                with inc_col:
                    st.markdown("**:green[Features that INCREASED the price:]**")
                    for grp, impact in sorted_groups:
                        if impact > 0:
                            lbl = group_labels.get(grp, grp)
                            pct = (impact / abs(total_impact)) * 100 if total_impact != 0 else 0
                            st.markdown(
                                f"- **{lbl}** → **+Rs {impact:.1f}M** "
                                f"({pct:.0f}% of total change)"
                            )

                with dec_col:
                    st.markdown("**:red[Features that DECREASED the price:]**")
                    for grp, impact in sorted_groups:
                        if impact < 0:
                            lbl = group_labels.get(grp, grp)
                            pct = (abs(impact) / abs(total_impact)) * 100 if total_impact != 0 else 0
                            st.markdown(
                                f"- **{lbl}** → **Rs {impact:.1f}M** "
                                f"({pct:.0f}% of total change)"
                            )

                # ─── Step-by-step calculation ─────────────────────
                st.markdown("---")
                st.markdown("#### How the Prediction Was Calculated")

                calc_lines = [
                    f"1. **Start** with the average property price: **Rs {base_price/1e6:.1f} Million**",
                    f"   _(This is the average predicted price across all ~1,100 properties in our dataset)_",
                    "",
                ]
                step = 2
                running_price = base_price / 1e6
                for grp, impact in sorted_groups:
                    lbl = group_labels.get(grp, grp)
                    running_price += impact
                    if impact >= 0:
                        calc_lines.append(
                            f"{step}. **{lbl}** pushes price **up** by Rs {impact:.1f}M "
                            f"→ Running total: **Rs {running_price:.1f}M**"
                        )
                    else:
                        calc_lines.append(
                            f"{step}. **{lbl}** pulls price **down** by Rs {abs(impact):.1f}M "
                            f"→ Running total: **Rs {running_price:.1f}M**"
                        )
                    step += 1

                calc_lines.append("")
                calc_lines.append(
                    f"**Final Predicted Price = Rs {predicted_price/1e6:.1f} Million**"
                )

                st.markdown("\n".join(calc_lines))

                # ─── Why negative/positive explanation ────────────
                st.markdown("---")
                st.markdown("#### Understanding Positive & Negative Values")
                st.markdown(
                    "SHAP values are **relative to the dataset average**. "
                    "A feature shows a **negative** impact when your input is "
                    "**below the average** for that feature in the dataset, and "
                    "**positive** when it is **above average**.\n\n"
                    "**Example:** If the average house size in the dataset is ~2,200 sqft "
                    "and you entered 1,300 sqft, that is below average — so House Size "
                    "pulls the price **down**. If you enter 5,000 sqft (above average), "
                    "it would push the price **up**.\n\n"
                    "_The same logic applies to all features: District, Bedrooms, "
                    "Land Size, etc._"
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
