# Sri Lanka Property Price Predictor

A machine learning system that predicts residential property prices in Sri Lanka using data scraped from **ikman.lk**. Built with **XGBoost** regression, comprehensive **explainability (XAI)** analysis, and an interactive **Streamlit** web application.

---

## Problem Statement

Predicting property prices in Sri Lanka is challenging due to diverse regional markets, varying property characteristics, and limited structured data. This project addresses this by:

1. **Collecting** real-world property listings from ikman.lk (Sri Lanka's largest classifieds platform)
2. **Training** an XGBoost regression model to predict prices based on property features
3. **Explaining** model predictions using 4 XAI techniques (SHAP, LIME, PDP, Feature Importance)
4. **Deploying** an interactive Streamlit web app for users to get predictions with explanations

---

## Dataset

- **Source**: [ikman.lk](https://ikman.lk) — Houses and Apartments for sale
- **Collection Method**: Web scraping using `requests` + `BeautifulSoup`
- **Size**: ~20,000+ property listings
- **Ethical Use**: Only publicly available listing data; no personal/sensitive information

### Features

| Feature | Description | Type |
|---------|-------------|------|
| `district` | District location (25 Sri Lankan districts) | Categorical |
| `property_type` | House or Apartment | Categorical |
| `bedrooms` | Number of bedrooms (1-10) | Numeric |
| `bathrooms` | Number of bathrooms (1-10) | Numeric |
| `house_size` | House size in sqft | Numeric |
| `land_size` | Land size in perches | Numeric |

### Engineered Features

| Feature | Description |
|---------|-------------|
| `log_price` | Log-transformed target (handles skewness) |
| `log_house_size` | Log-transformed house size |
| `log_land_size` | Log-transformed land size |
| `bed_bath_ratio` | Bedrooms / Bathrooms |
| `total_rooms` | Bedrooms + Bathrooms |
| `has_house_size` | Binary: 1 if house size known |
| `has_land_size` | Binary: 1 if land size known |
| `is_apartment` | Binary: 1 if apartment |

### Preprocessing

- Price filtering: Rs 500K – Rs 500M (removes outliers)
- Missing sizes: Not imputed — XGBoost handles NaN natively
- Location: Standardized to 25 official districts
- Encoding: LabelEncoder for district and property type
- Target: log(1 + price) transformation

---

## Machine Learning Algorithm

### XGBoost Regressor (Primary Model)

**Why XGBoost?**
- Handles missing values natively (critical for house/land size)
- Built-in L1/L2 regularization prevents overfitting
- Feature importance built-in
- Compatible with SHAP TreeExplainer (exact, fast)
- State-of-the-art on tabular data
- **Not a deep learning model** — satisfies assignment constraint

### Baseline Models (for comparison)
- Linear Regression
- Ridge Regression
- Random Forest Regressor

### Hyperparameter Tuning
- **Method**: RandomizedSearchCV (50 iterations, 5-fold CV)
- **Parameters tuned**: n_estimators, max_depth, learning_rate, subsample, colsample_bytree, min_child_weight, reg_alpha, reg_lambda, gamma

---

## Model Evaluation

### Metrics
- **R² Score**: Coefficient of determination
- **RMSE**: Root Mean Squared Error (both log-scale and LKR)
- **MAE**: Mean Absolute Error
- **MAPE**: Mean Absolute Percentage Error

### Data Split
- Training: 70%
- Validation: 15%
- Test: 15%

### Visualizations
- Actual vs Predicted scatter plots
- Residual analysis (scatter + histogram)
- Learning curves (overfitting detection)
- Prediction error by price range
- Model comparison bar charts

---

## Explainability & Interpretation (XAI)

### 1. SHAP (SHapley Additive exPlanations)
- **TreeExplainer** for exact SHAP values
- Beeswarm summary plot (global feature impact)
- Bar importance plot (mean |SHAP value|)
- Waterfall plots (local explanation for individual predictions)
- Force plots (visual feature contributions)
- Dependence plots (feature effect curves)

### 2. XGBoost Built-in Feature Importance
- **Weight**: Number of times a feature is used in splits
- **Gain**: Average improvement when feature is used
- **Cover**: Average number of samples affected

### 3. Partial Dependence Plots (PDP)
- Individual PDP + ICE (Individual Conditional Expectation) curves
- 2D interaction plots for key feature pairs

### 4. LIME (Local Interpretable Model-Agnostic Explanations)
- Local linear approximations around individual predictions
- Feature contribution analysis for specific properties

---

## Front-End Application (Streamlit)

Interactive web dashboard with three tabs:

1. **Price Prediction**: Enter property details, get predicted price + SHAP waterfall
2. **Model Performance**: Metrics table, comparison charts, residual analysis
3. **Explainability**: All XAI visualizations (SHAP, PDP, LIME, Feature Importance)

---

## Project Structure

```
property-price-predictor/
├── scraper.py              # Web scraper for ikman.lk
├── data_preprocessing.py   # Data cleaning & feature engineering
├── model_training.py       # XGBoost + baseline training + tuning
├── model_evaluation.py     # Metrics & evaluation plots
├── explainability.py       # SHAP, LIME, PDP, feature importance
├── app.py                  # Streamlit web application
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── data/
│   ├── raw_data.csv           # Scraped property data
│   ├── processed_data.csv     # Cleaned & engineered data
│   ├── X_train.csv            # Training features
│   ├── X_val.csv              # Validation features
│   ├── X_test.csv             # Test features
│   ├── y_train.csv            # Training target
│   ├── y_val.csv              # Validation target
│   └── y_test.csv             # Test target
├── models/
│   ├── xgboost_model.joblib   # Trained XGBoost model
│   ├── encoders.joblib        # Label encoders
│   ├── feature_names.joblib   # Feature column names
│   ├── shap_values.joblib     # Pre-computed SHAP values
│   └── best_params.json       # Best hyperparameters
├── plots/                     # All generated visualizations
└── results/
    ├── model_comparison.csv   # Model comparison table
    └── evaluation_report.txt  # Detailed evaluation report
```

---

## Setup & Usage

### Prerequisites
- Python 3.9+

### Installation

```bash
pip install -r requirements.txt
```

### Running the Pipeline

Execute scripts in this order:

```bash
# 1. Scrape data from ikman.lk (~18 hours for full dataset)
python scraper.py

# 2. Clean and preprocess data (~2 minutes)
python data_preprocessing.py

# 3. Train models with hyperparameter tuning (~10-30 minutes)
python model_training.py

# 4. Generate evaluation metrics and plots (~2-5 minutes)
python model_evaluation.py

# 5. Generate XAI plots (~5-10 minutes)
python explainability.py

# 6. Launch the web application
streamlit run app.py
```

---

## Limitations

- **Advertised prices**: Data represents listing prices, not actual transaction prices
- **Data completeness**: Many listings lack house_size or land_size
- **Temporal bias**: Prices reflect the scraping date, not historical trends
- **Geographic coverage**: Urban areas (Colombo, Gampaha) are overrepresented
- **Feature limitations**: Excludes important factors like age of building, floor level, proximity to amenities

## Ethical Considerations

- Only publicly available data was collected from ikman.lk
- No personal or sensitive information is stored
- Model predictions should be used as estimates, not definitive valuations
- Potential bias toward urban, higher-income property markets
- Rate-limited scraping to minimize impact on ikman.lk servers

---

## Technologies

| Category | Tools |
|----------|-------|
| Web Scraping | requests, BeautifulSoup4 |
| Data Processing | pandas, NumPy |
| Machine Learning | XGBoost, scikit-learn |
| Explainability | SHAP, LIME |
| Visualization | matplotlib, seaborn |
| Web App | Streamlit |
