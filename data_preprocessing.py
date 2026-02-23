"""
Data Preprocessing Pipeline
Cleans raw scraped data, engineers features, encodes categoricals,
and splits into train/validation/test sets.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import re
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ─── Paths ───────────────────────────────────────────────────────────────────

DATA_DIR = Path("data")
MODELS_DIR = Path("models")
RAW_CSV = DATA_DIR / "raw_data.csv"
PROCESSED_CSV = DATA_DIR / "processed_data.csv"

# ─── Sri Lanka Districts ─────────────────────────────────────────────────────

SRI_LANKA_DISTRICTS = [
    "Colombo", "Gampaha", "Kalutara",          # Western Province
    "Kandy", "Matale", "Nuwara Eliya",          # Central Province
    "Galle", "Matara", "Hambantota",            # Southern Province
    "Jaffna", "Kilinochchi", "Mannar",          # Northern Province
    "Vavuniya", "Mullaitivu",
    "Batticaloa", "Ampara", "Trincomalee",      # Eastern Province
    "Kurunegala", "Puttalam",                    # North Western Province
    "Anuradhapura", "Polonnaruwa",              # North Central Province
    "Badulla", "Moneragala",                     # Uva Province
    "Ratnapura", "Kegalle",                      # Sabaragamuwa Province
]

# Common spelling variations / aliases
DISTRICT_ALIASES = {
    "nuwara-eliya": "Nuwara Eliya",
    "nuwara eliya": "Nuwara Eliya",
    "nuwaraeliya": "Nuwara Eliya",
    "mt lavinia": "Colombo",
    "mount lavinia": "Colombo",
    "dehiwala": "Colombo",
    "negombo": "Gampaha",
    "panadura": "Kalutara",
    "beruwala": "Kalutara",
    "hikkaduwa": "Galle",
    "unawatuna": "Galle",
    "tangalle": "Hambantota",
    "ella": "Badulla",
    "haputale": "Badulla",
    "dambulla": "Matale",
    "sigiriya": "Matale",
    "chilaw": "Puttalam",
    "trinco": "Trincomalee",
}


# ─── Cleaning Functions ──────────────────────────────────────────────────────


def load_raw_data() -> pd.DataFrame:
    """Load raw data and print summary."""
    df = pd.read_csv(RAW_CSV)
    print(f"Raw data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}")
    print(f"\nNull counts:\n{df.isnull().sum()}")
    return df


def clean_price(df: pd.DataFrame) -> pd.DataFrame:
    """Filter prices to a reasonable range."""
    df = df.copy()
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    initial = len(df)
    # Remove nulls and zeros
    df = df.dropna(subset=["price"])
    df = df[df["price"] > 0]

    # Filter reasonable range: Rs 500K to Rs 500M
    df = df[(df["price"] >= 500_000) & (df["price"] <= 500_000_000)]

    print(f"Price cleaning: {initial} -> {len(df)} rows "
          f"(removed {initial - len(df)})")
    print(f"  Price range: Rs {df['price'].min():,.0f} - Rs {df['price'].max():,.0f}")
    print(f"  Median: Rs {df['price'].median():,.0f}")
    return df


def parse_bedrooms_bathrooms(value) -> int | None:
    """Parse bedroom/bathroom string to integer."""
    if pd.isna(value) or value == "":
        return None
    value = str(value).strip()
    # Handle "10+" format
    value = value.replace("+", "")
    # Extract first number
    match = re.search(r"(\d+)", value)
    if match:
        return int(match.group(1))
    return None


def clean_bedrooms_bathrooms(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and parse bedroom/bathroom columns."""
    df = df.copy()

    df["bedrooms"] = df["bedrooms"].apply(parse_bedrooms_bathrooms)
    df["bathrooms"] = df["bathrooms"].apply(parse_bedrooms_bathrooms)

    # Cap at reasonable values
    df.loc[df["bedrooms"] > 10, "bedrooms"] = 10
    df.loc[df["bathrooms"] > 10, "bathrooms"] = 10

    # Fill missing with median (by property type)
    for ptype in df["property_type"].unique():
        mask = df["property_type"] == ptype
        med_bed = df.loc[mask, "bedrooms"].median()
        med_bath = df.loc[mask, "bathrooms"].median()
        if pd.notna(med_bed):
            df.loc[mask & df["bedrooms"].isna(), "bedrooms"] = med_bed
        if pd.notna(med_bath):
            df.loc[mask & df["bathrooms"].isna(), "bathrooms"] = med_bath

    # Global fallback
    df["bedrooms"] = df["bedrooms"].fillna(df["bedrooms"].median())
    df["bathrooms"] = df["bathrooms"].fillna(df["bathrooms"].median())

    df["bedrooms"] = df["bedrooms"].astype(int)
    df["bathrooms"] = df["bathrooms"].astype(int)

    print(f"Bedrooms range: {df['bedrooms'].min()} - {df['bedrooms'].max()}")
    print(f"Bathrooms range: {df['bathrooms'].min()} - {df['bathrooms'].max()}")
    return df


def parse_size(value: str) -> float | None:
    """Parse size strings like '2,800.0 sqft' or '10.5 perches' to float."""
    if pd.isna(value) or value == "":
        return None
    value = str(value).strip()
    # Remove commas and extract number
    cleaned = value.replace(",", "")
    match = re.search(r"([\d.]+)", cleaned)
    if match:
        num = float(match.group(1))
        # Convert acres to perches if needed (for land_size)
        if "acre" in value.lower():
            num = num * 160  # 1 acre = 160 perches
        return num if num > 0 else None
    return None


def clean_sizes(df: pd.DataFrame) -> pd.DataFrame:
    """Parse and clean house_size and land_size columns."""
    df = df.copy()

    df["house_size"] = df["house_size"].apply(parse_size)
    df["land_size"] = df["land_size"].apply(parse_size)

    # Remove extreme outliers
    df.loc[df["house_size"] > 20_000, "house_size"] = np.nan  # > 20K sqft
    df.loc[df["land_size"] > 500, "land_size"] = np.nan       # > 500 perches

    # Don't impute - XGBoost handles NaN natively
    house_valid = df["house_size"].notna().sum()
    land_valid = df["land_size"].notna().sum()
    print(f"House size: {house_valid}/{len(df)} valid ({house_valid/len(df)*100:.1f}%)")
    print(f"Land size: {land_valid}/{len(df)} valid ({land_valid/len(df)*100:.1f}%)")
    return df


def clean_location(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and standardize district names."""
    df = df.copy()

    # Standardize district names
    valid_districts = {d.lower(): d for d in SRI_LANKA_DISTRICTS}

    def map_district(district):
        if pd.isna(district) or district == "":
            return None
        d = str(district).strip()
        # Direct match
        if d.lower() in valid_districts:
            return valid_districts[d.lower()]
        # Alias match
        if d.lower() in DISTRICT_ALIASES:
            return DISTRICT_ALIASES[d.lower()]
        # Partial match
        for valid_name in SRI_LANKA_DISTRICTS:
            if valid_name.lower() in d.lower() or d.lower() in valid_name.lower():
                return valid_name
        return d  # keep original if no match found

    df["district"] = df["district"].apply(map_district)

    # Also try location_listing as fallback for empty districts
    mask = df["district"].isna() | (df["district"] == "")
    if "location_listing" in df.columns:
        df.loc[mask, "district"] = df.loc[mask, "location_listing"].apply(map_district)

    # Drop rows where district is still empty
    initial = len(df)
    df = df.dropna(subset=["district"])
    df = df[df["district"] != ""]
    print(f"Location cleaning: {initial} -> {len(df)} rows")
    print(f"Districts found: {df['district'].nunique()}")
    print(f"Top districts:\n{df['district'].value_counts().head(10)}")
    return df


# ─── Feature Engineering ─────────────────────────────────────────────────────


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived features for the model."""
    df = df.copy()

    # Log-transform the target (price is right-skewed)
    df["log_price"] = np.log1p(df["price"])

    # Log-transform size features
    df["log_house_size"] = np.log1p(df["house_size"])
    df["log_land_size"] = np.log1p(df["land_size"])

    # Bedroom-bathroom ratio
    df["bed_bath_ratio"] = df["bedrooms"] / df["bathrooms"].replace(0, 1)
    df["bed_bath_ratio"] = df["bed_bath_ratio"].clip(upper=5)

    # Total rooms
    df["total_rooms"] = df["bedrooms"] + df["bathrooms"]

    # Binary flags for missing sizes
    df["has_house_size"] = df["house_size"].notna().astype(int)
    df["has_land_size"] = df["land_size"].notna().astype(int)

    # Property type flag
    df["is_apartment"] = (
        df["property_type"].str.lower().str.contains("apartment", na=False)
    ).astype(int)

    print(f"\nEngineered features added:")
    print(f"  log_price, log_house_size, log_land_size")
    print(f"  bed_bath_ratio, total_rooms")
    print(f"  has_house_size, has_land_size, is_apartment")
    return df


# ─── Encoding & Splitting ───────────────────────────────────────────────────


def encode_and_split(df: pd.DataFrame):
    """Label-encode categoricals, select features, split data."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Encode district
    le_district = LabelEncoder()
    df["district_encoded"] = le_district.fit_transform(df["district"])

    # Encode property type
    le_ptype = LabelEncoder()
    df["property_type_encoded"] = le_ptype.fit_transform(df["property_type"])

    # Save encoders
    encoders = {
        "district_encoder": le_district,
        "property_type_encoder": le_ptype,
        "district_classes": list(le_district.classes_),
        "property_type_classes": list(le_ptype.classes_),
    }
    joblib.dump(encoders, MODELS_DIR / "encoders.joblib")
    print(f"Encoders saved: {len(le_district.classes_)} districts, "
          f"{len(le_ptype.classes_)} property types")

    # Define feature columns
    feature_cols = [
        "district_encoded",
        "property_type_encoded",
        "bedrooms",
        "bathrooms",
        "house_size",
        "land_size",
        "bed_bath_ratio",
        "total_rooms",
        "has_house_size",
        "has_land_size",
        "is_apartment",
        "log_house_size",
        "log_land_size",
    ]
    target_col = "log_price"

    # Save feature names
    joblib.dump(feature_cols, MODELS_DIR / "feature_names.joblib")

    X = df[feature_cols]
    y = df[target_col]

    # Split: 70% train / 15% validation / 15% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42
    )

    print(f"\nData split:")
    print(f"  Train: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
    print(f"  Val:   {X_val.shape[0]} samples ({X_val.shape[0]/len(X)*100:.1f}%)")
    print(f"  Test:  {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")

    # Save splits
    X_train.to_csv(DATA_DIR / "X_train.csv", index=False)
    X_val.to_csv(DATA_DIR / "X_val.csv", index=False)
    X_test.to_csv(DATA_DIR / "X_test.csv", index=False)
    y_train.to_csv(DATA_DIR / "y_train.csv", index=False)
    y_val.to_csv(DATA_DIR / "y_val.csv", index=False)
    y_test.to_csv(DATA_DIR / "y_test.csv", index=False)

    # Save processed dataset (full)
    df.to_csv(PROCESSED_CSV, index=False)
    print(f"\nProcessed data saved: {PROCESSED_CSV}")
    print(f"Train/val/test splits saved to {DATA_DIR}/")

    return X_train, X_val, X_test, y_train, y_val, y_test


# ─── Main Pipeline ───────────────────────────────────────────────────────────


def main():
    """Run the full preprocessing pipeline."""
    print("=" * 60)
    print("DATA PREPROCESSING PIPELINE")
    print("=" * 60)

    # Load
    df = load_raw_data()

    # Clean
    print("\n--- Cleaning Price ---")
    df = clean_price(df)

    print("\n--- Cleaning Bedrooms & Bathrooms ---")
    df = clean_bedrooms_bathrooms(df)

    print("\n--- Cleaning Sizes ---")
    df = clean_sizes(df)

    print("\n--- Cleaning Location ---")
    df = clean_location(df)

    # Feature engineering
    print("\n--- Feature Engineering ---")
    df = engineer_features(df)

    # Encode and split
    print("\n--- Encoding & Splitting ---")
    X_train, X_val, X_test, y_train, y_val, y_test = encode_and_split(df)

    # Final summary
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"Final dataset: {len(df)} rows, {len(df.columns)} columns")
    print(f"\nPrice statistics (LKR):")
    print(f"  Mean:   Rs {df['price'].mean():>15,.0f}")
    print(f"  Median: Rs {df['price'].median():>15,.0f}")
    print(f"  Std:    Rs {df['price'].std():>15,.0f}")
    print(f"  Min:    Rs {df['price'].min():>15,.0f}")
    print(f"  Max:    Rs {df['price'].max():>15,.0f}")
    print(f"\nFeature columns: {joblib.load(MODELS_DIR / 'feature_names.joblib')}")


if __name__ == "__main__":
    main()
