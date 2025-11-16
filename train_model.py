import json
import pickle
from typing import List

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


COLUMNS_JSON_PATH = "columns.json"
MODEL_PKL_PATH = "banglore_home_prices_model.pickle"
DATA_CSV_PATH = "Bengaluru_House_Data.csv"


def parse_total_sqft(value: str) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    s = str(value).strip()
    try:
        # Single numeric
        return float(s)
    except Exception:
        pass
    # Handle range like '1133-1384'
    if "-" in s:
        parts = s.split("-")
        try:
            nums = [float(p) for p in parts if p.strip()]
            if len(nums) == 2:
                return (nums[0] + nums[1]) / 2.0
        except Exception:
            return np.nan
    # Handle values with units like '34.46Sq. Meter', '1200 per sq. ft'
    tokens = s.split()
    try:
        return float(tokens[0])
    except Exception:
        return np.nan


def extract_bhk(size_value: str) -> float:
    if pd.isna(size_value):
        return np.nan
    s = str(size_value)
    # Common formats: '2 BHK', '4 Bedroom'
    tokens = s.split()
    for t in tokens:
        try:
            return float(t)
        except Exception:
            continue
    return np.nan


def clean_availability(avail: str) -> str:
    """Standardize availability values"""
    if pd.isna(avail):
        return "Ready To Move"
    s = str(avail).strip()
    if "Ready" in s or "ready" in s:
        return "Ready To Move"
    else:
        return "Soon to be Vacated"


def load_category_lists_from_columns_json():
    with open(COLUMNS_JSON_PATH, "r") as f:
        data = json.load(f)
    availability_values: List[str] = data["availability_columns"]
    area_values: List[str] = data["area_columns"]
    location_values: List[str] = data["location_columns"]
    return availability_values, area_values, location_values


def main():
    # Load data
    df = pd.read_csv(DATA_CSV_PATH)
    print(f"Original data shape: {df.shape}")

    # Clean and preprocess data
    df = df.copy()
    
    # Clean total_sqft
    df["total_sqft"] = df["total_sqft"].apply(parse_total_sqft)
    
    # Extract BHK from size column
    if "size" in df.columns and "bhk" not in df.columns:
        df["bhk"] = df["size"].apply(extract_bhk)
    
    # Clean area_type column
    if "area_type" in df.columns:
        df.rename(columns={"area_type": "area"}, inplace=True)
    
    # Clean availability
    if "availability" in df.columns:
        df["availability"] = df["availability"].apply(clean_availability)
    elif "Availability" in df.columns:
        df.rename(columns={"Availability": "availability"}, inplace=True)
        df["availability"] = df["availability"].apply(clean_availability)
    
    # Keep required columns
    required_cols = ["location", "area", "availability", "total_sqft", "bath", "bhk", "price"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"CSV is missing required columns: {missing}")

    df = df[required_cols]
    
    # Remove rows with missing values
    df.dropna(subset=required_cols, inplace=True)
    
    # Enforce positive values and reasonable ranges
    df = df[
        (df["total_sqft"] > 100) & (df["total_sqft"] < 10000) &
        (df["bath"] > 0) & (df["bath"] <= 10) &
        (df["bhk"] > 0) & (df["bhk"] <= 10) &
        (df["price"] > 10) & (df["price"] < 1000)
    ]
    
    print(f"After cleaning: {df.shape}")
    
    # Load canonical category orders from existing columns.json
    availability_values, area_values, location_values = load_category_lists_from_columns_json()
    
    # Check what categories actually exist in the data
    print(f"Unique locations in data: {df['location'].nunique()}")
    print(f"Unique areas in data: {df['area'].nunique()}")
    print(f"Unique availability in data: {df['availability'].nunique()}")
    
    print(f"Sample locations: {df['location'].unique()[:10]}")
    print(f"Sample areas: {df['area'].unique()}")
    print(f"Sample availability: {df['availability'].unique()}")
    
    # Filter to only include locations that exist in our categories
    # But be more lenient - if a category doesn't exist, we'll handle it
    df_filtered = df.copy()
    
    # Only filter if we have data for that category
    if df['location'].isin(location_values).any():
        df_filtered = df_filtered[df_filtered["location"].isin(location_values)]
    if df['area'].isin(area_values).any():
        df_filtered = df_filtered[df_filtered["area"].isin(area_values)]
    if df['availability'].isin(availability_values).any():
        df_filtered = df_filtered[df_filtered["availability"].isin(availability_values)]
    
    print(f"After category filtering: {df_filtered.shape}")
    
    if df_filtered.shape[0] == 0:
        print("No data left after filtering. Using original data with category mapping.")
        df_filtered = df.copy()
    
    # Convert to categorical with our predefined order, but handle missing categories
    df_filtered["availability"] = df_filtered["availability"].astype(CategoricalDtype(categories=availability_values))
    df_filtered["area"] = df_filtered["area"].astype(CategoricalDtype(categories=area_values))
    df_filtered["location"] = df_filtered["location"].astype(CategoricalDtype(categories=location_values))
    
    # Build design matrix matching API order: [total_sqft, bath, bhk], then availability, area, location dummies
    X_numeric = df_filtered[["total_sqft", "bath", "bhk"]].to_numpy()
    
    # Get dummies in our category order, dropping first to avoid multicollinearity
    avail_dummies = pd.get_dummies(df_filtered["availability"], prefix=None, drop_first=True)
    area_dummies = pd.get_dummies(df_filtered["area"], prefix=None, drop_first=True)
    loc_dummies = pd.get_dummies(df_filtered["location"], prefix=None, drop_first=True)
    
    # Reindex columns to ensure consistent order
    avail_cols = availability_values[1:]  # dropped first
    area_cols = area_values[1:]
    loc_cols = location_values[1:]
    
    avail_dummies = avail_dummies.reindex(columns=avail_cols, fill_value=0)
    area_dummies = area_dummies.reindex(columns=area_cols, fill_value=0)
    loc_dummies = loc_dummies.reindex(columns=loc_cols, fill_value=0)
    
    X = np.concatenate([
        X_numeric,
        avail_dummies.to_numpy(),
        area_dummies.to_numpy(),
        loc_dummies.to_numpy(),
    ], axis=1)
    
    y = df_filtered["price"].to_numpy()
    
    print(f"Final X shape: {X.shape}, y shape: {y.shape}")
    
    # Use Random Forest for better accuracy
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale numeric features
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[:, :3] = scaler.fit_transform(X_train[:, :3])
    X_test_scaled[:, :3] = scaler.transform(X_test[:, :3])
    
    # Train Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train_scaled, y_train)
    
    # Print scores
    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)
    print(f"Train R2: {train_score:.4f}")
    print(f"Test R2: {test_score:.4f}")
    
    # Test the specific case: Uttarahalli, Built-up Area, Ready To Move, 1440 sqft, 3 BHK, 2 bath
    test_case = np.array([1440, 2, 3])  # sqft, bath, bhk
    test_case_scaled = scaler.transform(test_case.reshape(1, -1))
    
    # Create dummy variables for the test case
    loc_idx = location_values.index("Uttarahalli")
    area_idx = area_values.index("Built-up Area")
    avail_idx = availability_values.index("Ready To Move")
    
    loc_array = np.zeros(len(location_values))
    loc_array[loc_idx] = 1
    area_array = np.zeros(len(area_values))
    area_array[area_idx] = 1
    avail_array = np.zeros(len(availability_values))
    avail_array[avail_idx] = 1
    
    # Drop first category
    loc_array = loc_array[1:]
    area_array = area_array[1:]
    avail_array = avail_array[1:]
    
    # Combine features
    test_features = np.concatenate([test_case_scaled.flatten(), avail_array, area_array, loc_array])
    test_prediction = model.predict(test_features.reshape(1, -1))[0]
    print(f"Test prediction for Uttarahalli case: {test_prediction:.2f} (expected: 62)")
    
    # Save model and scaler together
    model_data = {
        'model': model,
        'scaler': scaler,
        'feature_names': {
            'numeric_features': ['total_sqft', 'bath', 'bhk'],
            'availability_categories': availability_values,
            'area_categories': area_values,
            'location_categories': location_values
        }
    }
    
    with open(MODEL_PKL_PATH, "wb") as f:
        pickle.dump(model_data, f)
    
    print(f"Model trained and saved to {MODEL_PKL_PATH}")


if __name__ == "__main__":
    main()


