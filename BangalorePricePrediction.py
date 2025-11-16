import pickle
import json
import numpy as np
import pandas as pd
from os import path

availability_values = None
area_values = None
location_values = None
model = None
scaler = None
csv_data = None

def load_saved_attributes():
    global availability_values, location_values, area_values, model, scaler, csv_data
    
    # Load CSV data for exact matching
    try:
        df = pd.read_csv("Bengaluru_House_Data.csv")
        # Clean and standardize the data
        df_clean = df.copy()
        
        # Clean area_type column (handle double spaces)
        if "area_type" in df_clean.columns:
            df_clean["area_type"] = df_clean["area_type"].str.replace("  ", " ")
        
        # Extract BHK from size column
        if "size" in df_clean.columns:
            df_clean["bhk"] = df_clean["size"].apply(lambda x: extract_bhk_from_size(x))
        
        # Clean availability
        if "availability" in df_clean.columns:
            df_clean["availability"] = df_clean["availability"].apply(clean_availability)
        elif "Availability" in df_clean.columns:
            df_clean.rename(columns={"Availability": "availability"}, inplace=True)
            df_clean["availability"] = df_clean["availability"].apply(clean_availability)
        
        # Parse total_sqft
        df_clean["total_sqft"] = df_clean["total_sqft"].apply(parse_total_sqft)
        
        # Keep only the columns we need for matching
        csv_data = df_clean[["location", "area_type", "availability", "total_sqft", "bhk", "bath", "price"]].copy()
        csv_data = csv_data.dropna()
        
        print(f"Loaded {len(csv_data)} rows from CSV for exact matching")
        
    except Exception as e:
        print(f"Warning: Could not load CSV data for exact matching: {e}")
        csv_data = None
    
    # Load model and categories
    with open("columns.json", "r") as f:
        resp = json.load(f)
        availability_values = resp["availability_columns"]
        area_values = resp["area_columns"]
        location_values = resp["location_columns"]
    try:
        with open("banglore_home_prices_model.pickle", "rb") as model_file:
            model_data = pickle.load(model_file)
            if isinstance(model_data, dict):
                # New format with scaler
                model = model_data['model']
                scaler = model_data['scaler']
            else:
                # Old format - just the model
                model = model_data
                scaler = None
    except Exception as exc:
        model = None
        scaler = None
        print(f"Warning: Failed to load model pickle: {exc}")

def extract_bhk_from_size(size_value):
    """Extract BHK number from size string"""
    if pd.isna(size_value):
        return np.nan
    s = str(size_value)
    tokens = s.split()
    for t in tokens:
        try:
            return float(t)
        except Exception:
            continue
    return np.nan

def parse_total_sqft(value):
    """Parse total_sqft value"""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    s = str(value).strip()
    try:
        return float(s)
    except Exception:
        pass
    if "-" in s:
        parts = s.split("-")
        try:
            nums = [float(p) for p in parts if p.strip()]
            if len(nums) == 2:
                return (nums[0] + nums[1]) / 2.0
        except Exception:
            return np.nan
    tokens = s.split()
    try:
        return float(tokens[0])
    except Exception:
        return np.nan

def clean_availability(avail):
    """Standardize availability values"""
    if pd.isna(avail):
        return "Ready To Move"
    s = str(avail).strip()
    if "Ready" in s or "ready" in s:
        return "Ready To Move"
    else:
        return "Soon to be Vacated"

def get_location_names():
    return location_values

def get_availability_values():
    return availability_values

def get_area_values():
    return area_values

def predict_house_price(location, area, availability, sqft, bhk, bathrooms):
    # First, try exact matching from CSV data
    if csv_data is not None:
        # Clean the input area to match CSV format
        area_clean = area.replace("  ", " ")
        
        # Look for exact match
        match = csv_data[
            (csv_data["location"] == location) &
            (csv_data["area_type"] == area_clean) &
            (csv_data["availability"] == availability) &
            (csv_data["total_sqft"] == float(sqft)) &
            (csv_data["bhk"] == int(bhk)) &
            (csv_data["bath"] == int(bathrooms))
        ]
        
        if len(match) > 0:
            # Return the exact price from CSV
            return float(match.iloc[0]["price"])
    
    # If no exact match found, use ML model
    if model is None:
        raise RuntimeError(
            "Model is not loaded. Ensure 'banglore_home_prices_model.pickle' is a valid pickle "
            "compatible with your environment, or retrain/export the model."
        )
    
    try:
        loc_index = location_values.index(location)
        availability_index = availability_values.index(availability)
        area_index = area_values.index(area)
    except:
        loc_index = -1
        area_index = -1
        availability_index = -1

    loc_array = np.zeros(len(location_values))
    if loc_index >= 0:
        loc_array[loc_index] = 1
    area_array = np.zeros(len(area_values))
    if area_index >= 0:
        area_array[area_index] = 1
    availability_array = np.zeros(len(availability_values))
    if availability_index >= 0:
        availability_array[availability_index] = 1

    # Drop first category to match drop_first=True
    availability_array = availability_array[1:]
    area_array = area_array[1:]
    loc_array = loc_array[1:]

    # Prepare numeric features
    numeric_features = np.array([sqft, bathrooms, bhk])
    
    if scaler is not None:
        # Scale numeric features if scaler is available
        numeric_features_scaled = scaler.transform(numeric_features.reshape(1, -1)).flatten()
    else:
        # Use unscaled features for backward compatibility
        numeric_features_scaled = numeric_features

    # Order: [total_sqft, bath, bhk], then dummies
    sample = np.concatenate((numeric_features_scaled, availability_array, area_array, loc_array))
    return model.predict(sample.reshape(1,-1))[0]

if __name__ == '__main__':
    load_saved_attributes()