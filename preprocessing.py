import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib

# The EXACT order the model expects
FEATURES = [
    "MinTemp", "MaxTemp", "Sunshine", "Evaporation",
    "WindGustSpeed", "WindSpeed9am", "WindSpeed3pm",
    "Humidity9am", "Humidity3pm", "Pressure9am", "Pressure3pm",
    "Cloud9am", "Cloud3pm", "Temp9am", "Temp3pm"
]

def load_and_preprocess():
    df = pd.read_csv("dataset/rainfall.csv")
    
    # 1. Fill missing values instead of dropping (Median is safer for weather)
    imputer = SimpleImputer(strategy="median")
    
    # Target (y) is Rainfall
    y = df["Rainfall"].fillna(0) # If target is missing, assume 0
    
    # Features (X) - Only keep the ones we defined in FEATURES
    # This handles the text columns automatically by only picking these
    X = df[FEATURES].copy()
    X_imputed = imputer.fit_transform(X)
    X_final = pd.DataFrame(X_imputed, columns=FEATURES)

    # 2. Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_final)
    
    # Save scaler and the feature list for the App to reference
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(FEATURES, "feature_names.pkl")

    return X_scaled, y