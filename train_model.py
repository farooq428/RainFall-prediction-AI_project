import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Phase 1: Raw Data Compilation [cite: 5]
try:
    df = pd.read_csv("./dataset/rainfall.csv")
except FileNotFoundError:
    print("Error: Place 'rainfall.csv' in the project folder.")
    exit()

# Phase 2: Preprocessing & Feature Selection [cite: 6, 9]
df['RainTomorrow'] = (df['Rainfall'].shift(-1) >= 1).astype(int)
df = df.dropna()

features = ["MinTemp", "MaxTemp", "Rainfall", "Evaporation", "Sunshine", 
            "WindGustSpeed", "WindSpeed9am", "WindSpeed3pm", "Humidity9am", 
            "Humidity3pm", "Pressure9am", "Pressure3pm", "Cloud9am", "Cloud3pm", "Temp9am", "Temp3pm"]

X = df[features]
y = df['RainTomorrow']

# 80% Training - 20% Testing [cite: 11, 184, 186]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Phase 3: Random Forest Algorithm 
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Save Best Models [cite: 24]
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(features, "feature_names.pkl")
print("Model Components Saved Successfully.")