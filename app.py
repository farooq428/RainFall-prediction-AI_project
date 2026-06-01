from flask import Flask, render_template, request
import joblib, json, requests, numpy as np
from datetime import datetime

app = Flask(__name__)

# Load Saved Models [cite: 24]
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

with open('pakistan_cities.json', 'r', encoding='utf-8') as f:
    ALL_CITIES = sorted(json.load(f))

API_KEY = "133d554dfa4a38e05d6c80341e71458d"

def fetch_weather(city):
    query = city.strip()
    url = f"https://api.openweathermap.org/data/2.5/weather?q={query},PK&appid={API_KEY}&units=metric"
    r = requests.get(url).json()
    
    # City Search Fallback
    if r.get("cod") != 200:
        alt = query.replace("Ahmed", "Ahmad") if "Ahmed" in query else query.replace("Ahmad", "Ahmed")
        r = requests.get(f"https://api.openweathermap.org/data/2.5/weather?q={alt},PK&appid={API_KEY}&units=metric").json()
    
    if r.get("cod") != 200: return None
    
    m, w = r['main'], r['wind']
    # Mapping real-time API to 15 ML features [cite: 4, 168-173]
    vals = {
        "MinTemp": m['temp_min'], "MaxTemp": m['temp_max'], "Rainfall": 0.0,
        "Evaporation": 4.5, "Sunshine": 6.0, "WindGustSpeed": w.get('speed',0)*1.2,
        "WindSpeed9am": w['speed'], "WindSpeed3pm": w['speed']*1.1,
        "Humidity9am": m['humidity'], "Humidity3pm": m['humidity'], # Fix for UI blank
        "Pressure9am": m['pressure'], "Pressure3pm": m['pressure'],
        "Cloud9am": 2, "Cloud3pm": 4, "Temp9am": m['temp']-2, "Temp3pm": m['temp']+2
    }
    
    # 5-Day Forecast Data for Chart [cite: 49, 75]
    f_url = f"https://api.openweathermap.org/data/2.5/forecast?q={r['name']}&appid={API_KEY}&units=metric"
    f_res = requests.get(f_url).json()
    forecast = [{"day": e['dt'], "temp": e['main']['temp']} for e in f_res.get('list', [])[::8]]
    
    return [vals[f] for f in feature_names], r['name'], m, w, forecast

@app.route("/", methods=["GET", "POST"])
def home():
    city = request.form.get("city", "Faisalabad")
    weather = fetch_weather(city)
    today = datetime.now().strftime("%A, %d %B")

    if not weather:
        return render_template("index.html", error=True, cities=ALL_CITIES, city=city, today=today)

    feats, name, m, w, forecast = weather
    # Classification: Predicting Rain Probability [cite: 71, 136]
    prob = round(model.predict_proba(scaler.transform([feats]))[0][1] * 100, 1)

    return render_template("index.html", city=name, temp=round(m['temp']), 
                           prediction=prob, humidity=m['humidity'], wind=w['speed'],
                           forecast=forecast, cities=ALL_CITIES, today=today)

if __name__ == "__main__":
    app.run(debug=True)