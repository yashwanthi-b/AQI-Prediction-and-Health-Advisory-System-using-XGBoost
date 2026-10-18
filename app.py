from flask import Flask, render_template, request
import joblib
import pandas as pd


app = Flask(__name__)


# Load trained model, scaler and city encoder
model = joblib.load("xgb_model.pkl")
scaler = joblib.load("scaler.pkl")
city_encoder = joblib.load("city_encoder.pkl")

# Cities the model was trained on
cities = sorted(city_encoder.classes_)


def get_category_and_advice(aqi):
    """Indian National AQI (CPCB) categories."""
    if aqi <= 50:
        return "Good", "Air quality is good. Minimal impact on health."
    elif aqi <= 100:
        return "Satisfactory", "Minor breathing discomfort to sensitive people."
    elif aqi <= 200:
        return "Moderate", "People with asthma, heart or lung disease should reduce prolonged outdoor activity."
    elif aqi <= 300:
        return "Poor", "Most people may feel breathing discomfort on prolonged exposure. Limit outdoor activity."
    elif aqi <= 400:
        return "Very Poor", "Risk of respiratory illness on prolonged exposure. Avoid outdoor activity."
    else:
        return "Severe", "Serious health impact, even on healthy people. Stay indoors."


# Home page
@app.route("/")
def home():
    return render_template("index.html", cities=cities)


# Prediction
@app.route("/predict", methods=["POST"])
def predict():
    try:
        city = request.form["city"]
        pm25 = float(request.form["pm25"])
        pm10 = float(request.form["pm10"])
        no2 = float(request.form["no2"])
        so2 = float(request.form["so2"])
        co = float(request.form["co"])
        o3 = float(request.form["o3"])
    except (KeyError, ValueError):
        return render_template(
            "index.html", cities=cities,
            error="Please enter valid numbers in all fields."
        )

    if city not in cities:
        return render_template(
            "index.html", cities=cities,
            error="Please select a valid city."
        )

    # Convert city to number
    city_encoded = city_encoder.transform([city])[0]

    # Create input dataframe
    input_data = pd.DataFrame(
        [[city_encoded, pm25, pm10, no2, so2, co, o3]],
        columns=["City", "PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]
    )

    # Scale input and predict AQI
    input_scaled = scaler.transform(input_data)
    prediction = max(0, round(float(model.predict(input_scaled)[0]), 2))

    category, recommendation = get_category_and_advice(prediction)

    return render_template(
        "index.html",
        cities=cities,
        selected_city=city,
        prediction=prediction,
        category=category,
        recommendation=recommendation
    )


if __name__ == "__main__":
    app.run(debug=True)