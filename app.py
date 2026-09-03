from flask import Flask, render_template, request
import joblib
import pandas as pd


app = Flask(__name__)


# Load trained model
model = joblib.load("xgb_model.pkl")

# Load scaler
scaler = joblib.load("scaler.pkl")

# Load city encoder
city_encoder = joblib.load("city_encoder.pkl")


# Home page
@app.route("/")
def home():
    return render_template("index.html")


# Prediction
@app.route("/predict", methods=["POST"])
def predict():

    city = request.form["city"]
    pm25 = float(request.form["pm25"])
    pm10 = float(request.form["pm10"])
    no2 = float(request.form["no2"])
    so2 = float(request.form["so2"])
    co = float(request.form["co"])
    o3 = float(request.form["o3"])

    # Convert city to number
    city_encoded = city_encoder.transform([city])[0]

    # Create input dataframe
    input_data = pd.DataFrame([[
        city_encoded,
        pm25,
        pm10,
        no2,
        so2,
        co,
        o3
    ]], columns=[
        "City",
        "PM2.5",
        "PM10",
        "NO2",
        "SO2",
        "CO",
        "O3"
    ])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Predict AQI
    prediction = model.predict(input_scaled)[0]

    prediction = round(prediction, 2)


    # Health recommendation
    if prediction <= 50:
        recommendation = "Good air quality. Air pollution poses little or no risk."
    elif prediction <= 100:
        recommendation = "Moderate air quality. Sensitive people should take care."
    elif prediction <= 150:
        recommendation = "Unhealthy for sensitive groups. Reduce prolonged outdoor activity."
    elif prediction <= 200:
        recommendation = "Unhealthy air quality. Consider reducing outdoor activities."
    elif prediction <= 300:
        recommendation = "Very unhealthy air quality. Avoid prolonged outdoor exposure."
    else:
        recommendation = "Hazardous air quality. Avoid outdoor activities."


    return render_template(
        "index.html",
        prediction=prediction,
        recommendation=recommendation
    )


if __name__ == "__main__":
    app.run(debug=True)