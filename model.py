import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBRegressor


# 1. Load the dataset
data = pd.read_csv("data/air_quality.csv")


# 2. Convert city names into numbers
city_encoder = LabelEncoder()
data["City"] = city_encoder.fit_transform(data["City"])


# 3. Select input features
X = data[["City", "PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]]


# 4. Select target value
y = data["AQI"]


# 5. Scale the input data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# 6. Create XGBoost model
model = XGBRegressor(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    random_state=42
)


# 7. Train the model
model.fit(X_scaled, y)


# 8. Save the model
joblib.dump(model, "xgb_model.pkl")

# Save scaler
joblib.dump(scaler, "scaler.pkl")

# Save city encoder
joblib.dump(city_encoder, "city_encoder.pkl")


print("Model trained successfully!")
print("Model, scaler and city encoder saved successfully.")