import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor


# 1. Load the dataset
data = pd.read_csv("data/city_day.csv")
print("Rows in file:", len(data))


# 2. Keep only the columns we need and remove rows with missing values
columns = ["City", "PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "AQI"]
data = data[columns].dropna()
print("Rows after removing missing values:", len(data))
print("Cities:", data["City"].nunique())


# 3. Convert city names into numbers
city_encoder = LabelEncoder()
data["City"] = city_encoder.fit_transform(data["City"])


# 4. Select input features and target
X = data[["City", "PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]]
y = data["AQI"]


# 5. Split into training and testing data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 6. Scale the input data (fit on training data only)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# 7. Create and train the XGBoost model
model = XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)
model.fit(X_train_scaled, y_train)


# 8. Evaluate the model on unseen test data
predictions = model.predict(X_test_scaled)

r2 = r2_score(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
mae = mean_absolute_error(y_test, predictions)

print("R2 Score :", round(r2, 4))
print("RMSE     :", round(rmse, 2))
print("MAE      :", round(mae, 2))


# 9. Save the model, scaler and city encoder
joblib.dump(model, "xgb_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(city_encoder, "city_encoder.pkl")

print("Model, scaler and city encoder saved successfully.")