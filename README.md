# AQI Prediction and Health Advisory System using XGBoost

## Project Overview

AQI Prediction and Health Advisory System using XGBoost is a machine learning based web application developed to predict Air Quality Index (AQI) using major air pollution parameters.

The system processes air quality data, performs data preprocessing and feature preparation, trains an XGBoost machine learning model, and provides AQI predictions through a Flask web application.

The project is designed around a data processing pipeline that connects data preparation, machine learning prediction, and a web interface into a single workflow.

## Objectives

- Predict Air Quality Index (AQI) from pollutant parameters using machine learning.
- Process and prepare air quality data for prediction.
- Support AQI prediction for multiple cities.
- Provide health recommendations based on predicted AQI.
- Develop a simple and user-friendly web interface.
- Build a reusable machine learning prediction system.

## Features

- Air Quality Index prediction
- Multiple city support
- PM2.5, PM10, NO2, SO2, CO and O3 inputs
- Data preprocessing using Pandas
- Feature scaling using Scikit-learn
- City encoding
- XGBoost machine learning model
- Flask web application
- Instant AQI prediction through a web interface
- Health recommendations based on predicted AQI
- Saved machine learning model and preprocessing files

## Technologies Used

### Programming Language
- Python

### Machine Learning and Data Processing
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Joblib

### Web Technologies
- Flask
- HTML
- CSS

### Development Tools
- Visual Studio Code
- Git
- GitHub

## System Workflow

```text
Air Quality Dataset
        |
        v
Data Collection
        |
        v
Data Preprocessing
        |
        v
Feature Engineering
        |
        v
City Encoding & Feature Scaling
        |
        v
XGBoost Model Training
        |
        v
Model Evaluation
        |
        v
Save Trained Model
        |
        v
Flask Web Application
        |
        v
User Input
        |
        v
AQI Prediction
        |
        v
Health Recommendation
