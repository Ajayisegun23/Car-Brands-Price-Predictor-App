# Car-Brands-Resale price-Predictor Model

Multilinear Regression Model: Car Brands Price Predictor Model
This model outlines the process of developing and evaluating a multilinear regression model to predict the prices of car brands in United State of America. The model leverages the relationship between Price as the Target with feature components to predict the actual price base on these relationships, and its performance is measured using the Mean Absolute Error (MAE) metric.

Problem Statement
• Objective
To create a predictive model that can predict the prices of car model based on selected features mix.
• Goal
The model will take input values for selected features like Engine_Size (is a measurement of the total volume of the cylinders in a car’s engine), Mileage (The distance travelled) and Year (Manufacture year) predict the inflation rate, providing a tool for forecasting and understanding economic trends.
Data Description
Car Price Prediction Dataset Description:
This dataset contains 10,000 entries created for the purpose of predicting car prices. Each row represents information about a car and its price. The descriptions of the columns are as follows:
Columns Descriptions:
Brand: Specifies the brand of the car (e.g., Toyota, BMW, Ford). Example values: "Toyota", "BMW", "Mercedes".
Model: Specifies the model of the car (e.g., Corolla, Focus, X5). Example values: "Corolla", "Focus", "X5".
Year: The production year of the car. Newer years typically indicate higher prices. Example values: 2005, 2018, 2023.
Engine_Size: Specifies the engine size in litres (L). Larger engines generally correlate with higher prices.
Example values: 1.6, 2.0, 3.5.
Fuel_Type: indicates the type of fuel used by the car:
Petrol: Cars running on gasoline.
Diesel: Cars running on diesel fuel.
Hybrid: Cars that use both fuel and electricity.
Electric: Fully electric cars.
Transmission: The type of transmission in the car:
Manual: Manual transmission.
Automatic: Automatic transmission.
Semi-Automatic: Semi-automatic transmission.
Mileage: The total distance the car has travelled, measured in kilometres. Lower mileage generally indicates a higher price. Example values: 15,000, 75,000, 230,000.
Doors: The number of doors in the car. Commonly 2, 3, 4, or 5 doors. Example values: 2, 3, 4, 5.
Owner_Count: The number of previous owners of the car. Fewer owners generally indicate a higher price. Example values: 1, 2, 3, 4.
Price: The estimated selling price of the car. It is calculated based on several factors such as production year, engine size, mileage, fuel type, and transmission. Example values: 5,000, 15,000, 30,000.

Model Training
Import Statements and Dataset
To begin with, the necessary libraries and modules are imported, and the dataset is loaded into the model.
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from category_encoders import OneHotEncoder
from sklearn.utils.validation import check_is_fitted
from ipywidgets import Dropdown, FloatSlider, IntSlider, interact
import streamlit as st
import pickle
df = pd.read_csv("/content/drive/MyDrive/Car price dataset  analysis/car_price_dataset.csv")

Data Splitting
The dataset contains both features (independent variables) Engine_Size, Mileage, and Year against the target variable (Price). 
defining the columns for feature and target
feature = ["Year", "Engine_Size", "Mileage"] 
target = "Price"
The data is split into training and test sets, with an 80%/20% ratio for training and testing, respectively:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"X (feature)=>\ttrain: {X_train.shape} test:{X_test.shape} \ny (target)=>\ttrain:{y_train.shape} test:{y_test.shape}")
X = data[“Engine_Size”, “Mileage”, “Year”]
y = data[”Price”]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
Model Training
The multilinear regression model is trained using the training set (X_train and y_train):
model = make_pipeline(OneHotEncoder(), SimpleImputer(), Ridge())
model.fit(X_train, y_train)

Model Evaluation
After training the model, predictions are made on both the training and test sets. The performance is evaluated using the Mean Absolute Error (MAE) metric, which measures the average magnitude of errors in a set of predictions. A lower MAE indicates better model accuracy.
• Baseline Model MAE: 2553.9234 (This is the performance of a simple guess, like predicting the average value of the target variable for all data points.)
• Model (Train) MAE: 929.1285 (The MAE on the training data, showing how well the model fits the training set.)
• Model (Test) MAE: 888.9577 (The MAE on the test data, showing how well the model generalizes to unseen data.)
The test MAE of 888.9577 demonstrates that the model’s predictions are closer to the actual inflation rates compared to the baseline model (MAE of 2553.9234), indicating an improvement in predictive accuracy.
Make predictions on train and test sets
y_pred_training = model.predict(X_train)
print("Training MAE:", round(mean_absolute_error(y_train, y_pred_training), 4))

y_pred_test = model.predict(X_test)
print("Test MAE:", round(mean_absolute_error(y_test, y_pred_test),4))


Conclusion
The model has demonstrated a significant improvement in predictive accuracy compared to a baseline model, achieving a test MAE of 888.9577 This indicates that the multilinear regression model, trained on the car dataset, is a useful tool for predicting Car prices.

Future Work
Future work can focus on:
• Incorporating non numerical values for better prediction accuracy
• Evaluating model performance over longer time periods or using different time series techniques for more accurate forecasting.
• Experimenting with advanced models to improve prediction accuracy and handle non-linearity in the data.

