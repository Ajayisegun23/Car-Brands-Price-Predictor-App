
import pickle
import streamlit as st
import numpy as np
import pandas as pd

# Load your model file
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title('Car Resales Price Predictor App')

# Add input widgets for user inputs
Year = st.selectbox(
    "Year", [1999,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023,2024,2025]
)
Engine_Size = st.slider("Engine_Size", min_value=2, max_value=6, value=2.5)
Mileage = st.slider("Mileage", min_value=1, max_value=300000, value=150000)

# When the 'Predict' button is clicked
if st.button("Predict"):
    # Prepare the input data as a DataFrame (since pipelines often expect a DataFrame)
    input_data = pd.DataFrame({
        'Year': [Year],
        'Engine_Size': [Engine_Size],
        'Mileage': [Mileage]
    })
    prediction = model.predict(input_data)[0].round(2)
    st.write(f'The predicted Car Resale Value is: {prediction} thousand dollars')
