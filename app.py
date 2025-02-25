
import pickle
import streamlit as st
import numpy as np
import pandas as pd

# Load your model file
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title('Car Brand Price Predictor App')

# Add input widgets for user inputs
Engine_Size = st.selectbox(
    "Engine_Size", [0:5]
)
price_in_thousands = st.slider("Price (thousand dollars)", min_value=3, max_value=53, value=26)
Mileage = st.slider("Mileage", min_value=1, max_value=5, value=2)

# When the 'Predict' button is clicked
if st.button("Predict"):
    # Prepare the input data as a DataFrame (since pipelines often expect a DataFrame)
    input_data = pd.DataFrame({
        'Year': [Year],
        'Engine_Size': [Engine_Size],
        'Mileage': [Mileage]
    })
    prediction = model.predict(input_data)[0].round(2)
    st.write(f'The predicted value is: {prediction} thousand dollars')
