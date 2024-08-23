import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the model
with open('price_model.pkl', 'rb') as file:
    model = pickle.load(file)
    
# Load the feature names
with open('feature_names.pkl', 'rb') as file:
    feature_names = pickle.load(file)

# Function to preprocess input data
def preprocess_input(data):
    input_df = pd.DataFrame([data])
    input_df = pd.get_dummies(input_df, columns=['Fuel_Type','Transmission'], drop_first=True)

    # Ensure all expected columns are present
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0
    
    # Reorder the columns to match the training order
    input_df = input_df[feature_names]
    
    return input_df

# Function to predict car price
def predict_price(model, input_data):
    preprocessed_data = preprocess_input(input_data)
    return model.predict(preprocessed_data)

# Streamlit application
st.title("Bike and Car Price Prediction")

# Collect user input
Year = st.number_input('Year of Purchase', min_value=2000, max_value=2024, value=2010, step=1)
Present_Price = st.number_input('Present Price (in Lakhs)', min_value=0.0, max_value=50.0, value=5.0, step=0.1)
Driven_kms = st.number_input('Distance Driven (in km)', min_value=0, max_value=500000, value=15000, step=1000)
Fuel_Type = st.selectbox('Fuel Type', ['Petrol', 'Diesel', 'Electric'])
Transmission = st.selectbox('Transmission', ['Manual', 'Automatic'])

# Create a dictionary from the user input
input_data = {
    'Year': Year,
    'Present_Price': Present_Price,
    'Driven_kms': Driven_kms,
    'Fuel_Type': Fuel_Type,
    'Transmission': Transmission
}
    
# When the user clicks the button, predict the price
b1,sub,b2 = st.columns(3)
if sub.button('Predict Price'):
    prediction = predict_price(model, input_data)
    st.write(f"Predicted price : {prediction[0]:.2f} Lakhs")


