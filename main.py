import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model and data
model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))
car = pd.read_csv('Cleaned_Car_data.csv')

# Custom CSS
st.markdown("""
    <style>
    body {
        background-color: #f1f2f6;
        font-family: 'Segoe UI', sans-serif;
    }
    .main {
        background-color: white;
        padding: 40px;
        border-radius: 10px;
        max-width: 700px;
        margin: auto;
        box-shadow: 0 0 10px rgba(0,0,0,0.1);
    }
    h1 {
        text-align: center;
        color: #2c3e50;
    }
    .stButton>button {
        width: 100%;
        background-color: #3378FF;
        color: white;
        font-size: 16px;
        padding: 10px;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# Prepare data for dropdowns
companies = sorted(car['company'].unique())
car_models = sorted(car['name'].unique())
years = sorted(car['year'].unique(), reverse=True)
fuel_types = sorted(car['fuel_type'].unique())

# Start layout
st.markdown('<div class="main">', unsafe_allow_html=True)
st.markdown("<h1>Welcome to Car Price Predictor</h1>", unsafe_allow_html=True)
st.write("This app predicts the price of a car you want to sell. Try filling the details below:")

# Form inputs
company = st.selectbox("Select the company:", companies)
model_name = st.selectbox("Select the model:", car_models)
year = st.selectbox("Select Year of Purchase:", years)
fuel = st.selectbox("Select the Fuel Type:", fuel_types)
kms_driven = st.text_input("Enter the Number of Kilometres that the car has travelled:", placeholder="Enter the kilometres driven")

# Predict button
if st.button("Predict Price"):
    try:
        driven = int(kms_driven)
        input_df = pd.DataFrame(
            [[model_name, company, int(year), driven, fuel]],
            columns=['name', 'company', 'year', 'kms_driven', 'fuel_type']
        )
        prediction = model.predict(input_df)[0]
        st.success(f"💰 Estimated Selling Price: ₹ {np.round(prediction, 2)}")
    except ValueError:
        st.error("❌ Please enter a valid number for kilometres driven.")

st.markdown("</div>", unsafe_allow_html=True)
