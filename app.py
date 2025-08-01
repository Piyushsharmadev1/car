import streamlit as st
import pandas as pd
import numpy as np
import pickle
from dotenv import load_dotenv
import os
import google.generativeai as genai

# ✅ Load environment variable
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# ✅ Load ML model and dataset
model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))
car = pd.read_csv('Cleaned_Car_data.csv')

# ✅ Initialize Gemini model
model_gen = genai.GenerativeModel("gemini-1.5-flash")  # or "gemini-1.5-pro"

# ✅ GenAI Functions using Gemini
def generate_price_explanation(car_info, predicted_price):
    prompt = f"""
You are an expert in car valuation. A user has provided the following car information:

{car_info}

Based on a machine learning model, the predicted price is ₹{predicted_price}.

Explain in simple terms why the price might be this value, considering factors like brand, year, kms driven, fuel type, transmission, and owner type.
"""
    response = model_gen.generate_content(prompt)
    return response.text

def generate_ad_description(car_details):
    prompt = f"""Write a catchy 2-3 line advertisement to help sell this car:
{car_details}"""
    response = model_gen.generate_content(prompt)
    return response.text

# ✅ Custom CSS
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

# ✅ Dropdown values
companies = sorted(car['company'].unique())
car_models = sorted(car['name'].unique())
years = sorted(car['year'].unique(), reverse=True)
fuel_types = sorted(car['fuel_type'].unique())

# ✅ UI Layout
st.markdown('<div class="main">', unsafe_allow_html=True)
st.markdown("<h1>Welcome to Car Price Predictor</h1>", unsafe_allow_html=True)
st.write("This app predicts the price of a car you want to sell. Try filling the details below:")

# ✅ User Inputs
company = st.selectbox("Select the company:", companies)
model_name = st.selectbox("Select the model:", car_models)
year = st.selectbox("Select Year of Purchase:", years)
fuel = st.selectbox("Select the Fuel Type:", fuel_types)
kms_driven = st.text_input("Enter the Number of Kilometres that the car has travelled:", placeholder="Enter the kilometres driven")

# ✅ Prediction
prediction = None
car_info = None

if st.button("Predict Price"):
    try:
        driven = int(kms_driven)
        input_df = pd.DataFrame(
            [[model_name, company, int(year), driven, fuel]],
            columns=['name', 'company', 'year', 'kms_driven', 'fuel_type']
        )
        prediction = model.predict(input_df)[0]
        st.success(f"💰 Estimated Selling Price: ₹ {np.round(prediction, 2)}")
        car_info = f"{company} {model_name}, Year: {year}, Fuel: {fuel}, {driven} km driven"

        st.session_state["last_pred"] = np.round(prediction, 2)
        st.session_state["car_info"] = car_info

    except ValueError:
        st.error("❌ Please enter a valid number for kilometres driven.")

# ✅ GenAI Buttons
if "last_pred" in st.session_state and "car_info" in st.session_state:
    if st.button("💡 Why this price?"):
        explanation = generate_price_explanation(st.session_state["car_info"], st.session_state["last_pred"])
        st.info("📘 Price Explanation:")
        st.write(explanation)

    if st.button("📣 Generate Car Ad"):
        ad_text = generate_ad_description(st.session_state["car_info"])
        st.info("📝 Car Ad Description:")
        st.write(ad_text)

st.markdown("</div>", unsafe_allow_html=True)
