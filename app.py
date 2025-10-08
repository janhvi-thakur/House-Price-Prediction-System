import streamlit as st
import pandas as pd
import pickle

# Load trained pipeline
model_pipeline = pickle.load(open("model/house_price_model.pkl", "rb"))

st.title("🏠 House Price Prediction App")
st.write("Enter the details below to predict the house price.")

# Numeric inputs only
area = st.number_input("Area (sq ft)", min_value=500, max_value=20000, step=100)
bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10)
bathrooms = st.number_input("Bathrooms", min_value=1, max_value=5)
stories = st.number_input("Number of Stories", min_value=1, max_value=5)
parking = st.number_input("Parking Spaces", min_value=0, max_value=5)

# Prediction button
if st.button("Predict Price"):
    # Prepare dataframe for prediction
    input_df = pd.DataFrame([[
        area, bedrooms, bathrooms, stories, parking
    ]], columns=["area", "bedrooms", "bathrooms", "stories", "parking"])

    # Predict
    predicted_price = model_pipeline.predict(input_df)[0]
    st.success(f"Estimated Price: ₹{predicted_price:,.2f}")
