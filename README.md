🏠 House Price Prediction (Streamlit App)
A Machine Learning project that predicts house prices based on property features like area, bedrooms, bathrooms, stories, and parking.
This project uses Python, scikit-learn, and Streamlit to create a simple interactive app for local use.

📌 Features
Predicts house prices using a trained Linear Regression model
Simple and easy-to-use Streamlit interface
Uses 5 numeric inputs for prediction
Runs entirely on localhost — no deployment needed

📊 How It Works
Training
Loads the dataset (Housing.csv)
Uses Linear Regression with data scaling
Saves the trained model as a .pkl file
Prediction
User enters details (area, bedrooms, bathrooms, stories, parking)
The app scales inputs and predicts the price
Displays result instantly
