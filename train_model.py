import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Load dataset
df = pd.read_csv("data/Housing.csv")

# Select only numeric features you want
num_features = ["area", "bedrooms", "bathrooms", "stories", "parking"]
X = df[num_features]
y = df["price"]

# Create pipeline: Scaling + Linear Regression
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("regressor", LinearRegression())
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit model
pipeline.fit(X_train, y_train)

# Save trained model
with open("model/house_price_model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("✅ Model trained and saved as model/house_price_model.pkl")
