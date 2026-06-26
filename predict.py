import joblib

model = joblib.load("stock_prediction_model.pkl")
encoder = joblib.load("stock_encoder.pkl")

print("Model Loaded Successfully")
print("Encoder Loaded Successfully")