from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import os

app = Flask(__name__)
CORS(app)

# -------------------------------
# 1️⃣ Load model and scaler
# -------------------------------
MODEL_PATH = os.path.join(os.getcwd(), 'model.pkl')
SCALER_PATH = os.path.join(os.getcwd(), 'scaler.pkl')
DATA_PATH = os.path.join(os.getcwd(), 'dataset', 'study_hours.csv')

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("Model and scaler loaded successfully.")
except Exception as e:
    print("Error loading model or scaler:", e)

# -------------------------------
# 2️⃣ /predict route
# -------------------------------

@app.route("/")
def home():
    return "Study Hours Prediction API is Live 🚀"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = np.array([[
            data['past_score'],
            data['subjects'],
            data['last_week_hours'],
            data['stress_level'],
            data['sleep_hours'],
            data['target_score']
        ]])
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)
        return jsonify({'recommended_study_hours': round(float(prediction[0]), 1)})
    except Exception as e:
        return jsonify({'error': str(e)})



# -------------------------------
# 4️⃣ Run Flask app
# -------------------------------
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
