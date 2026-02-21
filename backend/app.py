from flask import Flask, request, jsonify,render_template
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
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'scaler.pkl')
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
    return render_template("index.html")

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



