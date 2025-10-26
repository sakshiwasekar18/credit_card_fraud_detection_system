import joblib
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
import re

app = Flask(__name__)

# --- Model Loading ---
try:
    # Model name confirmed as 'fraud_detection_model.joblib'
    # NOTE: This file is required to run real predictions.
    model = joblib.load('fraud_detection_model.joblib')
    print("Model loaded successfully.")
except FileNotFoundError:
    print("WARNING: Model file 'fraud_detection_model.joblib' not found.")
    print("Using dummy prediction logic.")
    model = None

# --- Helper Function for Prediction ---

def predict_transaction(features):
    """Makes a prediction and calculates the confidence score."""
    if model is None:
        # Dummy logic if model file is missing 
        v16_index = 15 
        if features[v16_index] < -5.0:
            return 1, 0.95 # Fraud, 95% confidence
        else:
            return 0, 0.05 # Valid, 5% fraud confidence

    # Reshape features for the model (single sample, 29 features)
    features_array = np.array(features).reshape(1, -1)
    
    # Predict the class (0 for Valid, 1 for Fraud)
    prediction = model.predict(features_array)[0]
    
    # Predict the probability for each class (e.g., [prob_0, prob_1])
    probabilities = model.predict_proba(features_array)[0]
    
    # Use the probability of the Fraud class (Class 1) for the confidence score
    confidence = probabilities[1] 

    return prediction, confidence

# --- Flask Routes ---

@app.route('/')
def welcome():
    """Serves the welcome page (welcome.html)."""
    return render_template('welcome.html')

@app.route('/home.html')
def home():
    """Serves the input page (home.html)."""
    return render_template('home.html')

@app.route('/valid.html')
def valid_result():
    """Serves the valid result page (valid.html)."""
    # The probability is expected to be passed via URL parameter
    return render_template('valid.html')

@app.route('/fraud.html')
def fraud_result():
    """Serves the fraud result page (fraud.html)."""
    # The probability is expected to be passed via URL parameter
    return render_template('fraud.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles the prediction request and redirects to the appropriate result page."""
    
    # 1. Get raw text input from the form
    raw_text = request.form['featureInput']

    # 2. Clean and parse the input string to extract numbers
    string_values = re.split(r'[\s,]+', raw_text.strip())
    string_values = [v for v in string_values if v] 

    try:
        features = [float(v) for v in string_values]
    except ValueError:
        print("Error: Non-numeric input detected. Redirecting to home.")
        # In a production environment, you might display an error message here.
        return redirect(url_for('home'))

    # 3. Validate the feature count
    if len(features) != 29:
        print(f"Error: Expected 29 features, got {len(features)}. Redirecting to home.")
        return redirect(url_for('home'))

    # 4. Make the prediction
    prediction, probability_fraud = predict_transaction(features)

    # 5. Redirect to the appropriate result page, passing the probability in the URL
    if prediction == 1:
        # Fraud detected, redirect to fraud.html
        return redirect(url_for('fraud_result') + f"?prob={probability_fraud}")
    else:
        # Valid transaction, redirect to valid.html
        return redirect(url_for('valid_result') + f"?prob={probability_fraud}")

if __name__ == '__main__':
    app.run(debug=True)
