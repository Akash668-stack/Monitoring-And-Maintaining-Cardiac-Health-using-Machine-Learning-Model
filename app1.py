from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

# Initialize Flask App
app = Flask(__name__)

# Load Pretrained Model & Scaler
try:
    model = pickle.load(open("static/model/cardiac_model.sav", "rb"))  # CNN + LSTM + SVM + RF
    scaler = pickle.load(open("static/model/scaler.pkl", "rb"))  # Feature Scaler
except Exception as e:
    print(f"Error loading model or scaler: {e}")

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/predict", methods=["POST", "GET"])
def predict():
    if request.method == "POST":
        try:
            # Extract form data
            form_data = request.form.to_dict()
            features = [
                float(form_data["ecg_signal"]),
                float(form_data["respiratory_rate"]),
                float(form_data["oxygen_level"]),
                float(form_data["heart_rate"]),
                float(form_data["blood_pressure"]),
                float(form_data["temperature"]),
                float(form_data["st_depression"]),
            ]

            # Preprocess Data
            scaled_features = scaler.transform([features])

            # Model Prediction
            prediction_prob = model.predict_proba(scaled_features).max() * 100
            prediction = int(model.predict(scaled_features)[0])

            # Prepare Result
            result = {
                "prediction": "High Risk" if prediction == 1 else "Low Risk",
                "prediction_prob": round(prediction_prob, 2),
                "features": form_data
            }

            return render_template("result.html", results=result)

        except Exception as e:
            return jsonify({"error": f"Prediction failed: {e}"})

    return render_template("cardiac_form.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

if __name__ == "__main__":
    app.run(debug=True)
