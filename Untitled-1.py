@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        data = request.get_json()
        features = np.array([
            data["ecg_signal"],
            data["respiratory_rate"],
            data["oxygen_level"],
            data["heart_rate"],
            data["blood_pressure"],
            data["temperature"],
            data["st_depression"],
        ]).reshape(1, -1)

        # Preprocess Data
        scaled_features = scaler.transform(features)

        # Predict
        prediction_prob = model.predict_proba(scaled_features).max() * 100
        prediction = int(model.predict(scaled_features)[0])

        return jsonify({
            "prediction": "High Risk" if prediction == 1 else "Low Risk",
            "confidence": round(prediction_prob, 2)
        })
    except Exception as e:
        return jsonify({"error": str(e)})
