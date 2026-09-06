from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib

app = Flask(__name__)
CORS(app)  # Allows the Flutter web app to call this API from a different origin

# Load the trained model and label encoder once, when the server starts
model = joblib.load('portion_model.pkl')
le = joblib.load('category_encoder.pkl')


@app.route('/', methods=['GET'])
def home():
    """Simple health-check route to confirm the API is running."""
    return jsonify({
        "status": "running",
        "message": "Portion Size Prediction API is live",
        "valid_categories": list(le.classes_)
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Expects a JSON body like:
    {
        "food_name": "Chicken Biryani",
        "weight_grams": 350,
        "calories": 480,
        "category": "Main Course"
    }

    Returns a JSON response with the predicted portion size and confidence.
    """
    try:
        data = request.get_json()

        # Basic validation - make sure all required fields are present
        required_fields = ["food_name", "weight_grams", "calories", "category"]
        missing = [f for f in required_fields if f not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {missing}"}), 400

        food_name = data["food_name"]
        weight_grams = data["weight_grams"]
        calories = data["calories"]
        category = data["category"]

        # Reject categories the model was never trained on
        if category not in le.classes_:
            return jsonify({
                "error": f"'{category}' is an unknown category.",
                "valid_categories": list(le.classes_)
            }), 400

        # Encode category into the numeric form the model expects
        category_encoded = le.transform([category])[0]

        # Build input dataframe with the same column names/order used in training
        input_df = pd.DataFrame(
            [[weight_grams, calories, category_encoded]],
            columns=['Weight_grams', 'Calories', 'Category_Encoded']
        )

        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
        confidence = round(max(probabilities) * 100, 1)

        return jsonify({
            "food": food_name,
            "predicted_portion": prediction,
            "confidence": f"{confidence}%"
        })

    except Exception as e:
        # Catch-all so the API never crashes silently on bad input
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    import os
    # Render (and most hosting platforms) assign a PORT via environment variable.
    # Falls back to 5000 for local testing.
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)