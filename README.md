# AI Portion Recommendation Module

A machine learning microservice that predicts the portion size (**Small / Medium / Large**) of a restaurant food item based on its weight, calorie count, and category. Built as part of the **Food AR** Final Year Project — this module powers the "AI Portion Recommendation" feature described in the project proposal.

🔗 **Live API:** [https://malaikashoukat.pythonanywhere.com](https://malaikashoukat.pythonanywhere.com)

---

## Overview

Restaurant menus rarely tell customers how big a portion actually is. This module uses a **Random Forest Classifier** trained on a restaurant food dataset to predict portion size, helping customers make better ordering decisions in the Flutter-based Food AR app.

| | |
|---|---|
| **Model** | Random Forest Classifier (scikit-learn) |
| **Accuracy** | 98.3% |
| **Classes** | Small, Medium, Large |
| **Features used** | Weight (grams), Calories, Category |
| **Dataset size** | 300 samples |

---

## Project Structure

```
portion-api/
├── app.py                    # Flask API (local development version)
├── train_model.py            # Training script — run to (re)train the model
├── requirements.txt          # Python dependencies
├── portion_model.pkl         # Trained Random Forest model
├── category_encoder.pkl      # LabelEncoder for the Category feature
└── README.md
```

> **Note:** The live deployment on PythonAnywhere uses `flask_app.py` (via WSGI), which contains the same logic as `app.py`.

---

## How It Works

1. The Flutter app sends a food item's details (name, weight, calories, category) to the API.
2. The trained Random Forest model predicts the portion size based on weight, calories, and encoded category.
3. The API returns the predicted portion size along with the model's confidence score.

---

## API Reference

### `POST /predict`

**Request body (JSON):**

| Field | Type | Description |
|---|---|---|
| `food_name` | string | Name of the dish (for display only, not used by the model) |
| `weight_grams` | number | Weight of the item in grams |
| `calories` | number | Calorie count of the item |
| `category` | string | One of: `Main Course`, `Appetizer`, `Dessert`, `Beverage` |

**Example request:**
```json
{
  "food_name": "Chicken Biryani",
  "weight_grams": 350,
  "calories": 480,
  "category": "Main Course"
}
```

**Example response:**
```json
{
  "food": "Chicken Biryani",
  "predicted_portion": "Medium",
  "confidence": "100.0%"
}
```

**Error response** (e.g. missing field or invalid category):
```json
{
  "error": "Missing fields: ['category']"
}
```

### `GET /`

Health-check endpoint — confirms the API is live and lists valid categories.

```json
{
  "status": "running",
  "message": "Portion Size Prediction API is live",
  "valid_categories": ["Appetizer", "Beverage", "Dessert", "Main Course"]
}
```

---

## Retraining the Model

To retrain the model (e.g. with an updated dataset), run:

```bash
python3 train_model.py
```

This regenerates `portion_model.pkl`, `category_encoder.pkl`, `confusion_matrix.png`, and `feature_importance.png`, using whatever `restaurant_food_portion_dataset.csv` is present in the same folder.

> ⚠️ Train and deploy in the **same environment** (i.e. directly on the deployment server) to avoid `numpy`/`scikit-learn` version mismatches between the training machine and the server.

---

## Flutter Integration

A ready-to-use Dart service class (`portion_prediction_service.dart`) is available for calling this API from a Flutter app. Add the `http` package to `pubspec.yaml`, drop the file into `lib/services/`, and call:

```dart
final result = await PortionPredictionService.predictPortion(
  foodName: "Chicken Biryani",
  weightGrams: 350,
  calories: 480,
  category: "Main Course",
);
```

---

## Tech Stack

- **Python 3.10**
- **Flask** + **Flask-CORS** — REST API
- **scikit-learn** — Random Forest Classifier
- **pandas** — data handling
- **joblib** — model persistence
- **PythonAnywhere** — hosting

---

## Author

Malaika Shoukat — Food AR Final Year Project
