"""
Random Forest Classifier - Restaurant Food Portion Prediction
Predicts Portion_Label (Small/Medium/Large) from Weight_grams, Calories, and Category.
"""

import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load data
df = pd.read_csv("restaurant_food_portion_dataset.csv")

# 2. Encode categorical feature
le = LabelEncoder()
df['Category_Encoded'] = le.fit_transform(df['Category'])

# 3. Features and target
X = df[['Weight_grams', 'Calories', 'Category_Encoded']]
y = df['Portion_Label']

# 4. Train/test split (stratified to preserve class balance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. Evaluate
pred = model.predict(X_test)

print("=" * 50)
print("RANDOM FOREST MODEL RESULTS")
print("=" * 50)
print(f"Accuracy: {accuracy_score(y_test, pred):.4f}\n")

print("Classification Report:")
print(classification_report(y_test, pred))

print("Confusion Matrix:")
print(pd.DataFrame(
    confusion_matrix(y_test, pred, labels=model.classes_),
    index=[f"Actual_{c}" for c in model.classes_],
    columns=[f"Pred_{c}" for c in model.classes_]
))

print("\nFeature Importances:")
for feat, imp in sorted(zip(X.columns, model.feature_importances_), key=lambda x: -x[1]):
    print(f"  {feat}: {imp:.4f}")

# 7. Save the trained model and label encoder for later use
joblib.dump(model, "portion_rf_model.pkl")
joblib.dump(le, "category_label_encoder.pkl")
print("\nModel saved as 'portion_rf_model.pkl'")
print("Label encoder saved as 'category_label_encoder.pkl'")
