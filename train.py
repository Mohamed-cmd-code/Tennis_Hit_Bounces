from src.dataset import build_dataset
from src.model import build_model

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import joblib
import os

DATA_FOLDER = "data"
MODEL_PATH = "models/hit_bounce_rf.pkl"

print("Loading dataset...")
X, y = build_dataset(DATA_FOLDER)


X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training model...")
model = build_model()
model.fit(X_train, y_train)

# =========================
# testing on training data 
# =========================
train_preds = model.predict(X_train)
print("\n--- Training Metrics ---")
print("Accuracy:", accuracy_score(y_train, train_preds))
print(classification_report(y_train, train_preds))

# =========================
# testing on validation data
# =========================
val_preds = model.predict(X_val)
print("\n--- Validation Metrics ---")
print("Accuracy:", accuracy_score(y_val, val_preds))
print(classification_report(y_val, val_preds))

print("\nConfusion Matrix (Validation):")
print(confusion_matrix(y_val, val_preds))

# save the model
os.makedirs("models", exist_ok=True)
joblib.dump(model, MODEL_PATH)
print(f"\nModel saved to {MODEL_PATH}")
