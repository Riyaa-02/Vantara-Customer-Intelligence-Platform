# ==========================================================
# Day 7 - Customer Churn Prediction
# ==========================================================

import os
import warnings
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)

warnings.filterwarnings("ignore")


# ==========================================================
# Create Required Directories
# ==========================================================

os.makedirs("outputs", exist_ok=True)
os.makedirs("outputs/plots", exist_ok=True)
os.makedirs("outputs/reports", exist_ok=True)


# ==========================================================
# Load Processed Datasets
# ==========================================================

print("\nLoading processed datasets...")

X_train = pd.read_csv("data/final/X_train.csv")
X_validation = pd.read_csv("data/final/X_validation.csv")
X_test = pd.read_csv("data/final/X_test.csv")

y_train = pd.read_csv("data/final/y_train.csv").squeeze()
y_validation = pd.read_csv("data/final/y_validation.csv").squeeze()
y_test = pd.read_csv("data/final/y_test.csv").squeeze()

print("Training Set   :", X_train.shape)
print("Validation Set :", X_validation.shape)
print("Testing Set    :", X_test.shape)


# ==========================================================
# Validate Dataset
# ==========================================================

print("\nValidating dataset...")

if X_train.empty:
    raise ValueError("Training dataset is empty.")

if y_train.empty:
    raise ValueError("Training labels are empty.")

if len(X_train) != len(y_train):
    raise ValueError("Mismatch between training features and labels.")

print("Dataset validation completed successfully.")


# ==========================================================
# Check Churn Distribution
# ==========================================================

print("\nChurn Distribution (Training Set)")
print(y_train.value_counts())

print("\nPercentage Distribution")
print((y_train.value_counts(normalize=True) * 100).round(2))

# ==========================================================
# Train Random Forest Model
# ==========================================================

print("\nTraining Random Forest Classifier...")

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

print("Model training completed successfully.")

# ==========================================================
# Model Evaluation - Validation Set
# ==========================================================

print("\nEvaluating model on validation dataset...")

# Predict class labels
y_val_pred = rf_model.predict(X_validation)

# Predict probabilities
y_val_prob = rf_model.predict_proba(X_validation)[:, 1]

# Calculate evaluation metrics
accuracy = accuracy_score(y_validation, y_val_pred)
precision = precision_score(y_validation, y_val_pred)
recall = recall_score(y_validation, y_val_pred)
f1 = f1_score(y_validation, y_val_pred)
roc_auc = roc_auc_score(y_validation, y_val_prob)

print("\nValidation Results")
print("-" * 40)
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")

print("\nClassification Report")
print(classification_report(y_validation, y_val_pred))

# ==========================================================
# Confusion Matrix
# ==========================================================

cm = confusion_matrix(y_validation, y_val_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Non-Churn", "Churn"]
)

disp.plot(
    cmap="Blues",
    values_format="d",
    colorbar=False
)
plt.title(
    "Validation Confusion Matrix",
    fontsize=14,
    fontweight="bold"
)
plt.tight_layout()

plt.savefig(
    "outputs/plots/validation_confusion_matrix.png",
    dpi=300
)

plt.show()

# ==========================================================
# ROC Curve
# ==========================================================

RocCurveDisplay.from_predictions(
    y_validation,
    y_val_prob
)

plt.title(
    "Validation ROC Curve",
    fontsize=14,
    fontweight="bold"
)

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    "outputs/plots/validation_roc_curve.png",
    dpi=300
)

plt.show()

# ==========================================================
# Feature Importance
# ==========================================================

print("\nGenerating Feature Importance...")

feature_importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": rf_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 10 Important Features")
print(feature_importance.head(10))


plt.figure(figsize=(10,6))

plt.barh(
    feature_importance["Feature"][:10][::-1],
    feature_importance["Importance"][:10][::-1]
)

plt.xlabel("Importance")
plt.ylabel("Feature")

plt.title(
    "Top 10 Feature Importance",
    fontsize=14,
    fontweight="bold"
)

plt.tight_layout()

plt.savefig(
    "outputs/plots/feature_importance.png",
    dpi=300
)

plt.show()
feature_importance.to_csv(
    "outputs/reports/feature_importance.csv",
    index=False
)
# ==========================================================
# Model Evaluation - Test Dataset
# ==========================================================

print("\nEvaluating model on test dataset...")

y_test_pred = rf_model.predict(X_test)
y_test_prob = rf_model.predict_proba(X_test)[:, 1]

test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)
test_roc_auc = roc_auc_score(y_test, y_test_prob)

print("\nTest Results")
print("-" * 40)
print(f"Accuracy : {test_accuracy:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall   : {test_recall:.4f}")
print(f"F1 Score : {test_f1:.4f}")
print(f"ROC-AUC  : {test_roc_auc:.4f}")

print("\nClassification Report")
print(classification_report(y_test, y_test_pred))

# ==========================================================
# Save Trained Model
# ==========================================================

print("\nSaving trained model...")

joblib.dump(
    rf_model,
    "models/customer_churn_model.pkl"
)

print("Model saved successfully.")

print("\nDay 7 completed successfully.")