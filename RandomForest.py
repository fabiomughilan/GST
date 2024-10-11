import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, 
    confusion_matrix, log_loss, balanced_accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns

# Load the datasets from CSV files
Xtrain_data = pd.read_csv('X_Train_Data_Input.csv')
Ytrain_data = pd.read_csv('Y_Train_Data_Target.csv')
Xtest_data = pd.read_csv('X_Test_Data_Input.csv')
Ytest_data = pd.read_csv('Y_Test_Data_Target.csv')

# Extract features and target labels
X_train = Xtrain_data.iloc[:, 1:-1].values
y_train = Ytrain_data.iloc[:, -1].values
X_test = Xtest_data.iloc[:, 1:-1].values
y_test = Ytest_data.iloc[:, -1].values

# Initialize Random Forest Classifier
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,         # You can adjust the max depth based on your dataset
    random_state=42,
    class_weight='balanced',  # Handles imbalanced data
)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)
y_pred_proba = rf_model.predict_proba(X_test)[:, 1]

# Metrics calculations
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
balanced_acc = balanced_accuracy_score(y_test, y_pred)
logloss_value = log_loss(y_test, y_pred_proba)

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Print metrics
print(f"Random Forest Test Accuracy: {accuracy:.4f}")
print(f"Random Forest Test Precision: {precision:.4f}")
print(f"Random Forest Test Recall: {recall:.4f}")
print(f"Random Forest Test F1 Score: {f1:.4f}")
print(f"Random Forest Test Balanced Accuracy: {balanced_acc:.4f}")
print(f"Random Forest Test Log Loss: {logloss_value:.4f}")
print(f"Random Forest Test AUC-ROC: {roc_auc:.4f}")

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Diagonal line for random guessing
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Confusion Matrix Plot
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Confusion Matrix")
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
