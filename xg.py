import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, 
    confusion_matrix, log_loss, balanced_accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns


Xtrain_data = pd.read_csv('X_Train_Data_Input.csv')
Ytrain_data = pd.read_csv('Y_Train_Data_Target.csv')
Xtest_data = pd.read_csv('X_Test_Data_Input.csv')
Ytest_data = pd.read_csv('Y_Test_Data_Target.csv')


X_train = Xtrain_data.iloc[:, 1:-1].values
y_train = Ytrain_data.iloc[:, -1].values
X_test = Xtest_data.iloc[:, 1:-1].values
y_test = Ytest_data.iloc[:, -1].values


xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    missing=np.nan
)


xgb_model.fit(X_train, y_train)


y_pred = xgb_model.predict(X_test)
y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
balanced_acc = balanced_accuracy_score(y_test, y_pred)
logloss_value = log_loss(y_test, y_pred_proba)


fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)


cm = confusion_matrix(y_test, y_pred)


print(f"XGBoost Test Accuracy: {accuracy:.4f}")
print(f"XGBoost Test Precision: {precision:.4f}")
print(f"XGBoost Test Recall: {recall:.4f}")
print(f"XGBoost Test F1 Score: {f1:.4f}")
print(f"XGBoost Test Balanced Accuracy: {balanced_acc:.4f}")
print(f"XGBoost Test Log Loss: {logloss_value:.4f}")
print(f"XGBoost Test AUC-ROC: {roc_auc:.4f}")


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


plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Confusion Matrix")
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
