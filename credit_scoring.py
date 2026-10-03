import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

print("Machine Learning environment is ready!")
print("Pandas version:", pd.__version__)
print("NumPy version:", np.__version__)
print("Scikit-learn version:", sklearn.__version__)

# Load the dataset
data = pd.read_csv("data/cs-training.csv")

# Remove unnecessary index column
data = data.drop("Unnamed: 0", axis=1)

# Fill missing values
data["MonthlyIncome"] = data["MonthlyIncome"].fillna(
    data["MonthlyIncome"].median()
)

data["NumberOfDependents"] = data["NumberOfDependents"].fillna(
    data["NumberOfDependents"].median()
)

print("\nDataset loaded and cleaned successfully!")

print("\nShape of dataset:")
print(data.shape)

print("\nMissing values after cleaning:")
print(data.isnull().sum())

# Feature Engineering

# Create a feature for total past-due occurrences
data["TotalPastDue"] = (
    data["NumberOfTime30-59DaysPastDueNotWorse"]
    + data["NumberOfTime60-89DaysPastDueNotWorse"]
    + data["NumberOfTimes90DaysLate"]
)

# Create a feature for total number of credit lines
data["TotalCreditLines"] = (
    data["NumberOfOpenCreditLinesAndLoans"]
    + data["NumberRealEstateLoansOrLines"]
)

# Separate features and target
X = data.drop("SeriousDlqin2yrs", axis=1)
y = data["SeriousDlqin2yrs"]

print("\nFeature engineering completed!")

print("\nNew features:")
print(["TotalPastDue", "TotalCreditLines"])

print("\nNew features shape:")
print(X.shape)

print("\nTarget shape:")
print(y.shape)


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining data shape:")
print(X_train.shape)

print("\nTesting data shape:")
print(X_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())


# Scale the features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create Logistic Regression model
logistic_model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=42
)

# Train the model
logistic_model.fit(X_train_scaled, y_train)

print("\nLogistic Regression model trained successfully!")

# Make predictions on the test data
y_pred = logistic_model.predict(X_test_scaled)

print("\nPredictions made successfully!")
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\nAccuracy:", accuracy)

print("\nConfusion Matrix:")
print(cm)
# Calculate Precision, Recall and F1-Score
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nPrecision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)
# Calculate ROC-AUC
y_prob = logistic_model.predict_proba(X_test_scaled)[:, 1]

roc_auc = roc_auc_score(y_test, y_prob)

print("ROC-AUC:", roc_auc)
# Create ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label="Logistic Regression")
plt.plot([0, 1], [0, 1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Logistic Regression")
plt.legend()
plt.show()
# Create Decision Tree model
decision_tree = DecisionTreeClassifier(
    class_weight="balanced",
    random_state=42
)

# Train the model
decision_tree.fit(X_train, y_train)

print("\nDecision Tree model trained successfully!")
# Make predictions using Decision Tree
dt_pred = decision_tree.predict(X_test)

print("\nDecision Tree predictions made successfully!")
# Evaluate Decision Tree
dt_accuracy = accuracy_score(y_test, dt_pred)
dt_precision = precision_score(y_test, dt_pred)
dt_recall = recall_score(y_test, dt_pred)
dt_f1 = f1_score(y_test, dt_pred)

print("\nDecision Tree Accuracy:", dt_accuracy)
print("Decision Tree Precision:", dt_precision)
print("Decision Tree Recall:", dt_recall)
print("Decision Tree F1-Score:", dt_f1)
# Calculate Decision Tree ROC-AUC
dt_prob = decision_tree.predict_proba(X_test)[:, 1]

dt_roc_auc = roc_auc_score(y_test, dt_prob)

print("Decision Tree ROC-AUC:", dt_roc_auc)
# Create Random Forest model
random_forest = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
# Train the model
random_forest.fit(X_train, y_train)

print("\nRandom Forest model trained successfully!")
# Make predictions using Random Forest
rf_pred = random_forest.predict(X_test)

print("\nRandom Forest predictions made successfully!")
# Evaluate Random Forest
rf_accuracy = accuracy_score(y_test, rf_pred)
rf_precision = precision_score(y_test, rf_pred)
rf_recall = recall_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)

print("\nRandom Forest Accuracy:", rf_accuracy)
print("Random Forest Precision:", rf_precision)
print("Random Forest Recall:", rf_recall)
print("Random Forest F1-Score:", rf_f1)
# Calculate Random Forest ROC-AUC
rf_prob = random_forest.predict_proba(X_test)[:, 1]

rf_roc_auc = roc_auc_score(y_test, rf_prob)

print("Random Forest ROC-AUC:", rf_roc_auc)
# Compare all models
comparison = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest"
    ],
    "Accuracy": [
        accuracy,
        dt_accuracy,
        rf_accuracy
    ],
    "Precision": [
        precision,
        dt_precision,
        rf_precision
    ],
    "Recall": [
        recall,
        dt_recall,
        rf_recall
    ],
    "F1-Score": [
        f1,
        dt_f1,
        rf_f1
    ],
    "ROC-AUC": [
        roc_auc,
        dt_roc_auc,
        rf_roc_auc
    ]
})
print("\nDetailed Model Comparison:")
print(comparison.to_string(index=False))
# Tuned Random Forest
tuned_rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

tuned_rf.fit(X_train, y_train)

print("\nTuned Random Forest model trained successfully!")

tuned_rf_pred = tuned_rf.predict(X_test)

tuned_rf_accuracy = accuracy_score(y_test, tuned_rf_pred)
tuned_rf_precision = precision_score(y_test, tuned_rf_pred)
tuned_rf_recall = recall_score(y_test, tuned_rf_pred)
tuned_rf_f1 = f1_score(y_test, tuned_rf_pred)

tuned_rf_prob = tuned_rf.predict_proba(X_test)[:, 1]
tuned_rf_roc_auc = roc_auc_score(y_test, tuned_rf_prob)

print("\nTuned Random Forest Accuracy:", tuned_rf_accuracy)
print("Tuned Random Forest Precision:", tuned_rf_precision)
print("Tuned Random Forest Recall:", tuned_rf_recall)
print("Tuned Random Forest F1-Score:", tuned_rf_f1)
print("Tuned Random Forest ROC-AUC:", tuned_rf_roc_auc)

# Add Tuned Random Forest to comparison
tuned_comparison = pd.DataFrame({
    "Model": ["Tuned Random Forest"],
    "Accuracy": [tuned_rf_accuracy],
    "Precision": [tuned_rf_precision],
    "Recall": [tuned_rf_recall],
    "F1-Score": [tuned_rf_f1],
    "ROC-AUC": [tuned_rf_roc_auc]
})

comparison = pd.concat([comparison, tuned_comparison], ignore_index=True)

print("\nUpdated Model Comparison:")
print(comparison.to_string(index=False))
# Confusion Matrix for Tuned Random Forest
tuned_cm = confusion_matrix(y_test, tuned_rf_pred)

print("\nTuned Random Forest Confusion Matrix:")
print(tuned_cm)

plt.figure(figsize=(6, 5))
sns.heatmap(
    tuned_cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Delinquency", "Delinquency"],
    yticklabels=["No Delinquency", "Delinquency"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Tuned Random Forest")
plt.show()
# ROC Curve Comparison

plt.figure(figsize=(8, 6))

fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob)
fpr_dt, tpr_dt, _ = roc_curve(y_test, dt_prob)
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_prob)
fpr_tuned, tpr_tuned, _ = roc_curve(y_test, tuned_rf_prob)

plt.plot(fpr_lr, tpr_lr, label="Logistic Regression")
plt.plot(fpr_dt, tpr_dt, label="Decision Tree")
plt.plot(fpr_rf, tpr_rf, label="Random Forest")
plt.plot(fpr_tuned, tpr_tuned, label="Tuned Random Forest")

plt.plot([0, 1], [0, 1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.show()
# Save the trained model
joblib.dump(tuned_rf, "credit_scoring_model.pkl")

print("\nTuned Random Forest model saved successfully!")