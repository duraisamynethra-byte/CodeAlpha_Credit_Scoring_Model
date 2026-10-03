# Credit Scoring Model

A Machine Learning project that predicts the likelihood of serious credit delinquency using customer financial information.

## Project Objective

The objective of this project is to build a credit scoring model that can identify customers who may be at risk of serious delinquency within two years.

## Dataset

The project uses the Give Me Some Credit dataset.

The dataset contains customer financial information such as:

- Age
- Monthly Income
- Debt Ratio
- Revolving Credit Utilization
- Number of Open Credit Lines
- Number of Real Estate Loans
- Number of Dependents
- Past Due Payment History

The dataset is stored locally in the `data/` folder and is not uploaded to GitHub.

## Data Preprocessing

The following preprocessing steps were performed:

- Removed the unnecessary index column
- Filled missing Monthly Income values using the median
- Filled missing Number of Dependents values using the median
- Split the dataset into training and testing data

## Feature Engineering

Two additional features were created:

- `TotalPastDue` – total number of past-due occurrences
- `TotalCreditLines` – total number of open credit and real-estate loan lines

## Machine Learning Models

The following models were trained and compared:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Tuned Random Forest

## Model Evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix
- ROC Curve

### Model Comparison

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7762 | 0.1815 | 0.6693 | 0.2856 | 0.8021 |
| Decision Tree | 0.9021 | 0.2633 | 0.2584 | 0.2608 | 0.6033 |
| Random Forest | 0.9356 | 0.5658 | 0.1566 | 0.2453 | 0.8365 |
| Tuned Random Forest | 0.8289 | 0.2411 | 0.7262 | 0.3620 | 0.8657 |

## Prediction System

The trained Tuned Random Forest model is saved locally as:

`credit_scoring_model.pkl`

The `predict.py` program loads the saved model and accepts customer details from the user.

It provides:

- Predicted delinquency class
- Risk probability

Example:

```text
Prediction: No Serious Delinquency Risk
Risk Probability: 23.17%