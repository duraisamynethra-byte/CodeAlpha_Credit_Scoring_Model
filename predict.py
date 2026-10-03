import joblib
import pandas as pd

model = joblib.load("credit_scoring_model.pkl")
# Get customer details
age = float(input("Enter age: "))
monthly_income = float(input("Enter monthly income: "))
debt_ratio = float(input("Enter debt ratio: "))
revolving_utilization = float(input("Enter revolving utilization: "))
open_credit_lines = float(input("Enter number of open credit lines: "))
real_estate_loans = float(input("Enter number of real estate loans: "))
dependents = float(input("Enter number of dependents: "))
past_due_30_59 = float(input("Enter 30-59 days past due count: "))
past_due_60_89 = float(input("Enter 60-89 days past due count: "))
past_due_90 = float(input("Enter 90+ days late count: "))

# Create engineered features
total_past_due = past_due_30_59 + past_due_60_89 + past_due_90
total_credit_lines = open_credit_lines + real_estate_loans
# Create input data for the model
input_data = pd.DataFrame([[
    revolving_utilization,
    age,
    past_due_30_59,
    debt_ratio,
    monthly_income,
    open_credit_lines,
    past_due_90,
    real_estate_loans,
    past_due_60_89,
    dependents,
    total_past_due,
    total_credit_lines
]], columns=[
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
    "TotalPastDue",
    "TotalCreditLines"
])

# Make prediction
prediction = model.predict(input_data)[0]
risk_probability = model.predict_proba(input_data)[0][1] * 100

if prediction == 1:
    print("\nPrediction: Serious Delinquency Risk")
else:
    print("\nPrediction: No Serious Delinquency Risk")

print(f"Risk Probability: {risk_probability:.2f}%")