import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

print("Machine Learning environment is ready!")
print("Pandas version:", pd.__version__)
print("NumPy version:", np.__version__)
print("Scikit-learn version:", sklearn.__version__)
import pandas as pd

# Load the dataset
data = pd.read_csv("data/cs-training.csv")

# Display basic information
print("Dataset loaded successfully!")
print("Shape of dataset:", data.shape)

# Display column names
print("\nColumn names:")
print(data.columns.tolist())

# Display first 5 rows
print("\nFirst 5 rows:")
print(data.head())
import pandas as pd

# Load the dataset
data = pd.read_csv("data/cs-training.csv")

# Display basic information
print("Dataset loaded successfully!")
print("Shape of dataset:", data.shape)

# Display column names
print("\nColumn names:")
print(data.columns.tolist())

# Display first 5 rows
print("\nFirst 5 rows:")
print(data.head())
# Check data types
print("\nData types:")
print(data.dtypes)

# Check missing values
print("\nMissing values:")
print(data.isnull().sum())

# Check target distribution
print("\nTarget distribution:")
print(data["SeriousDlqin2yrs"].value_counts())

# Check target percentages
print("\nTarget percentage:")
print(data["SeriousDlqin2yrs"].value_counts(normalize=True) * 100)