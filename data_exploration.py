import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Check if the data files exist
data_path = 'data/example_test.csv'
if not os.path.exists(data_path):
    print(f"Data file not found: {data_path}")
    exit(1)

# Load the dataset
print(f"Loading dataset from {data_path}...")
df = pd.read_csv(data_path)

# Display basic information
print("\n=== Dataset Information ===")
print(f"Dataset shape: {df.shape}")
print(f"Number of samples: {df.shape[0]}")
print(f"Number of features: {df.shape[1]}")

# Display the first few rows
print("\n=== First 5 rows of the dataset ===")
print(df.head())

# Check for missing values
print("\n=== Missing values ===")
missing_values = df.isnull().sum()
print(f"Total missing values: {missing_values.sum()}")
if missing_values.sum() > 0:
    print(missing_values[missing_values > 0])

# Display feature names
print("\n=== Feature names ===")
print(df.columns.tolist())

# Basic statistics of features
print("\n=== Basic statistics ===")
print(df.describe())

# Save dataset info to a text file
with open('data/dataset_info.txt', 'w') as f:
    f.write(f"Dataset shape: {df.shape}\n")
    f.write(f"Number of samples: {df.shape[0]}\n")
    f.write(f"Number of features: {df.shape[1]}\n\n")
    f.write("Feature names:\n")
    f.write(', '.join(df.columns.tolist()))
    f.write("\n\nBasic statistics:\n")
    f.write(df.describe().to_string())

print("\nExploration completed. Results saved to data/dataset_info.txt") 