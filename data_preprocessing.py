import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
import pickle

# Define paths
data_path = 'data/example_test.csv'
processed_dir = 'data/processed'
os.makedirs(processed_dir, exist_ok=True)

# Load the dataset
print(f"Loading dataset from {data_path}...")
df = pd.read_csv(data_path)

# Extract features and other columns
print("Extracting features and metadata...")
feature_cols = [col for col in df.columns if col.startswith('feature')]
metadata_cols = ['date', 'ts_id', 'weight']

# Since this is test data, we don't have target values
# In a real scenario, we would have something like:
# target_col = 'resp' or 'target'

# Handle missing values
print("Handling missing values...")
features_df = df[feature_cols]
imputer = SimpleImputer(strategy='mean')
features_imputed = imputer.fit_transform(features_df)
features_df_imputed = pd.DataFrame(features_imputed, columns=feature_cols)

# Standardize the features
print("Standardizing features...")
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_df_imputed)
features_df_scaled = pd.DataFrame(features_scaled, columns=feature_cols)

# Add back metadata columns
for col in metadata_cols:
    if col in df.columns:
        features_df_scaled[col] = df[col].values

# Split into train and validation sets (80/20)
# In a real scenario with target values, we would do:
# X_train, X_val, y_train, y_val = train_test_split(features_df_scaled, targets, test_size=0.2, random_state=42)

# Since we don't have target values in this example dataset, we'll just split the features
print("Splitting into train and validation sets...")
train_df, val_df = train_test_split(features_df_scaled, test_size=0.2, random_state=42)

# Save the processed datasets
print("Saving processed datasets...")
train_df.to_csv(os.path.join(processed_dir, 'train_features.csv'), index=False)
val_df.to_csv(os.path.join(processed_dir, 'val_features.csv'), index=False)

# Save the preprocessing objects for later use
print("Saving preprocessing objects...")
with open(os.path.join(processed_dir, 'imputer.pkl'), 'wb') as f:
    pickle.dump(imputer, f)
    
with open(os.path.join(processed_dir, 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)

print("\nPreprocessing completed. Files saved to", processed_dir)
print(f"Train set shape: {train_df.shape}")
print(f"Validation set shape: {val_df.shape}") 