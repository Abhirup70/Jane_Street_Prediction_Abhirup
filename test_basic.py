import os
import sys
import numpy as np
import pandas as pd

print("Python version:", sys.version)
print("NumPy version:", np.__version__)
print("Pandas version:", pd.__version__)

# Create synthetic data
print("Creating synthetic data...")
features = np.random.randn(100, 5)
target = np.random.rand(100) > 0.5
df = pd.DataFrame(features, columns=[f'feature_{i}' for i in range(5)])
df['target'] = target

print("Data shape:", df.shape)
print("Data sample:")
print(df.head())

# Simple model with NumPy
print("\nTraining a simple model...")
weights = np.random.randn(5)
predictions = np.dot(features, weights) > 0
accuracy = np.mean(predictions == target)
print(f"Simple model accuracy: {accuracy:.4f}")

print("Test completed successfully!") 