import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

print("Python version:", sys.version)
print("NumPy version:", np.__version__)
print("PyTorch version:", torch.__version__)
print("Using device:", "cuda" if torch.cuda.is_available() else "cpu")

# Create synthetic data
print("\nCreating synthetic data...")
X = np.random.randn(1000, 10).astype(np.float32)
y = (np.sum(X[:, :3], axis=1) > 0).astype(np.float32)  # Simple rule: sum of first 3 features > 0

# Convert to PyTorch tensors
X_tensor = torch.FloatTensor(X)
y_tensor = torch.FloatTensor(y).view(-1, 1)

# Create dataset and dataloader
dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.layers(x)

# Initialize model
model = SimpleModel()
print("\nModel architecture:")
print(model)

# Train the model
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

print("\nTraining the model...")
n_epochs = 5
for epoch in range(n_epochs):
    running_loss = 0.0
    for inputs, targets in dataloader:
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    avg_loss = running_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}")

# Evaluate the model
print("\nEvaluating the model...")
model.eval()
with torch.no_grad():
    test_preds = model(X_tensor)
    test_preds_binary = (test_preds > 0.5).float()
    accuracy = (test_preds_binary == y_tensor).float().mean()
    print(f"Accuracy: {accuracy.item():.4f}")

print("\nTest completed successfully!") 