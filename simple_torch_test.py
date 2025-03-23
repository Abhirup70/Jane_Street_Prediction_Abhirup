import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

# Create tensors
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([4.0, 5.0, 6.0])

# Basic operations
print("x =", x)
print("y =", y)
print("x + y =", x + y)
print("x * y =", x * y)

# Create a simple model
model = torch.nn.Sequential(
    torch.nn.Linear(3, 1),
    torch.nn.Sigmoid()
)

# Forward pass
output = model(x)
print("Model output:", output) 