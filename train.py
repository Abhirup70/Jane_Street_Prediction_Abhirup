import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model import get_model
from sklearn.metrics import accuracy_score, roc_auc_score
import time
import matplotlib.pyplot as plt
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Train Jane Street Market Prediction model')
    parser.add_argument('--model_type', type=str, default='mlp', choices=['mlp', 'transformer'],
                        help='Type of model to train')
    parser.add_argument('--input_dim', type=int, default=130,
                        help='Number of feature dimensions')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay (L2 regularization)')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate')
    parser.add_argument('--data_dir', type=str, default='data/processed',
                        help='Directory containing processed data')
    parser.add_argument('--output_dir', type=str, default='models',
                        help='Directory to save model and results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    return parser.parse_args()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data(data_dir, feature_cols=None):
    """
    Load processed data for training
    """
    train_path = os.path.join(data_dir, 'train_features.csv')
    val_path = os.path.join(data_dir, 'val_features.csv')
    
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    
    # For this example, we'll create synthetic targets since our dataset doesn't have real targets
    # This is just for demonstration purposes; in a real scenario, you would use actual targets
    print("Creating synthetic targets for demonstration...")
    
    # Extract feature columns
    if feature_cols is None:
        feature_cols = [col for col in train_df.columns if col.startswith('feature')]
    
    # Create synthetic targets based on a simple rule
    # In a real scenario, these would be actual target values
    train_targets = (train_df[feature_cols].mean(axis=1) > 0).astype(float).values
    val_targets = (val_df[feature_cols].mean(axis=1) > 0).astype(float).values
    
    # Convert to PyTorch tensors
    X_train = torch.tensor(train_df[feature_cols].values, dtype=torch.float32)
    y_train = torch.tensor(train_targets, dtype=torch.float32).reshape(-1, 1)
    X_val = torch.tensor(val_df[feature_cols].values, dtype=torch.float32)
    y_val = torch.tensor(val_targets, dtype=torch.float32).reshape(-1, 1)
    
    print(f"Train data: {X_train.shape}, train targets: {y_train.shape}")
    print(f"Validation data: {X_val.shape}, validation targets: {y_val.shape}")
    
    return X_train, y_train, X_val, y_val, feature_cols


def create_dataloaders(X_train, y_train, X_val, y_val, batch_size):
    """
    Create DataLoader objects for training and validation data
    """
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    return train_loader, val_loader


def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train the model for one epoch
    """
    model.train()
    train_loss = 0
    all_targets = []
    all_preds = []
    
    for data, targets in tqdm(train_loader, desc="Training"):
        # Move data to device
        data, targets = data.to(device), targets.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # Collect statistics
        train_loss += loss.item() * data.size(0)
        all_preds.extend(outputs.sigmoid().detach().cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
    
    train_loss /= len(train_loader.dataset)
    all_preds = np.array(all_preds).flatten()
    all_targets = np.array(all_targets).flatten()
    
    # Calculate binary predictions using threshold 0.5
    binary_preds = (all_preds > 0.5).astype(int)
    
    # Calculate accuracy and AUC
    accuracy = accuracy_score(all_targets, binary_preds)
    try:
        auc = roc_auc_score(all_targets, all_preds)
    except:
        auc = 0.5  # Default value if AUC calculation fails
    
    return train_loss, accuracy, auc


def validate(model, val_loader, criterion, device):
    """
    Validate the model
    """
    model.eval()
    val_loss = 0
    all_targets = []
    all_preds = []
    
    with torch.no_grad():
        for data, targets in tqdm(val_loader, desc="Validation"):
            # Move data to device
            data, targets = data.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            # Collect statistics
            val_loss += loss.item() * data.size(0)
            all_preds.extend(outputs.sigmoid().detach().cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    val_loss /= len(val_loader.dataset)
    all_preds = np.array(all_preds).flatten()
    all_targets = np.array(all_targets).flatten()
    
    # Calculate binary predictions using threshold 0.5
    binary_preds = (all_preds > 0.5).astype(int)
    
    # Calculate accuracy and AUC
    accuracy = accuracy_score(all_targets, binary_preds)
    try:
        auc = roc_auc_score(all_targets, all_preds)
    except:
        auc = 0.5  # Default value if AUC calculation fails
    
    return val_loss, accuracy, auc


def save_model(model, model_path):
    """
    Save the trained model
    """
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


def plot_training_history(history, output_dir):
    """
    Plot training and validation loss, accuracy, and AUC
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot loss
    ax1.plot(epochs, history['train_loss'], label='Train')
    ax1.plot(epochs, history['val_loss'], label='Validation')
    ax1.set_title('Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Plot accuracy
    ax2.plot(epochs, history['train_acc'], label='Train')
    ax2.plot(epochs, history['val_acc'], label='Validation')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    # Plot AUC
    ax3.plot(epochs, history['train_auc'], label='Train')
    ax3.plot(epochs, history['val_auc'], label='Validation')
    ax3.set_title('AUC')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('AUC')
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    print(f"Training history plot saved to {os.path.join(output_dir, 'training_history.png')}")


def main():
    # Parse arguments
    args = parse_args()
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    X_train, y_train, X_val, y_val, feature_cols = load_data(args.data_dir)
    
    # Update input dimension based on actual feature count
    args.input_dim = len(feature_cols)
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(X_train, y_train, X_val, y_val, args.batch_size)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model
    model_kwargs = {
        'input_dim': args.input_dim,
        'dropout_rate': args.dropout if args.model_type == 'mlp' else args.dropout
    }
    model = get_model(model_type=args.model_type, **model_kwargs)
    model = model.to(device)
    print(model)
    
    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Train the model
    print(f"Training {args.model_type.upper()} model for {args.epochs} epochs...")
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'train_auc': [],
        'val_auc': []
    }
    
    best_val_loss = float('inf')
    best_model_path = os.path.join(args.output_dir, f'best_{args.model_type}_model.pt')
    
    start_time = time.time()
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Train and validate
        train_loss, train_acc, train_auc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_auc = validate(model, val_loader, criterion, device)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_auc'].append(train_auc)
        history['val_auc'].append(val_auc)
        
        # Print statistics
        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, AUC: {train_auc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, AUC: {val_auc:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, best_model_path)
            print(f"New best model saved with validation loss: {val_loss:.4f}")
    
    # Training finished
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, f'final_{args.model_type}_model.pt')
    save_model(model, final_model_path)
    
    # Plot training history
    plot_training_history(history, args.output_dir)
    
    # Save training history
    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(args.output_dir, 'training_history.csv'), index=False)


if __name__ == '__main__':
    main() 