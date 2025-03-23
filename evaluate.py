import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from model import get_model
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Jane Street Market Prediction model')
    parser.add_argument('--model_type', type=str, default='mlp', choices=['mlp', 'transformer'],
                        help='Type of model to evaluate')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to the model to evaluate. If not provided, will use the best model from the output directory.')
    parser.add_argument('--input_dim', type=int, default=130,
                        help='Number of feature dimensions')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch size for evaluation')
    parser.add_argument('--data_dir', type=str, default='data/processed',
                        help='Directory containing processed data')
    parser.add_argument('--output_dir', type=str, default='models',
                        help='Directory containing model and to save evaluation results')
    return parser.parse_args()


def load_data(data_path, feature_cols=None):
    """
    Load data for evaluation
    """
    # Load the dataset
    data_df = pd.read_csv(data_path)
    
    # Create synthetic targets for demonstration purposes
    # In a real scenario, these would be actual target values
    if feature_cols is None:
        feature_cols = [col for col in data_df.columns if col.startswith('feature')]
    
    targets = (data_df[feature_cols].mean(axis=1) > 0).astype(float).values
    
    # Convert to PyTorch tensors
    X = torch.tensor(data_df[feature_cols].values, dtype=torch.float32)
    y = torch.tensor(targets, dtype=torch.float32).reshape(-1, 1)
    
    return X, y, data_df, feature_cols


def evaluate_model(model, data_loader, criterion, device):
    """
    Evaluate the model
    """
    model.eval()
    eval_loss = 0
    all_targets = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for data, targets in tqdm(data_loader, desc="Evaluating"):
            # Move data to device
            data, targets = data.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            # Collect statistics
            eval_loss += loss.item() * data.size(0)
            probs = outputs.sigmoid().detach().cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend((probs > 0.5).astype(int))
            all_targets.extend(targets.cpu().numpy())
    
    eval_loss /= len(data_loader.dataset)
    all_preds = np.array(all_preds).flatten()
    all_probs = np.array(all_probs).flatten()
    all_targets = np.array(all_targets).flatten()
    
    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_preds)
    auc = roc_auc_score(all_targets, all_probs)
    precision = precision_score(all_targets, all_preds)
    recall = recall_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds)
    conf_matrix = confusion_matrix(all_targets, all_preds)
    
    return {
        'loss': eval_loss,
        'accuracy': accuracy,
        'auc': auc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': conf_matrix,
        'predictions': all_preds,
        'probabilities': all_probs,
        'targets': all_targets
    }


def plot_confusion_matrix(conf_matrix, output_path):
    """
    Plot confusion matrix
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_roc_curve(targets, probabilities, output_path):
    """
    Plot ROC curve
    """
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(targets, probabilities)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_precision_recall_curve(targets, probabilities, output_path):
    """
    Plot precision-recall curve
    """
    from sklearn.metrics import precision_recall_curve, average_precision_score
    precision, recall, _ = precision_recall_curve(targets, probabilities)
    avg_precision = average_precision_score(targets, probabilities)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {avg_precision:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    # Parse arguments
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    eval_dir = os.path.join(args.output_dir, 'evaluation')
    os.makedirs(eval_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load validation data
    val_path = os.path.join(args.data_dir, 'val_features.csv')
    X_val, y_val, val_df, feature_cols = load_data(val_path)
    
    # Create dataloader
    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    # Update input dimension based on actual feature count
    args.input_dim = len(feature_cols)
    
    # Initialize model
    model_kwargs = {
        'input_dim': args.input_dim
    }
    model = get_model(model_type=args.model_type, **model_kwargs)
    
    # Load model weights
    if args.model_path is None:
        model_path = os.path.join(args.output_dir, f'best_{args.model_type}_model.pt')
    else:
        model_path = args.model_path
    
    print(f"Loading model from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    
    # Define loss function
    criterion = nn.BCEWithLogitsLoss()
    
    # Evaluate model
    print(f"Evaluating {args.model_type.upper()} model...")
    metrics = evaluate_model(model, val_loader, criterion, device)
    
    # Print metrics
    print("\nEvaluation Metrics:")
    print(f"Loss: {metrics['loss']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"AUC: {metrics['auc']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"Confusion Matrix:\n{metrics['confusion_matrix']}")
    
    # Plot confusion matrix
    conf_matrix_path = os.path.join(eval_dir, 'confusion_matrix.png')
    plot_confusion_matrix(metrics['confusion_matrix'], conf_matrix_path)
    print(f"Confusion matrix saved to {conf_matrix_path}")
    
    # Plot ROC curve
    roc_curve_path = os.path.join(eval_dir, 'roc_curve.png')
    plot_roc_curve(metrics['targets'], metrics['probabilities'], roc_curve_path)
    print(f"ROC curve saved to {roc_curve_path}")
    
    # Plot precision-recall curve
    pr_curve_path = os.path.join(eval_dir, 'precision_recall_curve.png')
    plot_precision_recall_curve(metrics['targets'], metrics['probabilities'], pr_curve_path)
    print(f"Precision-recall curve saved to {pr_curve_path}")
    
    # Save evaluation metrics
    metrics_df = pd.DataFrame({
        'Metric': ['Loss', 'Accuracy', 'AUC', 'Precision', 'Recall', 'F1 Score'],
        'Value': [metrics['loss'], metrics['accuracy'], metrics['auc'], metrics['precision'], metrics['recall'], metrics['f1']]
    })
    metrics_path = os.path.join(eval_dir, 'metrics.csv')
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Evaluation metrics saved to {metrics_path}")
    
    # Save predictions
    preds_df = pd.DataFrame({
        'Target': metrics['targets'],
        'Prediction': metrics['predictions'],
        'Probability': metrics['probabilities']
    })
    preds_path = os.path.join(eval_dir, 'predictions.csv')
    preds_df.to_csv(preds_path, index=False)
    print(f"Predictions saved to {preds_path}")


if __name__ == '__main__':
    main() 