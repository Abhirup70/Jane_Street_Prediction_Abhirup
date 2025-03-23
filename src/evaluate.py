import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_curve
import os
import argparse

def evaluate_model(model, test_df, data_loader, output_dir='models'):
    """
    Evaluate a model on test data
    """
    # Extract features, targets, and weights
    X_test, y_test, weights_test = data_loader.get_features_targets(test_df)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Convert target to binary classification (1 if positive return, 0 otherwise)
    y_test_binary = (y_test > 0).astype(int)
    
    # Convert predictions to binary actions
    y_pred_binary = (y_pred > 0.5).astype(int).flatten()
    
    # Calculate metrics
    accuracy = accuracy_score(y_test_binary, y_pred_binary)
    auc = roc_auc_score(y_test_binary, y_pred.flatten())
    
    # Calculate utility score (financial return metric specific to Jane Street competition)
    utility_score = model.calculate_utility_score(y_test, y_pred, weights_test)
    
    # Print metrics
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test AUC: {auc:.4f}")
    print(f"Test Utility Score: {utility_score:.4f}")
    
    # Plot precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_test_binary, y_pred.flatten())
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'precision_recall_curve.png'))
    
    # Plot threshold distribution
    plt.figure(figsize=(10, 6))
    plt.hist(y_pred.flatten(), bins=50, alpha=0.7)
    plt.axvline(0.5, color='red', linestyle='--', label='Threshold (0.5)')
    plt.xlabel('Prediction Score')
    plt.ylabel('Count')
    plt.title('Prediction Score Distribution')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'prediction_distribution.png'))
    
    # Plot utility by threshold
    thresholds = np.linspace(0.1, 0.9, 41)
    utilities = []
    
    for threshold in thresholds:
        actions = (y_pred.flatten() > threshold).astype(int)
        returns = actions * y_test * weights_test
        utility = np.sum(returns)
        utilities.append(utility)
    
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, utilities, marker='o')
    plt.axvline(0.5, color='red', linestyle='--', label='Default Threshold (0.5)')
    plt.xlabel('Threshold')
    plt.ylabel('Utility Score')
    plt.title('Utility Score by Decision Threshold')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'utility_by_threshold.png'))
    
    # Save metrics to a CSV file
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'AUC', 'Utility Score'],
        'Value': [accuracy, auc, utility_score]
    })
    metrics_df.to_csv(os.path.join(output_dir, 'metrics.csv'), index=False)
    
    # Return metrics
    return {
        'accuracy': accuracy,
        'auc': auc,
        'utility_score': utility_score
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate Jane Street model')
    parser.add_argument('--model_dir', type=str, default='models', help='Directory to save/load model')
    parser.add_argument('--data_dir', type=str, default='data', help='Directory with data files')
    parser.add_argument('--model_name', type=str, default='jane_street_model', help='Model name')
    args = parser.parse_args()
    
    # Import here to avoid circular imports
    from data_loader import JaneStreetDataLoader
    from model import JaneStreetModel
    
    # Initialize data loader and load data
    data_loader = JaneStreetDataLoader(data_path=args.data_dir)
    data_loader.load_data()
    
    # Preprocess data
    train_df, val_df, test_df = data_loader.preprocess_data()
    
    # Load model if it exists, otherwise train a new one
    model = JaneStreetModel(model_dir=args.model_dir, model_name=args.model_name)
    
    model_path = os.path.join(args.model_dir, f'{args.model_name}.pt')
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        model.load_model(model_path)
    else:
        print("Training new model...")
        # Get features, targets, and weights
        X_train, y_train, weights_train = data_loader.get_features_targets(train_df)
        X_val, y_val, weights_val = data_loader.get_features_targets(val_df)
        
        # Build and train model
        model.build_model()
        history = model.train(
            X_train, y_train, weights_train,
            X_val, y_val, weights_val,
            batch_size=1024,
            epochs=30,
            verbose=1
        )
        
        # Plot training history
        plt.figure(figsize=(10, 6))
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(args.model_dir, 'training_history.png'))
    
    # Evaluate model
    metrics = evaluate_model(model, test_df, data_loader, output_dir=args.model_dir)
    print("Evaluation complete!") 