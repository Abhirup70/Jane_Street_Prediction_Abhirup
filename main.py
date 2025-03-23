import os
import sys
import argparse
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

# Add the current directory to the path to find src modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from src.data_loader import JaneStreetDataLoader
    from src.model import JaneStreetModel
    from src.evaluate import evaluate_model
    print("All modules imported successfully")
except Exception as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

def main(args):
    """
    Main function to run the Jane Street model pipeline
    """
    # Create directories if they don't exist
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Print hardware info
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print("\n" + "="*50)
    print(f"Jane Street Market Prediction Model - {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*50)
    
    # Initialize data loader and load data
    print("\n[1] Loading and preprocessing data")
    data_loader = JaneStreetDataLoader(data_path=args.data_dir)
    data_loader.load_data(sample_size=args.sample_size)
    
    # Preprocess data
    train_df, val_df, test_df = data_loader.preprocess_data()
    
    # Initialize model
    print("\n[2] Initializing model")
    model = JaneStreetModel(
        input_dim=len(data_loader.feature_cols),
        hidden_units=args.hidden_units,
        dropout_rates=args.dropout_rates,
        learning_rate=args.learning_rate,
        model_dir=args.model_dir,
        model_name=args.model_name
    )
    
    # Train or load model
    model_path = os.path.join(args.model_dir, f'{args.model_name}.pt')
    if os.path.exists(model_path) and not args.force_train:
        print(f"\n[3] Loading existing model from {model_path}")
        model.load_model(model_path)
    else:
        print("\n[3] Training new model")
        start_time = time.time()
        
        # Get features, targets, and weights
        X_train, y_train, weights_train = data_loader.get_features_targets(train_df, target_col='resp')
        X_val, y_val, weights_val = data_loader.get_features_targets(val_df, target_col='resp')
        
        print(f"Training data: {X_train.shape}, Validation data: {X_val.shape}")
        print(f"Target distribution: {np.mean(y_train > 0):.2%} positive returns")
        
        # Build and train model
        model.build_model()
        print(f"\nModel architecture: heloo abhirup")
        print(model.model)
        print(f"\nTotal parameters: {sum(p.numel() for p in model.model.parameters()):,}")
        
        history = model.train(
            X_train, y_train, weights_train,
            X_val, y_val, weights_val,
            batch_size=args.batch_size,
            epochs=args.epochs,
            verbose=args.verbose
        )
        
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
        
        # Plot training history
        if len(history['train_loss']) > 0:
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.plot(history['train_loss'], label='Train Loss')
            if 'val_loss' in history and len(history['val_loss']) > 0:
                plt.plot(history['val_loss'], label='Val Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training Loss')
            plt.legend()
            plt.grid(True)
            
            # Plot learning rate if available
            if 'lr' in history and len(history['lr']) > 0:
                plt.subplot(1, 2, 2)
                plt.plot(history['lr'])
                plt.xlabel('Epoch')
                plt.ylabel('Learning Rate')
                plt.title('Learning Rate Schedule')
                plt.grid(True)
                
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, 'training_history.png'))
            print(f"Training history saved to {os.path.join(args.output_dir, 'training_history.png')}")
    
    # Evaluate model
    if args.evaluate:
        print("\n[4] Evaluating model")
        metrics = evaluate_model(model, test_df, data_loader, output_dir=args.output_dir)
        
        # Print metrics summary
        print("\nMetrics Summary:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  AUC: {metrics['auc']:.4f}")
        print(f"  Utility Score: {metrics['utility_score']:.4f}")
        
        # Save metrics to CSV
        metrics_df = pd.DataFrame([metrics])
        metrics_path = os.path.join(args.output_dir, 'metrics.csv')
        metrics_df.to_csv(metrics_path, index=False)
        print(f"Metrics saved to {metrics_path}")
    
    print("\n" + "="*50)
    print("Pipeline completed successfully!")
    print("="*50)
    
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Jane Street Market Prediction Model')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='data', help='Directory with data files')
    parser.add_argument('--sample_size', type=int, default=None, help='Number of samples to use (default: all)')
    
    # Model parameters
    parser.add_argument('--model_dir', type=str, default='models', help='Directory to save/load model')
    parser.add_argument('--model_name', type=str, default='jane_street_model', help='Model name')
    parser.add_argument('--hidden_units', type=int, nargs='+', default=[256, 128, 64], help='Hidden layer units')
    parser.add_argument('--dropout_rates', type=float, nargs='+', default=[0.3, 0.2, 0.1], help='Dropout rates')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='output', help='Directory for output files')
    parser.add_argument('--verbose', type=int, default=1, help='Verbosity level (0=silent, 1=progress, 2=detailed)')
    
    # Control flags
    parser.add_argument('--force_train', action='store_true', help='Force training even if model exists')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate model after training/loading')
    
    args = parser.parse_args()
    
    # Print arguments
    print("Arguments:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
    
    main(args) 