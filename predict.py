import os
import argparse
import numpy as np
import pandas as pd
import torch
import pickle
from model import get_model


def parse_args():
    parser = argparse.ArgumentParser(description='Make predictions with Jane Street Market Prediction model')
    parser.add_argument('--model_type', type=str, default='mlp', choices=['mlp', 'transformer'],
                        help='Type of model to use')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to the model to use. If not provided, will use the best model from the output directory.')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the data to make predictions on')
    parser.add_argument('--preprocessing_dir', type=str, default='data/processed',
                        help='Directory containing preprocessing objects')
    parser.add_argument('--output_path', type=str, default=None,
                        help='Path to save predictions. If not provided, will use data_path with _predictions suffix.')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size for prediction')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for binary predictions')
    parser.add_argument('--output_dir', type=str, default='models',
                        help='Directory containing model')
    return parser.parse_args()


def load_preprocessing_objects(preprocessing_dir):
    """
    Load preprocessing objects (imputer and scaler)
    """
    imputer_path = os.path.join(preprocessing_dir, 'imputer.pkl')
    scaler_path = os.path.join(preprocessing_dir, 'scaler.pkl')
    
    with open(imputer_path, 'rb') as f:
        imputer = pickle.load(f)
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    return imputer, scaler


def preprocess_data(df, imputer, scaler, feature_cols=None):
    """
    Preprocess data using loaded preprocessing objects
    """
    # Extract feature columns
    if feature_cols is None:
        feature_cols = [col for col in df.columns if col.startswith('feature')]
    
    # Handle missing values
    features_df = df[feature_cols]
    features_imputed = imputer.transform(features_df)
    
    # Standardize features
    features_scaled = scaler.transform(features_imputed)
    
    return features_scaled, feature_cols


def predict_in_batches(model, data, batch_size, device):
    """
    Make predictions in batches to avoid memory issues
    """
    model.eval()
    num_samples = data.shape[0]
    predictions = []
    probabilities = []
    
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            end_idx = min(i + batch_size, num_samples)
            batch_data = torch.tensor(data[i:end_idx], dtype=torch.float32).to(device)
            
            # Forward pass
            outputs = model(batch_data)
            probs = torch.sigmoid(outputs).cpu().numpy()
            
            probabilities.extend(probs)
    
    return np.array(probabilities)


def main():
    # Parse arguments
    args = parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print(f"Loading data from {args.data_path}")
    df = pd.read_csv(args.data_path)
    
    # Load preprocessing objects
    print("Loading preprocessing objects")
    imputer, scaler = load_preprocessing_objects(args.preprocessing_dir)
    
    # Preprocess data
    print("Preprocessing data")
    features_scaled, feature_cols = preprocess_data(df, imputer, scaler)
    
    # Initialize model
    model_kwargs = {
        'input_dim': len(feature_cols)
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
    
    # Make predictions
    print("Making predictions")
    probabilities = predict_in_batches(model, features_scaled, args.batch_size, device)
    predictions = (probabilities > args.threshold).astype(int)
    
    # Add predictions to dataframe
    df['probability'] = probabilities
    df['prediction'] = predictions
    
    # Save predictions
    if args.output_path is None:
        output_path = os.path.splitext(args.data_path)[0] + '_predictions.csv'
    else:
        output_path = args.output_path
    
    print(f"Saving predictions to {output_path}")
    df.to_csv(output_path, index=False)
    
    # Print summary
    print("\nPrediction Summary:")
    print(f"Total samples: {len(df)}")
    print(f"Positive predictions: {predictions.sum()} ({predictions.sum() / len(predictions) * 100:.2f}%)")
    print(f"Negative predictions: {len(predictions) - predictions.sum()} ({(len(predictions) - predictions.sum()) / len(predictions) * 100:.2f}%)")


if __name__ == '__main__':
    main() 