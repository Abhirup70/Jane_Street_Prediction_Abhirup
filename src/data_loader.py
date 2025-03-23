import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import time

class JaneStreetDataLoader:
    def __init__(self, data_path='data'):
        self.data_path = data_path
        self.feature_cols = [f'feature_{i}' for i in range(130)]
        self.target_cols = ['resp', 'resp_1', 'resp_2', 'resp_3', 'resp_4']
        self.weight_col = 'weight'
        self.date_col = 'date'
        self.ts_id_col = 'ts_id'
        self.action_col = 'action'
        self.train_data = None
        self.scaler = StandardScaler()
        
    def load_data(self, sample_size=None, verbose=True):
        """
        Load data from CSV files in the data directory
        If sample_size is provided, only load a sample of the data
        """
        if verbose:
            print("Loading data...")
            start_time = time.time()
        
        # Path to the training data
        train_path = os.path.join(self.data_path, "train.csv")
        
        if os.path.exists(train_path):
            if verbose:
                print(f"Loading data from {train_path}")
            
            # Check file size to determine if we should use chunk loading
            file_size_gb = os.path.getsize(train_path) / (1024**3)
            
            if file_size_gb > 1 and sample_size is None:
                if verbose:
                    print(f"File size: {file_size_gb:.2f} GB. Loading in chunks...")
                
                # Load in chunks to manage memory
                chunk_size = 500000  # Adjust based on your system's memory
                chunks = []
                
                for chunk in pd.read_csv(train_path, chunksize=chunk_size):
                    if sample_size and sum(len(c) for c in chunks) >= sample_size:
                        break
                    chunks.append(chunk)
                
                self.train_data = pd.concat(chunks, ignore_index=True)
                
                if sample_size:
                    self.train_data = self.train_data.head(sample_size)
            else:
                # Load all data or a sample
                if sample_size:
                    # Read only the first 'sample_size' rows for efficiency
                    self.train_data = pd.read_csv(train_path, nrows=sample_size)
                else:
                    self.train_data = pd.read_csv(train_path)
        else:
            if verbose:
                print(f"No data found at {train_path}. Creating synthetic data for demonstration.")
            
            # Create synthetic data for demonstration
            self._create_synthetic_data(sample_size or 10000)
        
        if verbose:
            print(f"Data loaded with shape: {self.train_data.shape}")
            print(f"Loading time: {time.time() - start_time:.2f} seconds")
            
            # Display missing value information
            missing_values = self.train_data.isnull().sum()
            if missing_values.sum() > 0:
                print("\nMissing values in the dataset:")
                print(missing_values[missing_values > 0])
            
            # Display target distribution
            if self.action_col in self.train_data.columns:
                action_dist = self.train_data[self.action_col].value_counts(normalize=True) * 100
                print("\nTarget distribution:")
                print(f"  Action 0 (Pass): {action_dist.get(0, 0):.2f}%")
                print(f"  Action 1 (Trade): {action_dist.get(1, 0):.2f}%")
            
            # Display date range
            if self.date_col in self.train_data.columns:
                date_min = self.train_data[self.date_col].min()
                date_max = self.train_data[self.date_col].max()
                unique_dates = self.train_data[self.date_col].nunique()
                print(f"\nDate range: {date_min} to {date_max} ({unique_dates} unique days)")
        
        return self.train_data
    
    def _create_synthetic_data(self, n_samples=10000):
        """Create synthetic data for demonstration purposes"""
        np.random.seed(42)
        
        # Generate features
        features = np.random.randn(n_samples, 130)
        
        # Generate targets (responses)
        responses = np.random.randn(n_samples, 5) * 0.1
        
        # Generate weights, dates, and ts_ids
        weights = np.abs(np.random.randn(n_samples)) * 0.1
        dates = np.random.randint(0, 500, n_samples)
        ts_ids = np.sort(np.random.randint(0, 10000, n_samples))
        
        # Create DataFrame
        data = pd.DataFrame(features, columns=self.feature_cols)
        for i, col in enumerate(self.target_cols):
            data[col] = responses[:, i]
        
        data[self.weight_col] = weights
        data[self.date_col] = dates
        data[self.ts_id_col] = ts_ids
        
        # Add action column (0 or 1)
        data[self.action_col] = (responses[:, 0] > 0).astype(int)
        
        self.train_data = data
        
    def preprocess_data(self, train_size=0.8, val_size=0.1, test_size=0.1, random_state=42, verbose=True):
        """
        Preprocess the data:
        1. Handle missing values
        2. Scale features
        3. Split data into train/val/test sets
        """
        if self.train_data is None:
            self.load_data()
            
        if verbose:
            print("Preprocessing data...")
            start_time = time.time()
            
        df = self.train_data.copy()
        
        # Fill missing values
        for col in self.feature_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].mean())
        
        # Scale features
        if verbose:
            print("Scaling features...")
        df[self.feature_cols] = self.scaler.fit_transform(df[self.feature_cols])
        
        # Split data by date to avoid data leakage
        if verbose:
            print("Splitting data by date...")
        
        dates = df[self.date_col].unique()
        dates.sort()
        
        train_dates = dates[:int(len(dates) * train_size)]
        val_dates = dates[int(len(dates) * train_size):int(len(dates) * (train_size + val_size))]
        test_dates = dates[int(len(dates) * (train_size + val_size)):]
        
        train_df = df[df[self.date_col].isin(train_dates)]
        val_df = df[df[self.date_col].isin(val_dates)]
        test_df = df[df[self.date_col].isin(test_dates)]
        
        if verbose:
            print(f"Train set: {train_df.shape} ({train_df.shape[0]/df.shape[0]*100:.2f}% of data)")
            print(f"Validation set: {val_df.shape} ({val_df.shape[0]/df.shape[0]*100:.2f}% of data)")
            print(f"Test set: {test_df.shape} ({test_df.shape[0]/df.shape[0]*100:.2f}% of data)")
            print(f"Preprocessing time: {time.time() - start_time:.2f} seconds")
        
        return train_df, val_df, test_df
    
    def get_features_targets(self, data, target_col='resp'):
        """Extract features and target from data"""
        X = data[self.feature_cols].values
        
        # Check which target we're using
        if target_col in data.columns:
            y = data[target_col].values
        else:
            # Default to action for prediction if target_col not found
            y = (data[self.target_cols[0]] > 0).astype(int).values if self.target_cols[0] in data.columns else None
        
        # Get weights
        weights = data[self.weight_col].values if self.weight_col in data.columns else np.ones_like(y)
        
        return X, y, weights 