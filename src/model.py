import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
from tqdm import tqdm

class JaneStreetNN(nn.Module):
    def __init__(self, input_dim=130, hidden_units=[256, 128, 64], dropout_rates=[0.3, 0.2, 0.1]):
        super(JaneStreetNN, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Create hidden layers
        for i, (units, dropout_rate) in enumerate(zip(hidden_units, dropout_rates)):
            layers.append(nn.Linear(prev_dim, units))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(units))
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = units
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

class JaneStreetModel:
    def __init__(
        self,
        input_dim=130,
        hidden_units=[256, 128, 64],
        dropout_rates=[0.3, 0.2, 0.1],
        learning_rate=0.001,
        model_dir='models',
        model_name='jane_street_model'
    ):
        self.input_dim = input_dim
        self.hidden_units = hidden_units
        self.dropout_rates = dropout_rates
        self.learning_rate = learning_rate
        self.model_dir = model_dir
        self.model_name = model_name
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create model_dir if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
    def build_model(self):
        """Build and initialize the model"""
        model = JaneStreetNN(
            input_dim=self.input_dim,
            hidden_units=self.hidden_units,
            dropout_rates=self.dropout_rates
        )
        model = model.to(self.device)
        self.model = model
        return model
    
    def train(
        self,
        X_train,
        y_train,
        weights_train,
        X_val=None,
        y_val=None,
        weights_val=None,
        batch_size=1024,
        epochs=50,
        verbose=1
    ):
        """Train the model"""
        if self.model is None:
            self.build_model()
            
        # Convert target to binary classification
        y_train_binary = (y_train > 0).astype(int)
        
        # Convert numpy arrays to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train_binary).reshape(-1, 1).to(self.device)
        weights_train_tensor = torch.FloatTensor(weights_train).reshape(-1, 1).to(self.device)
        
        # Create dataset and dataloader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor, weights_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Validation data
        val_loader = None
        if X_val is not None and y_val is not None:
            y_val_binary = (y_val > 0).astype(int)
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val_binary).reshape(-1, 1).to(self.device)
            weights_val_tensor = torch.FloatTensor(weights_val).reshape(-1, 1).to(self.device)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor, weights_val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Define loss function and optimizer
        criterion = nn.BCELoss(reduction='none')
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
        )
        
        # Training loop
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        max_patience = 10
        
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', disable=not verbose)
            for X_batch, y_batch, w_batch in progress_bar:
                # Forward pass
                outputs = self.model(X_batch)
                
                # Calculate weighted loss
                loss = criterion(outputs, y_batch)
                weighted_loss = (loss * w_batch).mean()
                
                # Backward pass and optimize
                optimizer.zero_grad()
                weighted_loss.backward()
                optimizer.step()
                
                train_loss += weighted_loss.item() * X_batch.size(0)
                progress_bar.set_postfix({'train_loss': weighted_loss.item()})
            
            train_loss /= len(train_loader.dataset)
            history['train_loss'].append(train_loss)
            
            # Validation phase
            val_loss = 0.0
            if val_loader:
                self.model.eval()
                with torch.no_grad():
                    for X_batch, y_batch, w_batch in val_loader:
                        outputs = self.model(X_batch)
                        loss = criterion(outputs, y_batch)
                        weighted_loss = (loss * w_batch).mean()
                        val_loss += weighted_loss.item() * X_batch.size(0)
                
                val_loss /= len(val_loader.dataset)
                history['val_loss'].append(val_loss)
                
                # Update learning rate scheduler
                scheduler.step(val_loss)
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = self.model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if verbose:
                    print(f'Epoch {epoch+1}/{epochs} - train_loss: {train_loss:.6f} - val_loss: {val_loss:.6f}')
                
                # Early stopping
                if patience_counter >= max_patience:
                    print(f'Early stopping at epoch {epoch+1}')
                    break
            else:
                # Save best model based on training loss
                if train_loss < best_val_loss:
                    best_val_loss = train_loss
                    best_model_state = self.model.state_dict().copy()
                
                if verbose:
                    print(f'Epoch {epoch+1}/{epochs} - train_loss: {train_loss:.6f}')
        
        # Restore best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)
            
        # Save best model
        self._save_checkpoint(os.path.join(self.model_dir, f'{self.model_name}.pt'))
            
        return history
    
    def predict(self, X):
        """Make predictions with the model"""
        if self.model is None:
            raise ValueError("Model has not been built yet. Call build_model() first.")
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Make predictions
        with torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy()
        
        return predictions
    
    def calculate_utility_score(self, y_true, y_pred, weights):
        """
        Calculate the Jane Street utility score:
        - y_true: actual returns (resp)
        - y_pred: predicted probabilities (from model)
        - weights: sample weights
        """
        # Convert predictions to actions (0 or 1)
        actions = (y_pred > 0.5).astype(int).flatten()
        
        # Calculate daily returns
        returns = actions * y_true * weights
        
        # Sum returns by date (in a real case, we would group by date)
        # Here we're just summing all returns
        total_return = np.sum(returns)
        
        # Calculate utility score
        # In the actual competition, this would include a scaling factor based on the standard deviation
        utility_score = total_return
        
        return utility_score
    
    def _save_checkpoint(self, filepath):
        """Save model checkpoint"""
        if self.model is None:
            raise ValueError("Model has not been built yet. Call build_model() first.")
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'input_dim': self.input_dim,
            'hidden_units': self.hidden_units,
            'dropout_rates': self.dropout_rates
        }
        
        torch.save(checkpoint, filepath)
        print(f"Model saved to {filepath}")
    
    def save_model(self, filepath=None):
        """Save the model"""
        if filepath is None:
            filepath = os.path.join(self.model_dir, f'{self.model_name}.pt')
        
        self._save_checkpoint(filepath)
    
    def load_model(self, filepath=None):
        """Load the model"""
        if filepath is None:
            filepath = os.path.join(self.model_dir, f'{self.model_name}.pt')
            
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No model found at {filepath}")
        
        # Load checkpoint
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Initialize model
        self.input_dim = checkpoint.get('input_dim', self.input_dim)
        self.hidden_units = checkpoint.get('hidden_units', self.hidden_units)
        self.dropout_rates = checkpoint.get('dropout_rates', self.dropout_rates)
        
        self.build_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Model loaded from {filepath}")
        return self.model 