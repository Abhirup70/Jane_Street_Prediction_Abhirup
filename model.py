import torch
import torch.nn as nn
import torch.nn.functional as F


class JaneStreetMLP(nn.Module):
    """
    Multi-layer Perceptron model for Jane Street Market Prediction
    """
    def __init__(self, input_dim=130, hidden_dims=[256, 128, 64], dropout_rate=0.3):
        super(JaneStreetMLP, self).__init__()
        
        # Store dimensions
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        
        # Create layers
        layers = []
        prev_dim = input_dim
        
        # Hidden layers
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        # Combine layers
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


class JaneStreetTransformer(nn.Module):
    """
    Transformer-based model for Jane Street Market Prediction
    """
    def __init__(self, input_dim=130, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super(JaneStreetTransformer, self).__init__()
        
        # Feature embedding
        self.embedding = nn.Linear(input_dim, d_model)
        
        # Positional encoding is not used since features don't have a sequential order
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model*4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layer
        self.fc_out = nn.Linear(d_model, 1)
    
    def forward(self, x):
        # Reshape x to add sequence dimension if needed
        # x: [batch_size, features]
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # [batch_size, 1, features]
        
        # Embed features
        x = self.embedding(x)  # [batch_size, seq_len, d_model]
        
        # Apply transformer encoder
        x = self.transformer_encoder(x)  # [batch_size, seq_len, d_model]
        
        # Global average pooling over sequence dimension
        x = torch.mean(x, dim=1)  # [batch_size, d_model]
        
        # Output layer
        x = self.fc_out(x)  # [batch_size, 1]
        
        return x


def get_model(model_type='mlp', **kwargs):
    """
    Factory function to create a model instance
    
    Args:
        model_type: Type of model to create ('mlp' or 'transformer')
        kwargs: Additional arguments to pass to the model constructor
    
    Returns:
        A model instance
    """
    if model_type.lower() == 'mlp':
        return JaneStreetMLP(**kwargs)
    elif model_type.lower() == 'transformer':
        return JaneStreetTransformer(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}") 