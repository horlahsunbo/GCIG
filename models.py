"""
Neural network architectures for GCf-IG experiments.
"""

import torch
import torch.nn as nn


class MLPClassifier(nn.Module):
    """
    Simple MLP for binary classification.
    Outputs logits (use with BCEWithLogitsLoss).
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [64, 32],
        dropout: float = 0.1,
        use_batch_norm: bool = False
    ):
        """
        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer sizes
            dropout: Dropout probability
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer (logits)
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, input_dim)
        
        Returns:
            logits: (batch_size, 1)
        """
        return self.network(x).squeeze(-1)  # (batch_size,)


def create_model(input_dim: int, config: dict) -> nn.Module:
    """
    Factory function to create models based on config.
    
    Args:
        input_dim: Number of input features
        config: Dictionary with model hyperparameters
        
    Returns:
        model: PyTorch model
    """
    model_type = config.get('model_type', 'mlp')
    
    if model_type == 'mlp':
        return MLPClassifier(
            input_dim=input_dim,
            hidden_dims=config.get('hidden_dims', [64, 32]),
            dropout=config.get('dropout', 0.1),
            use_batch_norm=config.get('use_batch_norm', False)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
