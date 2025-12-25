"""MLP Model for Apache Log DDoS Detection."""
import torch
import torch.nn as nn
from typing import List


class ApacheDDoSModel(nn.Module):
    """
    MLP model for DDoS detection from Apache log features.
    Based on ICSE'22 paper methodology.
    """
    
    def __init__(
        self,
        input_dim: int = 16,
        hidden_layers: List[int] = [64, 32, 16],
        dropout_rate: float = 0.3
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def create_model(input_dim: int = 16) -> ApacheDDoSModel:
    """Create model with default configuration."""
    from .config import HIDDEN_LAYERS, DROPOUT_RATE
    return ApacheDDoSModel(input_dim, HIDDEN_LAYERS, DROPOUT_RATE)

