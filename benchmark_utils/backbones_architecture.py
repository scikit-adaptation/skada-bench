# This file contains backbone architectures for deep models used in domain adaptation tasks.

import torch
from torch import nn

class ShallowConvNet(nn.Module):
    # This is a simple convolutional neural network (CNN) for MNIST dataset
    # https://github.com/pytorch/examples/blob/main/mnist/main.py
    def __init__(self, n_classes):
        super().__init__()

        # Convolutional layers and first dropout in a ModuleList
        self.convs = nn.ModuleList([
            nn.Conv2d(3, 32, 3, 1),  # First convolutional layer
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1),  # Second convolutional layer
            nn.ReLU(),
            nn.Dropout(0.25)  # First dropout layer
        ])
        
        # Global Average Pooling (this layer is fixed and doesn't depend on input size)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # MLP layers with ReLU and Dropout
        self.feature_layer = nn.Linear(64, 128) # First MLP layer
        self.mlp = nn.ModuleList([
            nn.ReLU(),
            nn.Dropout(0.5),  # Dropout after the first MLP layer
            nn.Linear(128, n_classes)  # Output layer directly in MLP
        ])
    
    def forward(self, x, sample_weight=None):
        # Forward pass through convolutional layers and first dropout
        for layer in self.convs:
            x = layer(x)
        
        # Apply global average pooling
        x = self.global_pool(x)
        
        # Flatten the tensor
        x = torch.flatten(x, 1)
        
        # Forward pass through MLP layers
        x = self.feature_layer(x)
        for layer in self.mlp:
            x = layer(x)

        return x    # Return raw logits


class ShallowMLP(nn.Module):
    def __init__(self, input_dim, n_classes, hidden_dim=64):
        super().__init__()
        self.feature_layer = nn.Linear(input_dim, hidden_dim)
        self.mlp = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, n_classes)
        )
    
    def forward(self, x, sample_weight=None):
        x = self.feature_layer(x)
        x = self.mlp(x)

        return x    # Return raw logits
