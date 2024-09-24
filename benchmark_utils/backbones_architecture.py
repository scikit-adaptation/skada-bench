# This file contains backbone architectures for deep models used in domain adaptation tasks.

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import (
    resnet18, ResNet18_Weights,
    resnet50, ResNet50_Weights
)
from braindecode.models import ShallowFBCSPNet


class ShallowConvNet(nn.Module):
    # This is a simple convolutional neural network (CNN) for MNIST dataset
    # https://github.com/pytorch/examples/blob/main/mnist/main.py
    def __init__(self, n_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.feature_layer = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x, sample_weight=None):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.feature_layer(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)

        return x


class ResNet(nn.Module):
    def __init__(self, n_classes, model_name='resnet18'):
        super().__init__()

        # Load pretrained ResNet and rename it to feature_layer
        if model_name == 'resnet18':
            self.feature_layer = resnet18(weights=ResNet18_Weights.DEFAULT)
        elif model_name == 'resnet50':
            self.feature_layer = resnet50(weights=ResNet50_Weights.DEFAULT)

        # Get the number of features from the last layer
        num_ftrs = self.feature_layer.fc.in_features

        # Replace ResNet's fc layer with an identity function
        self.feature_layer.fc = nn.Identity()

        # Create a new fc layer
        self.fc = nn.Linear(num_ftrs, n_classes)

    def forward(self, x, sample_weight=None):
        x = self.feature_layer(x)
        x = self.fc(x)
        return x


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


class FBCSPNet(nn.Module):
    def __init__(self, n_chans, n_classes, input_window_samples):
        super().__init__()

        # Create the ShallowFBCSPNet
        self.feature_layer = ShallowFBCSPNet(
            n_chans, n_classes, input_window_samples, add_log_softmax=False, final_conv_length='auto'
        )

        # Take last layer from the ShallowFBCSPNet
        self.final_layer = self.feature_layer.final_layer

        # Replace ShallowFBCSPNet's last layer with an identity function
        self.feature_layer.final_layer = nn.Identity()

    def forward(self, x, sample_weight=None):
        x = self.feature_layer(x)
        x = self.final_layer(x)
    
        return x
