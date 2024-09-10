# This file contains backbone architectures for deep models used in domain adaptation tasks.

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


class ShallowConvNet(nn.Module):
    # This is a simple convolutional neural network (CNN) for MNIST dataset
    # https://github.com/pytorch/examples/blob/main/mnist/main.py
    def __init__(self, n_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x, sample_weight=None):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)

        return x

class OfficeConvNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        
        # Load pretrained ResNet50
        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        
        num_ftrs = self.resnet.fc.in_features
        self.fc1 = nn.Linear(num_ftrs, 1024)
        self.fc2 = nn.Linear(1024, n_classes)
        
        # Replace ResNet's fc layer with Identity
        self.resnet.fc = nn.Identity()

    def forward(self, x, sample_weight=None):
        x = self.resnet(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
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
