# This file contains backbone architectures for deep models
# used in domain adaptation tasks.

from torch import nn
from torchvision.models import (
    resnet18, ResNet18_Weights, resnet50, ResNet50_Weights
)
from torch.autograd import Function


class ShallowConvNet(nn.Module):
    # This is a simple convolutional neural network (CNN) for MNIST dataset
    # https://github.com/pytorch/examples/blob/main/mnist/main.py
    def __init__(self, n_classes):
        super().__init__()
        self.feature_layer = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25),
            nn.Flatten(),
            nn.Linear(9216, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.final_layer = nn.Linear(128, n_classes)

        self.n_features = 128

    def forward(self, x, sample_weight=None):
        x = self.feature_layer(x)
        x = self.final_layer(x)

        return x


class ResNet(nn.Module):
    def __init__(self, n_classes, model_name="resnet18"):
        super().__init__()

        # Load pretrained ResNet and rename it to feature_layer
        if model_name == "resnet18":
            self.feature_layer = resnet18(weights=ResNet18_Weights.DEFAULT)
        elif model_name == "resnet50":
            self.feature_layer = resnet50(weights=ResNet50_Weights.DEFAULT)

        # Get the number of features from the last layer
        self.n_features = num_ftrs = self.feature_layer.fc.in_features

        # Replace ResNet's fc layer with an identity function
        self.feature_layer.fc = nn.Identity()

        # Create a new fc layer
        self.final_layer = nn.Linear(num_ftrs, n_classes)

        self.n_features = num_ftrs

    def forward(self, x, sample_weight=None):
        x = self.feature_layer(x)
        x = self.final_layer(x)
        return x


class ShallowMLP(nn.Module):
    def __init__(self, input_dim, n_classes, hidden_dim=64):
        super().__init__()
        self.feature_layer = nn.Linear(input_dim, hidden_dim)
        self.mlp = nn.Sequential(
            nn.ReLU(), nn.Dropout(0.5), nn.Linear(hidden_dim, n_classes)
        )

        self.n_features = hidden_dim

    def forward(self, x, sample_weight=None):
        x = self.feature_layer(x)
        x = self.mlp(x)

        return x  # Return raw logits


class FBCSPNet(nn.Module):
    def __init__(self, n_chans, n_classes, input_window_samples):
        super().__init__()

        from braindecode.models import ShallowFBCSPNet

        # Create the ShallowFBCSPNet
        self.feature_layer = ShallowFBCSPNet(
            n_chans,
            n_classes,
            input_window_samples,
            add_log_softmax=False,
            final_conv_length="auto",
        )

        # Take last layer from the ShallowFBCSPNet
        self.final_layer = self.feature_layer.final_layer

        # Replace ShallowFBCSPNet's last layer with an identity function
        self.feature_layer.final_layer = nn.Identity()

        self.n_features = 40 * 69

    def forward(self, x, sample_weight=None):
        x = self.feature_layer(x)
        x = self.final_layer(x)

        return x


class GradientReversalLayer(Function):
    """Leaves the input unchanged during forward propagation
    and reverses the gradient by multiplying it by a
    negative scalar during the backpropagation.
    """

    @staticmethod
    def forward(ctx, x, alpha, sample_weight=None):
        """XXX add docstring here."""
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        """XXX add docstring here."""
        output = grad_output.neg() * ctx.alpha
        return output, None


class DomainClassifier(nn.Module):
    """Classifier Architecture from DANN paper [15]_.

    Parameters
    ----------
    num_features : int
        Size of the input, e.g size of the last layer of
        the feature extractor
    n_classes : int, default=1
        Number of classes

    References
    ----------
    .. [15]  Yaroslav Ganin et. al. Domain-Adversarial Training
            of Neural Networks  In Journal of Machine Learning
            Research, 2016.
    """

    def __init__(self, num_features, hidden_size=1024, n_classes=1, alpha=1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(num_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, n_classes),
            nn.Sigmoid(),
        )
        self.alpha = alpha

    def forward(self, x, sample_weight=None):
        """Forward pass.

        Parameters
        ----------
        x: torch.Tensor
            Batch of EEG windows of shape (batch_size, n_channels, n_times).
        alpha: float
            Parameter for the reverse layer.
        """
        reverse_x = GradientReversalLayer.apply(x, self.alpha)
        return self.classifier(reverse_x).flatten()
