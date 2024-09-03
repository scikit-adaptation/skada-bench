from benchopt import safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from benchmark_utils.base_solver import DASolver
    from skada.deep import DeepCoral
    import torch
    from torch import nn
    from torch.optim import AdamW
    from skada.metrics import SupervisedScorer, DeepEmbeddedValidation


# class ResNet50WithMLP(nn.Module):
#     def __init__(self, n_classes=2, hidden_size=256, input_shape=(3, 224, 224)):
#         super().__init__()
        
#         self.input_shape = input_shape
#         in_channels = input_shape[0]  # First dimension is always the number of channels

#         # Load pre-trained ResNet50
#         resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
#         # Modify the first layer if input channels != 3
#         if in_channels != 3:
#             resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         # Remove the last fully connected layer
#         self.features = nn.Sequential(*list(resnet.children())[:-1])
#         # To flatten the tensor
#         self.flatten = nn.Flatten()
#         # Add 2-layer MLP
#         self.mlp = nn.Sequential(
#             nn.Linear(2048, hidden_size),
#             nn.ReLU(),
#         )
#         self.last_layers = nn.Sequential(
#             nn.Linear(hidden_size, n_classes),
#             nn.Softmax(dim=1),
#         )

#     def forward(self, x, sample_weight=None):
#         x = x.reshape((x.shape[0], *self.input_shape))
#         x = self.features(x)
#         x = self.flatten(x)
#         x = self.mlp(x)
#         x = self.last_layers(x)

#         return x


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

        # Apply log softmax to get log probabilities
        logits = nn.LogSoftmax(dim=1)(x)

        return logits


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class Solver(DASolver):
    # Name to select the solver in the CLI and to display the results.
    name = 'DeepCORAL'

    # List of parameters for the solver. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    default_param_grid = {
        'max_epochs': [20],
        'lr': [1e-3],
    }


    def get_estimator(self, n_classes, device):
        # For testing purposes, we use the following criterions:
        self.criterions = {
            'supervised': SupervisedScorer(),
            'deep_embedded_validation': DeepEmbeddedValidation(),
        }

        #model = ResNet50WithMLP(n_classes=n_classes, input_shape=input_shape)
        model = ShallowConvNet(n_classes=n_classes)
        net = DeepCoral(
            model,
            optimizer=AdamW,
            reg=1,
            layer_name="feature_layer",
            batch_size=256,
            max_epochs=1,
            train_split=None,
            device=device,
        )

        return net
