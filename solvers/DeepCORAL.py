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
        return nn.LogSoftmax(dim=1)(x)


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


    def get_estimator(self, n_classes, device, dataset_name, **kwargs):
        # For testing purposes, we use the following criterions:
        self.criterions = {
            'supervised': SupervisedScorer(),
            'deep_embedded_validation': DeepEmbeddedValidation(),
        }

        dataset_name = dataset_name.split("[")[0].lower()

        if dataset_name in ['mnist_usps']:
            model = ShallowConvNet(n_classes=n_classes)
        elif dataset_name in ['office31', 'officehome']:
            # To change for a more suitable net
            model = ShallowConvNet(n_classes=n_classes)
        elif dataset_name in ['bci']:
            # Use double precision for BCI data
            model = ShallowMLP(input_dim=253, n_classes=n_classes).double()
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        
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
