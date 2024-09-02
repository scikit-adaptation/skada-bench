from benchopt import safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from skada import make_da_pipeline
    from benchmark_utils.base_solver import DASolver
    from skada.deep import DeepCoral
    from skada.deep.base import DomainAwareModule
    import torch
    from torch import nn
    import torchvision.models as models
    from torch.optim import Adam
    import collections


class ResNet50WithMLP(nn.Module):
    def __init__(self, n_classes=2, hidden_size=256, input_shape=(3, 224, 224)):
        super().__init__()
        
        self.input_shape = input_shape
        in_channels = input_shape[0]  # First dimension is always the number of channels

        # Load pre-trained ResNet50
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # Modify the first layer if input channels != 3
        if in_channels != 3:
            resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Remove the last fully connected layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        # To flatten the tensor
        self.flatten = nn.Flatten()
        # Add 2-layer MLP
        self.mlp = nn.Sequential(
            nn.Linear(2048, hidden_size),
            nn.ReLU(),
        )
        self.last_layers = nn.Sequential(
            nn.Linear(hidden_size, n_classes),
            nn.Softmax(dim=1),
        )

    def forward(self, x, sample_weight=None):
        x = x.reshape((x.shape[0], *self.input_shape))
        x = self.features(x)
        x = self.flatten(x)
        x = self.mlp(x)
        x = self.last_layers(x)

        return x


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class Solver(DASolver):
    # Name to select the solver in the CLI and to display the results.
    name = 'DeepCORAL'

    # List of parameters for the solver. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    default_param_grid = {
        'max_epochs': [20, 50],
        'optimizer__weight_decay': [1e-5, 1e-4, 1e-3],
        'lr': [1e-2, 1e-3, 1e-4],
    }


    def get_estimator(self, n_classes, input_shape, device):
        model = ResNet50WithMLP(n_classes=n_classes, input_shape=input_shape)
        net = DeepCoral(
            model,
            optimizer=Adam,
            reg=1,
            layer_name="mlp",
            batch_size=256,
            max_epochs=1,
            train_split=None,
            device=device,
        )

        return net
