from benchopt import safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from skada import make_da_pipeline
    from benchmark_utils.base_solver import DASolver
    from skada.deep import DeepCoral
    import torch
    from torch import nn
    import torchvision.models as models
    from torch.optim import Adam


class ResNet50WithMLP(nn.Module):
    def __init__(self, n_classes=2, hidden_size=256):
        super(ResNet50WithMLP, self).__init__()
        
        # Load pre-trained ResNet50
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # Remove the last fully connected layer
        self.features = nn.Sequential(*list(self.resnet.children())[:-1])
        
        # Add 2-layer MLP
        self.mlp = nn.Sequential(
            nn.Linear(2048, hidden_size),
            nn.ReLU(),
        )
        self.last_layers = nn.Sequential(
            nn.Linear(hidden_size, n_classes),
            nn.Softmax(dim=1),
        )
        #self.last_layer = nn.Linear(hidden_size, n_classes)
    
    def forward(self, x):
        x = x.reshape((x.shape[0], 3, 224, 224))
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
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
        'max_epochs': [50, 100, 200],
        'optimizer__weight_decay': [0, 1e-5, 1e-7],
        'lr': [1e-5, 1e-3, 1e-2]
    }


    def get_estimator(self, n_classes, device):
        model = ResNet50WithMLP(n_classes)
        net = DeepCoral(
            model,
            optimizer=Adam,
            reg=1,
            layer_name="mlp",
            batch_size=2056,
            max_epochs=100,
            train_split=None,
            device=device,
        )

        return net
