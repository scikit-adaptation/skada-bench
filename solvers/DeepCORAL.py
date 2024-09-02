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
    from torch.optim import Adam, Adadelta
    import collections
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
    def __init__(self, n_classes=10, hidden_size=128, input_shape=(1, 28, 28)):
        super().__init__()
        
        self.input_shape = input_shape
        in_channels = input_shape[0]  # First dimension is always the number of channels

        # Define the layers as per the Net architecture
        self.conv1 = nn.Conv2d(in_channels, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.mlp = nn.Linear(9216, hidden_size)
        self.fc2 = nn.Linear(hidden_size, n_classes)

    def forward(self, x, sample_weight=None):
        x = x.reshape((x.shape[0], *self.input_shape))
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(2)(x)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.mlp(x)
        x = nn.ReLU()(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = nn.LogSoftmax(dim=1)(x)

        return output


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
        'lr': [1],
    }


    def get_estimator(self, n_classes, input_shape, device):
        # For testing purposes, we use the following criterions:
        self.criterions = {
            'supervised': SupervisedScorer(),
            'deep_embedded_validation': DeepEmbeddedValidation(),
        }

        #model = ResNet50WithMLP(n_classes=n_classes, input_shape=input_shape)
        model = ShallowConvNet(n_classes=n_classes, input_shape=input_shape)
        net = DeepCoral(
            model,
            optimizer=Adadelta,  # Adadelta is now defined
            reg=1,
            layer_name="mlp",
            batch_size=256,
            max_epochs=1,
            train_split=None,
            device=device,
        )

        return net
