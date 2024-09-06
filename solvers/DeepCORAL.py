from benchopt import safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from benchmark_utils.base_solver import DASolver
    from benchmark_utils.backbones_architecture import ShallowConvNet, ShallowMLP
    from skorch.callbacks import LRScheduler
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from skada.deep import DeepCoral
    from torch.optim import AdamW
    from skada.metrics import SupervisedScorer, DeepEmbeddedValidation


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
        'criterion__reg': [1],
        'optimizer__weight_decay': [0.01],
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

        # Define learning rate scheduler
        lr_scheduler = LRScheduler(
            policy = ReduceLROnPlateau,
            monitor = 'train_loss',
            mode = 'min',
            patience = 3,
            factor = 0.1,
        )

        # Initialize DeepCORAL network
        net = DeepCoral(
            model,
            optimizer=AdamW,
            layer_name="feature_layer",
            batch_size=256,
            max_epochs=1,
            train_split=None,
            device=device,
            callbacks = [lr_scheduler],
        )

        return net
