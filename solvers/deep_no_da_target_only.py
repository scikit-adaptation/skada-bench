from benchopt import safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from benchmark_utils.base_solver import DASolver
    from benchmark_utils.utils import get_params_per_dataset
    from skada.metrics import SupervisedScorer, DeepEmbeddedValidation
    from skada.deep import TargetOnly


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class Solver(DASolver):
    # Name to select the solver in the CLI and to display the results.
    name = 'Deep_NO_DA_TARGET_ONLY'

    # List of parameters for the solver. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    default_param_grid = {
        'max_epochs': [14],
        'lr': [1],
    }

    def get_estimator(self, n_classes, device, dataset_name, **kwargs):
        # For testing purposes, we use the following criterions:
        self.criterions = {
            'supervised': SupervisedScorer(),
            'deep_embedded_validation': DeepEmbeddedValidation(),
        }

        dataset_name = dataset_name.split("[")[0].lower()

        params = get_params_per_dataset(
            dataset_name, n_classes, n_epochs=self.default_param_grid['max_epochs'][0]
        )

        net = TargetOnly(
            params['model'],
            optimizer=params['optimizer'],
            layer_name="feature_layer",
            batch_size=params['batch_size'],
            train_split=None,
            device=device,
            callbacks=[params['lr_scheduler']],
        )

        return net

