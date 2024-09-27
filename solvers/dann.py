from benchopt import safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from benchmark_utils.utils import get_params_per_dataset
    from benchmark_utils.base_solver import DeepDASolver
    from benchmark_utils.backbones_architecture import DomainClassifier
    from skada.deep import DANN
    from torch.optim import SGD
    from skada.metrics import (
        SupervisedScorer, DeepEmbeddedValidation,
        PredictionEntropyScorer, ImportanceWeightedScorer,
        SoftNeighborhoodDensity,
    )


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class Solver(DeepDASolver):
    # Name to select the solver in the CLI and to display the results.
    name = 'DANN'

    # List of parameters for the solver. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    default_param_grid = {
        'criterion__reg': [0.1, 0.01, 0.001],
    }

    def get_estimator(self, n_classes, device, dataset_name, **kwargs):
        self.criterions = {
            'supervised': SupervisedScorer(),
            'prediction_entropy': PredictionEntropyScorer(),
            'importance_weighted': ImportanceWeightedScorer(),
            'soft_neighborhood_density': SoftNeighborhoodDensity(),
            'deep_embedded_validation': DeepEmbeddedValidation(),
        }

        dataset_name = dataset_name.split("[")[0].lower()

        params = get_params_per_dataset(
            dataset_name, n_classes,
        )

        net = DANN(
            params['model'],
            optimizer=SGD,
            layer_name="feature_layer",
            batch_size=params['batch_size'],
            train_split=None,
            device=device,
            callbacks=[params['lr_scheduler']],
            max_epochs=params['max_epochs'],
            num_features=params['num_features'],
            domain_classifier=DomainClassifier(num_features=params['num_features']),
            optimizer__param_groups=[
                ('base_module_.feature_layer*', {'lr': params['lr']}),
                ('base_module_.final_layer*', {'lr': 10*params['lr']}),
                ('domain_classifier_*', {'lr': 10*params['lr'],}),
            ]
        )

        return net