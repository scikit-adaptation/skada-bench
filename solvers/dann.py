from benchopt import safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from benchmark_utils.utils import get_params_per_dataset
    from benchmark_utils.base_solver import DeepDASolver
    from benchmark_utils.backbones_architecture import DomainClassifier
    from skada.deep import DANN
    from skada.metrics import (
        SupervisedScorer, DeepEmbeddedValidation,
        PredictionEntropyScorer, ImportanceWeightedScorer,
        SoftNeighborhoodDensity, MixValScorer,
    )
    import numpy as np

    from benchmark_utils.base_solver import import_ctx as base_import_ctx
    if base_import_ctx.failed_import:
        exc, val, tb = base_import_ctx.import_error
        raise exc(val).with_traceback(tb)


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class Solver(DeepDASolver):
    # Name to select the solver in the CLI and to display the results.
    name = 'DANN'

    # List of parameters for the solver. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    default_param_grid = {
        'criterion__reg': np.logspace(-3, 0, 4),
    }

    def get_estimator(self, n_classes, device, dataset_name, **kwargs):
        self.criterions = {
            'supervised': SupervisedScorer(),
            'prediction_entropy': PredictionEntropyScorer(),
            'importance_weighted': ImportanceWeightedScorer(),
            'soft_neighborhood_density': SoftNeighborhoodDensity(),
            'deep_embedded_validation': DeepEmbeddedValidation(),
            'mix_val_both': MixValScorer(ice_type='both'),
            'mix_val_inter': MixValScorer(ice_type='inter'),
            'mix_val_intra': MixValScorer(ice_type='intra'),
        }

        dataset_name = dataset_name.split("[")[0].lower()

        params = get_params_per_dataset(
            dataset_name, n_classes,
        )
        # Reduce learning rate and increase momentum
        params['lr'] = params['lr'] * 0.1
        if 'optimizer__momentum' in params:
            params['optimizer__momentum'] = 0.9

        net = DANN(
            **params,
            layer_name="feature_layer",
            train_split=None,
            device=device,
            domain_classifier=DomainClassifier(
                num_features=params['module'].n_features),
            # optimizer__param_groups=[
            #     ('base_module_.feature_layer*', {'lr': params['lr']}),
            #     ('base_module_.final_layer*', {'lr': params['lr']}),
            #     ('domain_classifier_*', {'lr': params['lr'],}),
            # ]
        )

        return net
