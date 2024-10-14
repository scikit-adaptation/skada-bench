from benchopt import safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from benchmark_utils.utils import get_params_per_dataset
    from benchmark_utils.base_solver import DeepDASolver
    from skada.deep import SourceOnly
    from skada.metrics import (
        SupervisedScorer, DeepEmbeddedValidation,
        PredictionEntropyScorer, ImportanceWeightedScorer,
        SoftNeighborhoodDensity,
    )

    from benchmark_utils.base_solver import import_ctx as base_import_ctx
    if base_import_ctx.failed_import:
        exc, val, tb = base_import_ctx.import_error
        raise exc(val).with_traceback(tb)


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class Solver(DeepDASolver):
    # Name to select the solver in the CLI and to display the results.
    name = 'Deep_NO_DA_SOURCE_ONLY'

    # List of parameters for the solver. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    default_param_grid = {}

    def get_estimator(self, n_classes, device, dataset_name, **kwargs):
        self.criterions = {
            'supervised': SupervisedScorer(),
        }

        dataset_name = dataset_name.split("[")[0].lower()

        params = get_params_per_dataset(
            dataset_name, n_classes,
        )

        net = SourceOnly(
            **params,
            layer_name="feature_layer",
            train_split=None,
            device=device,
            warm_start=True,
        )

        return net
