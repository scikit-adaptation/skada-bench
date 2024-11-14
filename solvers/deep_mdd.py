from benchopt import safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from benchmark_utils.deep_base_solver import DeepDASolver
    from benchmark_utils.utils import get_params_per_dataset
    from benchmark_utils.backbones_architecture import DiscrepancyClassifier
    from skada.deep import MDD

    from benchmark_utils.deep_base_solver import import_ctx as base_import_ctx
    if base_import_ctx.failed_import:
        exc, val, tb = base_import_ctx.import_error
        raise exc(val).with_traceback(tb)


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class Solver(DeepDASolver):
    # Name to select the solver in the CLI and to display the results.
    name = 'deep_mdd'

    # List of parameters for the solver. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    default_param_grid = {
        'criterion__reg': [1e-2, 1e-1, 1],
        'gamma': [2., 4.]
    }

    def get_estimator(self, n_classes, device, dataset_name, **kwargs):
        dataset_name = dataset_name.split("[")[0].lower()

        params = get_params_per_dataset(
            dataset_name, n_classes,
        )

        net = MDD(
            **params,
            layer_name="feature_layer",
            train_split=None,
            device=device,
            warm_start=True,
            disc_classifier=DiscrepancyClassifier(
                num_features=params['module'].n_features,
                n_classes=n_classes,
            ),
        )

        return net
