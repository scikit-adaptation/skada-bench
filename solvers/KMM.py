from benchopt import safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from skada import KMMReweightAdapter, make_da_pipeline
    from benchmark_utils.base_solver import DASolver, FinalEstimator

    from benchmark_utils.base_solver import import_ctx as base_import_ctx
    if base_import_ctx.failed_import:
        exc, val, tb = base_import_ctx.import_error
        raise exc(val).with_traceback(tb)


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class Solver(DASolver):

    # Name to select the solver in the CLI and to display the results.
    name = 'KMM'

    default_param_grid = {
        'kmmreweightadapter__gamma': [
            0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100., 1000., None
        ],
        'kmmreweightadapter__B': [1000.0],
        'kmmreweightadapter__tol': [1e-6],
        'kmmreweightadapter__max_iter': [1000],
        'kmmreweightadapter__smooth_weights': [False],
        'finalestimator__estimator_name': ["LR", "SVC", "XGB"],
    }

    def get_estimator(self):
        # The estimator passed should have a 'predict_proba' method.
        return make_da_pipeline(
            KMMReweightAdapter(),
            FinalEstimator(),
        )
