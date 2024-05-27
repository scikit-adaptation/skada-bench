from benchopt import safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from skada import TransferSubspaceLearningAdapter, make_da_pipeline
    from benchmark_utils.base_solver import DASolver, FinalEstimator


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class Solver(DASolver):
    # Name to select the solver in the CLI and to display the results.
    name = 'transfer_subspace_learning'

    # List of parameters for the solver. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    default_param_grid = {
        'transfersubspacelearningadapter__n_components': [1, 2, 5, 10, 20, 50, 100],
        'transfersubspacelearningadapter__base_method': ['flda'],
        'transfersubspacelearningadapter__length_scale': [2],
        'transfersubspacelearningadapter__mu': [0.1, 1, 10],
        'transfersubspacelearningadapter__reg': [1e-4],
        'transfersubspacelearningadapter__max_iter': [300],
        'transfersubspacelearningadapter__tol': [1e-4],
        'finalestimator__estimator_name': ["LR", "SVC", "XGB"],
    }

    def get_estimator(self):
        # The estimator passed should have a 'predict_proba' method.
        return make_da_pipeline(
            TransferSubspaceLearningAdapter(),
            FinalEstimator(),
        )
