from benchopt import safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from skada import ConditionalTransferableComponentsAdapter, make_da_pipeline
    from benchmark_utils.base_solver import DASolver, FinalEstimator


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class Solver(DASolver):
    # Name to select the solver in the CLI and to display the results.
    name = 'CTC'

    # List of parameters for the solver. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    param_grid = {
        'conditionaltransferablecomponentsadapter__n_components': [1, 2, 5, 10, 20, 50, 100],
        'conditionaltransferablecomponentsadapter__lmbd': [1e-6, 1e-4, 1e-3],
        'conditionaltransferablecomponentsadapter__lmbd_s': [1e-6, 1e-4, 1e-3],
        'conditionaltransferablecomponentsadapter__lmbd_l': [1e-6, 1e-4, 1e-3],
        'finalestimator__estimator_name': ["LR", "SVC", "XGB"],
    }

    def get_estimator(self):
        # The estimator passed should have a 'predict_proba' method.
        return make_da_pipeline(
            ConditionalTransferableComponentsAdapter(),
            FinalEstimator(),
        )