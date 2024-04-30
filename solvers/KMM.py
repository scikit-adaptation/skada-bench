from benchopt import safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from skada import KMMReweightAdapter, make_da_pipeline
    from benchmark_utils.base_solver import DASolver
    from xgboost import XGBClassifier


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class Solver(DASolver):

    # Name to select the solver in the CLI and to display the results.
    name = 'KMM'

    requirements = [
        "pip:xgboost",
    ]

    # List of parameters for the solver. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    param_grid = {
        'kmmreweightadapter__gamma': [0.1, 1.0, 10.0],
        'kmmreweightadapter__B': [1000.0],
        'kmmreweightadapter__tol': [1e-6],
        'kmmreweightadapter__max_iter': [100],
        'kmmreweightadapter__smooth_weights': [False],
    }

    def get_estimator(self):
        # The estimator passed should have a 'predict_proba' method.
        return make_da_pipeline(
            KMMReweightAdapter(),
            XGBClassifier()
            .set_fit_request(sample_weight=True)
            .set_score_request(sample_weight=True),
        )
