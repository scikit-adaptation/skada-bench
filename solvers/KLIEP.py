from benchopt import safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from skada import KLIEPAdapter, make_da_pipeline
    from benchmark_utils.base_solver import DASolver
    from xgboost import XGBClassifier


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class Solver(DASolver):

    # Name to select the solver in the CLI and to display the results.
    name = 'KLIEP'

    # List of parameters for the solver. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    param_grid = {
        'kliepadapter__gamma': [0.1, 1, 10],
        'kliepadapter__cv': [3, 5],
        'kliepadapter__n_centers': [5, 10, 20],
        'kliepadapter__tol': [1e-3, 1e-4],
        'kliepadapter__random_state': [0],
    }

    def get_estimator(self):
        # The estimator passed should have a 'predict_proba' method.
        return make_da_pipeline(
            KLIEPAdapter(gamma=0.1),
            XGBClassifier().set_fit_request(sample_weight=True),
        )
