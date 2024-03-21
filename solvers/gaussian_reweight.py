from benchopt import safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from skada import GaussianReweightAdapter, make_da_pipeline
    from benchmark_utils.base_solver import DASolver
    from xgboost import XGBClassifier


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class Solver(DASolver):

    # Name to select the solver in the CLI and to display the results.
    name = 'gaussian_reweight'

    # List of parameters for the solver. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    param_grid_dict = {'simulated': {
            'gaussianreweightadapter__reg': ["auto", 1e-5, 0.5],
        }
    }

    def get_estimator(self):
        # The estimator passed should have a 'predict_proba' method.
        return make_da_pipeline(
            GaussianReweightAdapter(),
            XGBClassifier()
            .set_fit_request(sample_weight=True)
            .set_score_request(sample_weight=True),
        )
