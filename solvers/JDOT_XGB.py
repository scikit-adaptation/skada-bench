from benchopt import safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from skada import JDOTClassifier, make_da_pipeline
    from benchmark_utils.base_solver import DASolver
    from xgboost import XGBClassifier


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class Solver(DASolver):

    # Name to select the solver in the CLI and to display the results.
    name = 'JDOT_XGB'

    requirements = [
        "pip:xgboost",
        "pip:POT",
    ]

    # List of parameters for the solver. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    param_grid = {
        'jdotclassifier__alpha': [0.5, 0.7, 0.9],
        'jdotclassifier__n_iter_max': [100, 200, 300],
        'jdotclassifier__tol': [1e-5, 1e-6],
        'jdotclassifier__thr_weights': [1e-6, 1e-7],
    }

    def skip(self, X, y, sample_domain, unmasked_y_train, dataset_name):
        datasets_to_avoid = [
            '20NewsGroups',
        ]

        if dataset_name.split("[")[0] in datasets_to_avoid:
            return True, f"solver does not support the dataset {dataset_name}."

        return False, None

    def get_estimator(self):
        # The estimator passed should have a 'predict_proba' method.
        return make_da_pipeline(
            JDOTClassifier(base_estimator=XGBClassifier())
            .set_fit_request(sample_weight=True)
            .set_score_request(sample_weight=True),
        )
