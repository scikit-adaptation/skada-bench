from benchopt import safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from benchmark_utils.base_solver import DASolver
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from skada.base import SelectSource
    from sklearn.pipeline import Pipeline
    from xgboost import XGBClassifier


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class Solver(DASolver):

    # Name to select the solver in the CLI and to display the results.
    name = 'NO_DA_SOURCE_ONLY'

    requirements = [
        "pip:xgboost",
    ]

    # List of parameters for the solver. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    param_grid = {
        'classifier': [
            SelectSource(LogisticRegression()
                         .set_score_request(sample_weight=True)),
            SelectSource(SVC(kernel='linear', probability=True)
                         .set_score_request(sample_weight=True)),
            SelectSource(XGBClassifier()
                         .set_score_request(sample_weight=True)),
        ]
    }

    def get_estimator(self):
        # The estimator passed should have a 'predict_proba' method.
        return Pipeline([
            ('classifier', None)
        ])
