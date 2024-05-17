from benchopt import safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from skada import DiscriminatorReweightAdapter, make_da_pipeline
    from benchmark_utils.base_solver import DASolver, FinalEstimator
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from xgboost import XGBClassifier


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class Solver(DASolver):
    # Name to select the solver in the CLI and to display the results.
    name = 'discriminator_reweight'

    # List of parameters for the solver. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    param_grid = {'finalestimator__estimator_name': ["LR", "SVC", "SVC_mnist_usps", "XGB"]}
    param_grid = {
        'discriminatorreweightadapter__domain_classifier': [
            LogisticRegression(),
            KNeighborsClassifier(),
            SVC(probability=True),
            XGBClassifier(),
        ],
        'finalestimator__estimator_name': ["LR", "SVC", "SVC_mnist_usps", "XGB"],
    }

    def get_estimator(self):
        # The estimator passed should have a 'predict_proba' method.
        return make_da_pipeline(
            DiscriminatorReweightAdapter(),
            FinalEstimator(),
        )
