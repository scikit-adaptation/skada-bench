from benchopt import safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from skada import JDOTClassifier, make_da_pipeline
    from benchmark_utils.base_solver import DASolver, FinalEstimator
    from sklearn.svm import SVC


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class Solver(DASolver):

    # Name to select the solver in the CLI and to display the results.
    name = 'JDOT_SVC'

    requirements = [
        "pip:POT",
    ]

    # List of parameters for the solver. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    default_param_grid = {
        'jdotclassifier__alpha': [0.1, 0.3, 0.5, 0.7, 0.9],
        'jdotclassifier__n_iter_max': [100],
        'jdotclassifier__tol': [1e-6],
        'jdotclassifier__thr_weights': [1e-7],
        'jdotclassifier__base_estimator__estimator_name': ["SVC"],
    }

    def get_estimator(self):
        # The estimator passed should have a 'predict_proba' method.
        return make_da_pipeline(
            JDOTClassifier(base_estimator=FinalEstimator(),
                           metric='hinge')
            .set_fit_request(sample_weight=True)
            .set_score_request(sample_weight=True),
        )
