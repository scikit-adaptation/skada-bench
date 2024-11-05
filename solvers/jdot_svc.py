from benchopt import safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from skada import JDOTClassifier, make_da_pipeline
    from skada.transformers import StratifiedDomainSubsampler
    from benchmark_utils.base_solver import DASolver, FinalEstimator

    from benchmark_utils.base_solver import import_ctx as base_import_ctx
    if base_import_ctx.failed_import:
        exc, val, tb = base_import_ctx.import_error
        raise exc(val).with_traceback(tb)


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

    test_param_grid = {
        "jdotclassifier__base_estimator__estimator_name": ["SVC"],
        "jdotclassifier__n_iter_max": [10]
    }

    def get_estimator(self, **kwargs):
        # The estimator passed should have a 'predict_proba' method.
        subsampler = StratifiedDomainSubsampler(
            train_size=1000
        )

        return make_da_pipeline(
            subsampler,
            JDOTClassifier(base_estimator=FinalEstimator(),
                           metric='hinge')
            .set_fit_request(sample_weight=True)
            .set_score_request(sample_weight=True),
        )
