from benchopt import safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from skada import DASVMClassifier, make_da_pipeline
    from benchmark_utils.base_solver import DASolver, FinalEstimator

    # Temporary fix to avoid errors, to remove when issue
    from benchmark_utils.base_solver import import_ctx as base_import_ctx
    if base_import_ctx.failed_import:
        exc, val, tb = base_import_ctx.import_error
        raise exc(val).with_traceback(tb)


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class Solver(DASolver):

    # Name to select the solver in the CLI and to display the results.
    name = 'DASVM'

    # List of parameters for the solver. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    default_param_grid = {
        "dasvmclassifier__base_estimator__estimator_name": ["SVC"],
        "dasvmclassifier__max_iter": [200],
    }

    test_param_grid = {
        "dasvmclassifier__base_estimator__estimator_name": ["SVC"],
        "dasvmclassifier__max_iter": [10]
    }

    def skip(self, X, y, sample_domain, unmasked_y_train, dataset_name):
        datasets_to_avoid = [
            "Office31SURF",
            "BCI",
            "Office31",
            "OfficeHomeResnet",
            "mnist_usps",
        ]

        if dataset_name.split("[")[0] in datasets_to_avoid:
            return True, f"solver does not support the dataset {dataset_name}."

        return False, None

    def get_estimator(self, **kwargs):
        # The estimator passed should have a 'predict_proba' method.
        return make_da_pipeline(
            DASVMClassifier(base_estimator=FinalEstimator())
            .set_score_request(sample_weight=True)
        )
