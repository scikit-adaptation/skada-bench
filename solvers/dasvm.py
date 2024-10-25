from benchopt import safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import numpy as np
    from skada import DASVMClassifier, make_da_pipeline
    from skada._utils import _DEFAULT_MASKED_TARGET_CLASSIFICATION_LABEL
    from skada.transformers import StratifiedDomainSubsampler
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

    def skip(self, X, y, sample_domain, unmasked_y_train, dataset):
        # First, call the superclass skip method
        skip, msg = super().skip(
            X,
            y,
            sample_domain,
            unmasked_y_train,
            dataset
        )
        if skip:
            return skip, msg

        # Check if the dataset is multiclass, excluding y == -1
        n_classes = len(
            np.unique(y[y != _DEFAULT_MASKED_TARGET_CLASSIFICATION_LABEL])
        )
        if n_classes > 2:
            return True, (f"DASVM does not support multiclass datasets "
                          f"like {dataset.name}.")

        return False, None

    def get_estimator(self, **kwargs):
        # The estimator passed should have a 'predict_proba' method.
        subsampler = StratifiedDomainSubsampler(
            train_size=200
        )

        return make_da_pipeline(
            subsampler,
            DASVMClassifier(base_estimator=FinalEstimator())
            .set_score_request(sample_weight=True)
        )
