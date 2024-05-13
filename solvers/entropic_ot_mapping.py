from benchopt import safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from skada import EntropicOTMappingAdapter, make_da_pipeline
    from benchmark_utils.base_solver import DASolver, FinalEstimator


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class Solver(DASolver):
    # Name to select the solver in the CLI and to display the results.
    name = 'entropic_ot_mapping'

    # List of parameters for the solver. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    param_grid = {
        'entropicotmappingadapter__reg_e': [1, 10],
        'entropicotmappingadapter__metric': ['sqeuclidean'],
        'entropicotmappingadapter__norm': [None, 'median', 'max'],
        'entropicotmappingadapter__max_iter': [1000],
        'entropicotmappingadapter__tol': [10e-9],
        'finalestimator__estimator_name': ["LR", "SVC", "SVC_mnist_usps", "XGB"],
    }

    def get_estimator(self):
        # The estimator passed should have a 'predict_proba' method.
        return make_da_pipeline(
            EntropicOTMappingAdapter(),
            FinalEstimator(),
        )
