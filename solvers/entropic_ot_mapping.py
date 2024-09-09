from benchopt import safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from skada import EntropicOTMappingAdapter, make_da_pipeline
    from benchmark_utils.base_solver import DASolver, FinalEstimator

    from benchmark_utils.base_solver import import_ctx as base_import_ctx
    if base_import_ctx.failed_import:
        exc, val, tb = base_import_ctx.import_error
        raise exc(val).with_traceback(tb)


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class Solver(DASolver):
    # Name to select the solver in the CLI and to display the results.
    name = 'entropic_ot_mapping'

    # List of parameters for the solver. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    default_param_grid = {
        'entropicotmappingadapter__reg_e': [0.1, 0.5, 1.],
        'entropicotmappingadapter__metric': [
            'sqeuclidean', 'cosine', 'cityblock'
        ],
        'entropicotmappingadapter__norm': ['median'],
        'entropicotmappingadapter__max_iter': [1000],
        'entropicotmappingadapter__tol': [1e-6],
        'finalestimator__estimator_name': ["LR", "SVC", "XGB"],
    }

    def get_estimator(self):
        # The estimator passed should have a 'predict_proba' method.
        return make_da_pipeline(
            EntropicOTMappingAdapter(),
            FinalEstimator(),
        )
