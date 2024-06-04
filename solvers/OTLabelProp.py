from benchopt import safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from skada import OTLabelPropAdapter, make_da_pipeline
    from benchmark_utils.base_solver import DASolver, FinalEstimator

# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class Solver(DASolver):

    # Name to select the solver in the CLI and to display the results.
    name = 'OTLabelProp'

    # List of parameters for the solver. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    default_param_grid = [
        {
            'otlabelpropadapter__metric': ['sqeuclidean', 'cosine', 'cityblock'],
            'otlabelpropadapter__reg': [None],
            'otlabelpropadapter__n_iter_max': [10000],
            'finalestimator__estimator_name': ["LR", "SVC", "XGB"],
        },
        {
            'otlabelpropadapter__metric': ['sqeuclidean', 'cosine', 'cityblock'],
            'otlabelpropadapter__reg': [0.1, 1],
            'otlabelpropadapter__n_iter_max': [100],
            'finalestimator__estimator_name': ["LR", "SVC", "XGB"],
        },
    ]

    def get_estimator(self):
        # The estimator passed should have a 'predict_proba' method.
        return make_da_pipeline(
            OTLabelPropAdapter(),
            FinalEstimator(),
        )
