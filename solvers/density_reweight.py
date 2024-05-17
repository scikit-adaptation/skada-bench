from benchopt import safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from skada import DensityReweightAdapter, make_da_pipeline
    from benchmark_utils.base_solver import DASolver, FinalEstimator
    from sklearn.neighbors import KernelDensity


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class Solver(DASolver):
    # Name to select the solver in the CLI and to display the results.
    name = 'density_reweight'

    # List of parameters for the solver. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    param_grid = {
    'densityreweightadapter__weight_estimator': [
        KernelDensity(bandwidth=0.01),
        KernelDensity(bandwidth=0.1),
        KernelDensity(bandwidth=1.),
        KernelDensity(bandwidth=10.),
        KernelDensity(bandwidth=100.)
    ],
    'finalestimator__estimator_name': ["LR", "SVC", "SVC_mnist_usps", "XGB"],
    }

    def get_estimator(self):
        # The estimator passed should have a 'predict_proba' method.
        return make_da_pipeline(
            DensityReweightAdapter(),
            FinalEstimator(),
        )
