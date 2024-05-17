from benchopt import safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from skada import KLIEPReweightAdapter, make_da_pipeline
    from benchmark_utils.base_solver import DASolver, FinalEstimator


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class Solver(DASolver):

    # Name to select the solver in the CLI and to display the results.
    name = 'KLIEP'

    param_grid = {
        'kliepreweightadapter__gamma': [0.01, 0.1, 1., 10., 100. 'auto', 'scale'],
        'kliepreweightadapter__n_centers': [100, 1000],
        'kliepreweightadapter__cv': [5],
        'kliepreweightadapter__tol': [1e-6],
        'kliepreweightadapter__max_iter': [1000],
        'kliepreweightadapter__random_state': [0],
        'finalestimator__estimator_name': ["LR", "SVC", "SVC_mnist_usps", "XGB"],
    }

    def get_estimator(self):
        return make_da_pipeline(
            KLIEPReweightAdapter(gamma=None),
            FinalEstimator(),
        )
