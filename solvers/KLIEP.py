from benchopt import safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from skada import KLIEPReweightAdapter, make_da_pipeline
    from benchmark_utils.base_solver import DASolver


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class Solver(DASolver):

    # Name to select the solver in the CLI and to display the results.
    name = 'KLIEP'
    
    param_grid = {
        'kliepreweightadapter__gamma': [0.1, 1, 10, 'auto', 'scale'],
        'kliepreweightadapter__n_centers': [5, 10, 20],
        'kliepreweightadapter__cv': [5],
        'kliepreweightadapter__tol': [1e-4],
        'kliepreweightadapter__random_state': [0],
    }

    parameters = DASolver.parameters

    def get_estimator(self):
        base_estimator = self.get_base_estimator()
        return make_da_pipeline(
            KLIEPReweightAdapter(gamma=None),
            base_estimator,
        )
