from benchopt import safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from skada import TransferComponentAnalysisAdapter, make_da_pipeline
    from benchmark_utils.base_solver import DASolver, FinalEstimator


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class Solver(DASolver):
    # Name to select the solver in the CLI and to display the results.
    name = 'transfer_component_analysis'

    # List of parameters for the solver. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    param_grid = {
        'transfercomponentanalysisadapter__kernel': ['rbf'],
        'transfercomponentanalysisadapter__n_components': [1, 2, 3],
        'transfercomponentanalysisadapter__mu': [0.01, 0.1],
        'finalestimator__estimator_name': ["LR", "SVC", "XGB"],
    }

    def skip(self, X, y, sample_domain, unmasked_y_train, dataset_name):
         datasets_to_avoid = [
             'Digit',
         ]

         if dataset_name.split("[")[0] in datasets_to_avoid:
             return True, f"solver does not support the dataset {dataset_name}."

         return False, None

    def get_estimator(self):
        # The estimator passed should have a 'predict_proba' method.
        return make_da_pipeline(
            TransferComponentAnalysisAdapter(),
            FinalEstimator(),
        )
