from benchopt import safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from skada import ClassRegularizerOTMappingAdapter, make_da_pipeline
    from benchmark_utils.base_solver import DASolver, FinalEstimator


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class Solver(DASolver):
    # Name to select the solver in the CLI and to display the results.
    name = 'class_regularizer_ot_mapping'

    # List of parameters for the solver. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    param_grid = {
        'classregularizerotmappingadapter__reg_e': [1.0, 0.1],
        'classregularizerotmappingadapter__reg_cl': [0.1, 0.01],
        'classregularizerotmappingadapter__norm': ["lpl1", "l1l2"],
        'classregularizerotmappingadapter__metric': ["sqeuclidean"],
        'classregularizerotmappingadapter__max_iter': [100],
        'classregularizerotmappingadapter__max_inner_iter': [100],
        'classregularizerotmappingadapter__tol': [10e-9],
        'finalestimator__estimator_name': ["LR", "SVC", "XGB"],
    }

    def skip(self, X, y, sample_domain, unmasked_y_train, dataset_name):
         datasets_to_avoid = [
             'mnist_usps',
         ]

         if dataset_name.split("[")[0] in datasets_to_avoid:
             return True, f"solver does not support the dataset {dataset_name}."

         return False, None

    def get_estimator(self):
        # The estimator passed should have a 'predict_proba' method.
        return make_da_pipeline(
            ClassRegularizerOTMappingAdapter(),
            FinalEstimator(),
        )
