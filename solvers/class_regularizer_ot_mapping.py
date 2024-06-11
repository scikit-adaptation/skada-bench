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
    default_param_grid = [
        {
            'classregularizerotmappingadapter__reg_e': [0.1],
            'classregularizerotmappingadapter__reg_cl': [0.1],
            'classregularizerotmappingadapter__norm': ["lpl1"],
            'classregularizerotmappingadapter__metric': ['sqeuclidean', 'cosine', 'cityblock'],
            'classregularizerotmappingadapter__max_iter': [10],
            'classregularizerotmappingadapter__max_inner_iter': [1000],
            'classregularizerotmappingadapter__tol': [1e-6],
            'finalestimator__estimator_name': ["LR", "SVC", "XGB"],
        },
        {
            'classregularizerotmappingadapter__reg_e': [0.5],
            'classregularizerotmappingadapter__reg_cl': [0.5],
            'classregularizerotmappingadapter__norm': ["lpl1"],
            'classregularizerotmappingadapter__metric': ['sqeuclidean', 'cosine', 'cityblock'],
            'classregularizerotmappingadapter__max_iter': [10],
            'classregularizerotmappingadapter__max_inner_iter': [1000],
            'classregularizerotmappingadapter__tol': [1e-6],
            'finalestimator__estimator_name': ["LR", "SVC", "XGB"],
        },
        {
            'classregularizerotmappingadapter__reg_e': [1.],
            'classregularizerotmappingadapter__reg_cl': [1.],
            'classregularizerotmappingadapter__norm': ["lpl1"],
            'classregularizerotmappingadapter__metric': ['sqeuclidean', 'cosine', 'cityblock'],
            'classregularizerotmappingadapter__max_iter': [10],
            'classregularizerotmappingadapter__max_inner_iter': [1000],
            'classregularizerotmappingadapter__tol': [1e-6],
            'finalestimator__estimator_name': ["LR", "SVC", "XGB"],
        }
    ]

    def get_estimator(self):
        # The estimator passed should have a 'predict_proba' method.
        return make_da_pipeline(
            ClassRegularizerOTMappingAdapter(),
            FinalEstimator(),
        )
