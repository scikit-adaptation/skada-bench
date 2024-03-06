from benchopt import BaseSolver, safe_import_context
from skada import CORAL
# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import numpy as np

    # import your reusable functions here
    from benchmark_utils import gradient_ols


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class Solver(BaseSolver):

    # Name to select the solver in the CLI and to display the results.
    name = 'CORAL'

    # List of parameters for the solver. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    parameters = { reg : ["auto", 0.0, 0.5]
    }

    # List of packages needed to run the solver. See the corresponding
    # section in objective.py
    requirements = []

    def set_objective(self, X, y, sample_domain):
        # Define the information received by each solver from the objective.
        # The arguments of this function are the results of the
        # `Objective.get_objective`. This defines the benchmark's API for
        # passing the objective to the solver.
        # It is customizable for each benchmark.
        self.X, self.y, self.sample_domain = X, y, sample_domain

    def run(self, n_iter):
        # This is the function that is called to evaluate the solver.
        # It runs the algorithm for a given a number of iterations `n_iter`.
        # You can also use a `tolerance` or a `callback`, as described in
        # https://benchopt.github.io/performance_curves.html


        estimator = CORAl(reg=self.reg)
        estimator.fit(self.X, self.y, self.sample_domain)

        self.estimator_ = estimator

    def get_result(self):
        # Return the result from one optimization run.
        # The outputs of this function is a dictionary which defines the
        # keyword arguments for `Objective.evaluate_result`
        # This defines the benchmark's API for solvers' results.
        # it is customizable for each benchmark.
        dict_valid =  {'something': 0}
        dict_estimators = {'criterion1': self.estimator_}
        return dict(dict_valid=dict_valid, s=dict_estimators)
