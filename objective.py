from benchopt import BaseObjective, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from skada.model_selection import StratifiedDomainShuffleSplit


# The benchmark objective must be named `Objective` and
# inherit from `BaseObjective` for `benchopt` to work properly.
class Objective(BaseObjective):

    # Name to select the objective in the CLI and to display the results.
    name = "SKADA Domain Adaptation Benchmark"

    # URL of the main repo for this benchmark.
    url = "https://github.com/scikit-adaptation/skada-bench"

    # List of parameters for the objective. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    # This means the OLS objective will have a parameter `self.whiten_y`.

    # List of packages needed to run the benchmark.
    requirements = [
        'pip:git+https://github.com/scikit-adaptation/skada.git'
    ]

    # Minimal version of benchopt required to run this benchmark.
    # Bump it up if the benchmark depends on a new feature of benchopt.
    min_benchopt_version = "1.6"

    def set_data(self, X, y, sample_domain):
        # The keyword arguments of this function are the keys of the dictionary
        # returned by `Dataset.get_data`. This defines the benchmark's
        # API to pass data. This is customizable for each benchmark.
        self.X, self.y, self.sample_domain = X, y, sample_domain

        self.cv = StratifiedDomainShuffleSplit(
            n_splits=10,
        )

    def evaluate_result(self, cv_grid, model_per_criterion):
        # The keyword arguments of this function are the keys of the
        # dictionary returned by `Solver.get_result`. This defines the
        # benchmark's API to pass solvers' result. This is customizable for
        # each benchmark.
        diff = self.y - self.X @ beta

        # This method can return many metrics in a dictionary. One of these
        # metrics needs to be `value` for convergence detection purposes.
        return dict(
            value=.5 * diff @ diff,
        )

    def get_one_result(self):
        # Return one solution. The return value should be an object compatible
        # with `self.evaluate_result`. This is mainly for testing purposes.
        return dict(beta=np.zeros(self.X.shape[1]))

    def split(X, y, sample_domain, cv_fold):
        id_train, id_test = cv_fold

        self.X_train, self.X_test = X[id_train], X[id_test]
        self.y_train, self.y_test = y[id_train], y[id_test]
        self.sample_domain_train = sample_domain[id_train]
        self.sample_domain_test = sample_domain[id_test]

        # Mask the target in the train to pass to the solver
        y_train = self.y_train.copy()
        id_train_source, id_train_target = extract_domains_indices(
            sample_domain_train
        )
        y_train[id_train_target] = -1
        return self.X_train, y_train, self.sample_domain_train

    def get_objective(self):
        # Define the information to pass to each solver to run the benchmark.
        # The output of this function are the keyword arguments
        # for `Solver.set_objective`. This defines the
        # benchmark's API for passing the objective to the solver.
        # It is customizable for each benchmark.
        X, y, sample_domain = self.get_split(self.X, self.y)

        return dict(X=X, y=y, sample_domain=sample_domain)
