from benchopt import BaseObjective, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from skada.model_selection import (StratifiedDomainShuffleSplit,
                                       DomainShuffleSplit
                                       )
    from skada.utils import extract_source_indices, source_target_split
    from skada._utils import Y_Type, _find_y_type
    from skada._utils import (_DEFAULT_MASKED_TARGET_CLASSIFICATION_LABEL,
                              _DEFAULT_MASKED_TARGET_REGRESSION_LABEL
                              )
    from sklearn.metrics import (accuracy_score,
                                 balanced_accuracy_score,
                                 f1_score,
                                 roc_auc_score
                                 )
    import numpy as np

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
    min_benchopt_version = "1.5"

    metrics = {
        'accuracy': accuracy_score,
        'balanced_accuracy': balanced_accuracy_score,
        'f1_score': f1_score,
        'roc_auc_score': roc_auc_score,
    }

    def set_data(self, X, y, sample_domain, dataset_name):
        # The keyword arguments of this function are the keys of the dictionary
        # returned by `Dataset.get_data`. This defines the benchmark's
        # API to pass data. This is customizable for each benchmark.
        self.X, self.y, self.sample_domain = X, y, sample_domain
        self.dataset_name = dataset_name

        # check y is discrete or continuous
        self.is_discrete = _find_y_type(self.y) == Y_Type.DISCRETE

        if self.is_discrete:
            self.cv = StratifiedDomainShuffleSplit(
                n_splits=5,
                test_size=0.2
            )
        else:
            # We cant use StratifiedDomainShuffleSplit if y is continuous
            self.cv = DomainShuffleSplit(
                n_splits=5,
                test_size=0.2
            )

    def evaluate_result(self, cv_results, dict_estimators):
        # The keyword arguments of this function are the keys of the
        # dictionary returned by `Solver.get_result`. This defines the
        # benchmark's API to pass solvers' result. This is customizable for
        # each benchmark.

        # This method can return many metrics in a dictionary. One of these
        # metrics needs to be `value` for convergence detection purposes.

        # Train target-source split
        (X_train_source, X_train_target,
         y_train_source, y_train_target
         ) = source_target_split(self.X_train,
                                 self.y_train,
                                 sample_domain=self.sample_domain_train
                                 )

        # Test target-source split
        (X_test_source, X_test_target,
         y_test_source, y_test_target
         ) = source_target_split(self.X_test,
                                 self.y_test,
                                 sample_domain=self.sample_domain_test
                                 )

        all_metrics = {}

        for criterion, estimator in dict_estimators.items():
            # TRAIN metrics
            # Source accuracy
            y_pred_train_source = estimator.predict(X_train_source)
            y_pred_train_source_proba = estimator.predict_proba(X_train_source)

            # Target accuracy
            y_pred_train_target = estimator.predict(X_train_target)
            y_pred_train_target_proba = estimator.predict_proba(X_train_target)

            # TEST metrics
            # Source accuracy
            y_pred_test_source = estimator.predict(X_test_source)
            y_pred_test_source_proba = estimator.predict_proba(X_test_source)

            # Target accuracy
            y_pred_test_target = estimator.predict(X_test_target)
            y_pred_test_target_proba = estimator.predict_proba(X_test_target)

            for metric_name, metric in self.metrics.items():
                if metric_name == 'roc_auc_score':
                    if len(
                        np.unique(np.concatenate((self.y_train, self.y_test)))
                    ) > 2:
                        roc_args = {'multi_class': 'ovo', 'labels': np.unique(
                            np.concatenate((self.y_train, self.y_test)))}

                        all_metrics.update({
                            f'{criterion}_train_source_{metric_name}':
                                metric(
                                    y_train_source,
                                    y_pred_train_source_proba,
                                    **roc_args
                                ),
                            f'{criterion}_train_target_{metric_name}':
                                metric(
                                    y_train_target,
                                    y_pred_train_target_proba,
                                    **roc_args
                                ),
                            f'{criterion}_test_source_{metric_name}':
                                metric(y_test_source,
                                       y_pred_test_source_proba,
                                       **roc_args
                                       ),
                            f'{criterion}_test_target_{metric_name}':
                                metric(y_test_target,
                                       y_pred_test_target_proba,
                                       **roc_args
                                       ),
                        })
                        continue

                f1_args = {}
                if metric_name == 'f1_score':
                    f1_args = {'average': 'weighted'}

                all_metrics.update({
                    f'{criterion}_train_source_{metric_name}':
                        metric(y_train_source, y_pred_train_source, **f1_args),
                    f'{criterion}_train_target_{metric_name}':
                        metric(y_train_target, y_pred_train_target, **f1_args),
                    f'{criterion}_test_source_{metric_name}':
                        metric(y_test_source, y_pred_test_source, **f1_args),
                    f'{criterion}_test_target_{metric_name}':
                        metric(y_test_target, y_pred_test_target, **f1_args)
                })

        return dict(
            cv_results=cv_results,
            **all_metrics,
            value=1e-7,
        )

    def get_one_result(self):
        # Return one solution. The return value should be an object compatible
        # with `self.evaluate_result`. This is mainly for testing purposes.
        return dict(beta=np.zeros(self.X.shape[1]))

    def split(self, cv_fold, X, y, sample_domain):
        id_train, id_test = cv_fold

        self.X_train, self.X_test = X[id_train], X[id_test]
        self.y_train, self.y_test = y[id_train], y[id_test]
        self.sample_domain_train = sample_domain[id_train]
        self.sample_domain_test = sample_domain[id_test]

        # Mask the target in the train to pass to the solver
        y_train = self.y_train.copy()
        id_train_source = extract_source_indices(
            self.sample_domain_train
        )

        y_train[~id_train_source] = (
            _DEFAULT_MASKED_TARGET_CLASSIFICATION_LABEL
            if self.is_discrete else
            _DEFAULT_MASKED_TARGET_REGRESSION_LABEL
        )

        unmasked_y_train = self.y_train
        return (self.X_train, y_train,
                self.sample_domain_train, unmasked_y_train)

    def get_objective(self):
        # Define the information to pass to each solver to run the benchmark.
        # The output of this function are the keyword arguments
        # for `Solver.set_objective`. This defines the
        # benchmark's API for passing the objective to the solver.
        # It is customizable for each benchmark.
        X, y, sample_domain, unmasked_y_train = self.get_split(
            self.X, self.y, self.sample_domain)

        return dict(X=X,
                    y=y,
                    sample_domain=sample_domain,
                    dataset_name=self.dataset_name,
                    unmasked_y_train=unmasked_y_train
                    )
