from benchopt import BaseSolver, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from abc import abstractmethod
    import numpy as np
    from sklearn.base import clone
    from sklearn.model_selection import GridSearchCV
    from skada.metrics import (
        SupervisedScorer,
        PredictionEntropyScorer,
        ImportanceWeightedScorer,
        SoftNeighborhoodDensity,
        DeepEmbeddedValidation,
        CircularValidation,
    )
    from skada.model_selection import (
        StratifiedDomainShuffleSplit,
        DomainShuffleSplit
    )
    from skada._utils import Y_Type, _find_y_type

class DASolver(BaseSolver):
    strategy = "run_once"

    criterions = {
        'supervised': SupervisedScorer(),
        'prediction_entropy': PredictionEntropyScorer(),
        'importance_weighted': ImportanceWeightedScorer(),
        'soft_neighborhood_density': SoftNeighborhoodDensity(),
        'deep_embedded_validation': DeepEmbeddedValidation(),
        'circular_validation': CircularValidation()
    }

    @abstractmethod
    def get_estimator(self):
        """Return an estimator compatible with the `sklearn.GridSearchCV`."""
        pass

    def set_objective(self, X, y, sample_domain, unmasked_y_train):
        self.X, self.y, self.sample_domain = X, y, sample_domain
        self.unmasked_y_train = unmasked_y_train

        self.base_estimator = self.get_estimator()

        # check y is discrete or continuous
        self.is_discrete = _find_y_type(self.y) == Y_Type.DISCRETE

        # CV for the gridsearch
        if self.is_discrete:
            self.gs_cv = StratifiedDomainShuffleSplit(
                n_splits=5,
                test_size=0.2
            )
        else:
            # We cant use StratifiedDomainShuffleSplit if y is continuous
            self.gs_cv = DomainShuffleSplit(
                n_splits=5,
                test_size=0.2
            )

        self.clf = GridSearchCV(
            self.base_estimator, self.param_grid, refit=False,
            scoring=self.criterions, cv=self.gs_cv, error_score=0.0
        )

    def run(self, n_iter):
        if self.name == 'NO_DA_TARGET_ONLY':
            # We are in a case of no domain adaptation
            # We dont need to use masked targets
            self.clf.fit(
                self.X,
                self.unmasked_y_train,
                sample_domain=self.sample_domain,
                target_labels=self.unmasked_y_train,
            )
        else:
            # target_labels here is for the supervised_scorer
            self.clf.fit(
                self.X,
                self.y,
                sample_domain=self.sample_domain,
                target_labels=self.unmasked_y_train
            )

        self.cv_results_ = self.clf.cv_results_
        self.dict_estimators_ = {}
        for criterion in self.criterions:
            best_index = np.argmax(
                self.clf.cv_results_['mean_test_' + criterion]
            )
            best_params = self.clf.cv_results_['params'][best_index]
            refit_estimator = clone(self.base_estimator)
            refit_estimator.set_params(**best_params)

            if self.name == 'NO_DA_TARGET_ONLY':
                refit_estimator.fit(
                    self.X,
                    self.unmasked_y_train,
                    sample_domain=self.sample_domain
                )
            else:
                refit_estimator.fit(
                    self.X, self.y, sample_domain=self.sample_domain
                )
            self.dict_estimators_[criterion] = refit_estimator

    def get_result(self):
        return dict(
            cv_results=self.cv_results_,
            dict_estimators=self.dict_estimators_
        )
