from benchopt import BaseSolver, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from sklearn.model_selection import GridSearchCV
    from skada.metrics import SupervisedScorer
    from skada.metrics import PredictionEntropyScorer

class DASolver(BaseSolver):
    strategy = "run_once"

    criterions = {
        'supervised': SupervisedScorer(),
        'prediction_entropy': PredictionEntropyScorer()
    }

    def get_estimator(self):
        """Return an estimator compatible with the `sklearn.GridSearchCV`."""
        pass


    def set_objective(self, X, y, sample_domain):
        self.X, self.y, self.sample_domain = X, y, sample_domain

        self.base_estimator = self.get_estimator()
        self.clf = GridSearchCV(
            self.base_estimator, self.param_grid, refit=False,
            scoring=criterions
        )


    def run(self, n_iter):
        self.cfl.fit(self.X, self.y, sample_domain=self.sample_domain)

        self.cv_results_ = self.clf.cv_results_
        self.dict_estimators_ = {}
        for criterion in self.criterions:
            best_index = np.argmax(self.clf.cv_results_['mean_test_' + criterion])
            best_params = self.clf.cv_results_['params'][best_index]
            refit_estimator = self.base_estimator.clone()
            refit_estimator.set_params(**best_params)
            refit_estimator.fit(self.X, self.y, sample_domain=self.sample_domain)
            self.dict_estimators_[criterion] = refit_estimator

    def get_result(self):
        return dict(
            cv_result=self.cv_results_,
            dict_estimators=self.dict_estimators_
        )


