from benchopt import BaseSolver, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from abc import abstractmethod
    import numpy as np
    import os
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

    from sklearn.base import BaseEstimator
    from xgboost import XGBClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier


LR_C_GRID = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1,
             0.2, 0.5, 1., 2., 5., 10., 20.,
             50., 100., 200., 500., 1000.]

SVC_C_GRID = [0.001, 0.01, 0.1, 1., 10., 100., 1000.]
SVC_GAMMA_GRID = [0.001, 0.01, 0.1, 1., 10., 100., 1000.]

XGB_SUBSAMPLE_GRID = [0.5, 0.65, 0.8]
XGB_COLSAMPLE_GRID = [0.5, 0.65, 0.8]
XGB_MAXDEPTH_GRID = [3, 6, 10, 20]


BASE_ESTIMATOR_DICT = {
    "LR": LogisticRegression(),
    "SVC": SVC(probability=True),
    "SVC_mnist_usps": SVC(probability=True, kernel='rbf', C=100, gamma=0.01),
    "XGB": XGBClassifier(),
    "KNN": KNeighborsClassifier()
}

for c in LR_C_GRID:
    k = "LR_C%s"%str(c)
    BASE_ESTIMATOR_DICT[k] = LogisticRegression(C=c)

for c in SVC_C_GRID:
    for gamma in SVC_GAMMA_GRID:
        k = "SVC_C%s_Gamma%s"%(str(c), str(gamma))
        BASE_ESTIMATOR_DICT[k] = SVC(probability=True, kernel="rbf", C=c, gamma=gamma)

for subsample in XGB_SUBSAMPLE_GRID:
    for colsample in XGB_COLSAMPLE_GRID:
        for maxdepth in XGB_MAXDEPTH_GRID:
            k = "XGB_subsample%s_colsample%s_maxdepth%s"%(str(subsample),
                                                          str(colsample),
                                                          str(maxdepth))
            BASE_ESTIMATOR_DICT[k] = XGBClassifier(max_depth=maxdepth,
                                 subsample=subsample,
                                 colsample_bytree=colsample,
                                 colsample_bylevel=colsample,
                                 colsample_bynode=colsample)


class FinalEstimator(BaseEstimator):
    __metadata_request__fit = {"sample_weight": True}
    __metadata_request__score = {"sample_weight": True}

    def __init__(self, estimator_name="LR"):
        self.estimator_name = estimator_name

    def fit(self, X, y, sample_weight=None, **fit_params):
        self.estimator_ = clone(BASE_ESTIMATOR_DICT[self.estimator_name])
        self.estimator_.fit(X, y, sample_weight=sample_weight, **fit_params)
        return self

    def predict(self, X, **predict_params):
        return self.estimator_.predict(X)

    def score(self, X, y, **score_params):
        return self.estimator_.score(X, y, **score_params)

    def predict_proba(self, X, **predict_params):
        return self.estimator_.predict_proba(X, **predict_params)


class DASolver(BaseSolver):
    sampling_strategy = "run_once"

    requirements = [
        "pip:xgboost",
    ]

    criterions = {
        'supervised': SupervisedScorer(),
        'prediction_entropy': PredictionEntropyScorer(),
        'importance_weighted': ImportanceWeightedScorer(),
        'soft_neighborhood_density': SoftNeighborhoodDensity(),
        'deep_embedded_validation': DeepEmbeddedValidation(),
        'circular_validation': CircularValidation()
    }

    # List of parameters for the solver. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    parameters = {
        "param_grid": ["default"]
    }

    # Random state
    random_state = 0
    n_splits_cv = 5
    test_size_cv = 0.2

    if 'SLURM_CPUS_PER_TASK' in os.environ:
        n_jobs = int(os.environ['SLURM_CPUS_PER_TASK'])
    else:
        n_jobs = 1

    @abstractmethod
    def get_estimator(self):
        """Return an estimator compatible with the `sklearn.GridSearchCV`."""
        pass

    # def get_base_estimator(self):
    #     # The estimator passed should have a 'predict_proba' method.
    #     estimator = self.base_estimators_dict[self.base_estimator]
    #     if self.base_estimator_params is not None:
    #         estimator.set_params(**self.base_estimator_params)
    #     estimator.set_fit_request(sample_weight=True)
    #     estimator.set_score_request(sample_weight=True)
    #     return estimator

    def set_objective(self, X, y, sample_domain, unmasked_y_train, **kwargs):
        self.X, self.y, self.sample_domain = X, y, sample_domain
        self.unmasked_y_train = unmasked_y_train

        self.da_estimator = self.get_estimator()

        # check y is discrete or continuous
        self.is_discrete = _find_y_type(self.y) == Y_Type.DISCRETE

        # CV for the gridsearch
        if self.is_discrete:
            self.gs_cv = StratifiedDomainShuffleSplit(
                n_splits=self.n_splits_cv,
                test_size=self.test_size_cv,
                random_state=self.random_state,
            )
        else:
            # We cant use StratifiedDomainShuffleSplit if y is continuous
            self.gs_cv = DomainShuffleSplit(
                n_splits=self.n_splits_cv,
                test_size=self.test_size_cv,
                random_state=self.random_state,
            )

        self.clf = GridSearchCV(
            self.da_estimator, self.param_grid, refit=False,
            scoring=self.criterions, cv=self.gs_cv, error_score=-np.inf,
            n_jobs=self.n_jobs
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
            refit_estimator = clone(self.da_estimator)
            refit_estimator.set_params(**best_params)

            try:
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
            except Exception as e:
                print(f"Error while fitting estimator: {e}")

    def get_result(self):
        return dict(
            cv_results=self.cv_results_,
            dict_estimators=self.dict_estimators_
        )
