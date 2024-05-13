from benchopt import BaseSolver, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from abc import abstractmethod
    import numpy as np
    import os
    from pathlib import Path
    import fcntl
    import pandas as pd
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

    from sklearn.base import BaseEstimator, clone
    from xgboost import XGBClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC


BASE_ESTIMATOR_DICT = {
    "LR": LogisticRegression(),
    "SVC": SVC(probability=True),
    "XGB": XGBClassifier(),
}

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

    def set_objective(
            self,
            X,
            y,
            sample_domain,
            unmasked_y_train, 
            dataset,
            **kwargs
        ):
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

        # To log the experiment in a csv file
        self.dataset = dataset

        # Seems like benchopt executes set_objective() but not run()
        # When the experimenent is already done and cached
        # So we set each exp to finished and if it runs it,
        # it will be set to running
        log_experiment(self.dataset, self.name, 'Finished')

    def run(self, n_iter):
        # Set the experiment to running
        log_experiment(self.dataset, self.name, 'Running')

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
        
        # To log the experiment in a csv file
        log_experiment(self.dataset, self.name, 'Finished')

    def get_result(self):
        return dict(
            cv_results=self.cv_results_,
            dict_estimators=self.dict_estimators_
        )


def log_experiment(dataset, solver, status):
    # Define the directory path
    # This directory will be deleted
    # when doing `benchopt clean`
    directory = "./__cache__/exp_logs"

    # Check if the directory exists, if not, create it
    directory = Path(directory)

    if not directory.exists():
        directory.mkdir(parents=True)

    # Define the file path
    file_name = f"{directory}/{dataset.name}_experiment_log.csv"

    dataset_name = str(dataset)

    # Acquire a file lock
    with open(file_name, 'a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)

        # Read the CSV file into a DataFrame
        # If the file is empty, create an empty DataFrame
        # The error should occur the 1st time only
        # when the csv is still empty
        try:
            df = pd.read_csv(file_name)
        except pd.errors.EmptyDataError:
            df = pd.DataFrame(columns=['Dataset', 'Solver', 'Status'])

        # Update the status of the existing row
        mask = (df['Dataset'] == dataset_name) & (df['Solver'] == solver)

        # If row doesnt exist yet, create it
        if mask.sum() == 0:
            new_row = pd.DataFrame({'Dataset': [dataset_name], 'Solver': [solver], 'Status': [status]})
            df = pd.concat([df, new_row], ignore_index=True)
        else:
            df.loc[mask, 'Status'] = status

        # Write the DataFrame back to the CSV file
        df.to_csv(file_name, index=False)

        # Release the file lock
        fcntl.flock(f, fcntl.LOCK_UN)
