import yaml
from base_solver import BASE_ESTIMATOR_DICT

param_list = [{"finalestimator__estimator_name": [k]} for k in BASE_ESTIMATOR_DICT]
param_dict = {"solver": {"NO_DA_SOURCE_ONLY_BASE_ESTIM": {"param_grid": param_list}}}

dataset_list = [
    "AmazonReview",
    "BCI",
    "Mushrooms",
    "mnist_usps[n_samples_source=10000,n_samples_target=10000]",
    "mnist_usps[n_samples_source=3000,n_samples_target=3000]",
    "Office31",
    "OfficeHomeResnet",
    "Phishing",
    "20NewsGroups",
    "Simulated[shift=covariate_shift]",
    "Simulated[shift=target_shift]",
    "Simulated[shift=concept_drift]",
    "Simulated[shift=subspace]",
]

param_dict["dataset"] = dataset_list

with open('config/find_best_base_estimators_per_dataset.yml', 'w+') as ff:
    yaml.dump(param_dict, ff)