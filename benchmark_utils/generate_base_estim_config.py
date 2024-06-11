import os
import yaml
from os import walk
import importlib.util
import sys
from base_solver import BASE_ESTIMATOR_DICT

PATH = os.path.dirname(os.path.dirname(__file__))

if __name__ == "__main__":

    sys.path.append(PATH)

    param_list = [{"finalestimator__estimator_name": [k]} for k in BASE_ESTIMATOR_DICT]
    param_dict = {"solver": {"NO_DA_SOURCE_ONLY_BASE_ESTIM": {"param_grid": param_list}}}

    dataset_list = []

    filenames_dataset = next(walk(os.path.join(PATH, "datasets")), (None, None, []))[2]

    for name in filenames_dataset:
        spec = importlib.util.spec_from_file_location(name, os.path.join(PATH, "datasets", name))
        foo = importlib.util.module_from_spec(spec)
        sys.modules[name] = foo
        spec.loader.exec_module(foo)
        dataset_list.append(foo.Dataset.name)

    print(dataset_list)
    
    param_dict["dataset"] = dataset_list

    with open(os.path.join(PATH, 'config', 'find_best_base_estimators_per_dataset.yml'), 'w+') as ff:
        yaml.dump(param_dict, ff)