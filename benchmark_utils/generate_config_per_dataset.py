from os import walk
import os
import importlib.util
import sys
import yaml

PATH = os.path.dirname(os.path.dirname(__file__))

if __name__ == "__main__":

    sys.path.append(PATH)

    filenames = next(walk(os.path.join(PATH, "solvers")), (None, None, []))[2]

    dataset_list = []

    filenames_dataset = next(walk(os.path.join(PATH, "datasets")), (None, None, []))[2]

    for name in filenames_dataset:
        spec = importlib.util.spec_from_file_location(name, os.path.join(PATH, "datasets", name))
        foo = importlib.util.module_from_spec(spec)
        sys.modules[name] = foo
        spec.loader.exec_module(foo)
        dataset_list.append(foo.Dataset.name)

    with open(os.path.join(PATH, "config", "best_base_estimators.yml")) as stream:
        best_base_estimators = yaml.safe_load(stream)

    for dataset in dataset_list:

        best = best_base_estimators[dataset]["Best"]
        bestSVC = best_base_estimators[dataset]["BestSVC"]
        
        DD = {}
        DD["dataset"] = [dataset]
        DD["solver"] = []

        for i, name in enumerate(filenames):
            print(name, i)
            spec = importlib.util.spec_from_file_location(name, os.path.join(PATH, "solvers", name))
            foo = importlib.util.module_from_spec(spec)
            sys.modules[name] = foo
            spec.loader.exec_module(foo)
        
            if foo.Solver.name == "JDOT_SVC":
                param_grid = foo.Solver.default_param_grid
                param_grid['jdotclassifier__base_estimator__estimator_name'] = [bestSVC]
            
            elif foo.Solver.name == "DASVM":
                param_grid = foo.Solver.default_param_grid
                param_grid['dasvmclassifier__base_estimator__estimator_name'] = [bestSVC]
            
            else:
                param_grid = foo.Solver.default_param_grid
                if isinstance(param_grid, list):
                    for i in range(len(param_grid)):
                        param_grid[i]['finalestimator__estimator_name'] = [best]
                else:
                    param_grid['finalestimator__estimator_name'] = [best]
                
            DD["solver"].append({foo.Solver.name: {"param_grid": [param_grid]}})

        with open(os.path.join(PATH, "config", "datasets", "%s.yml"%dataset), 'w+') as ff:
            yaml.dump(DD, ff, default_flow_style=False)