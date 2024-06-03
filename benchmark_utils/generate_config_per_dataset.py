from pathlib import Path
import importlib.util
import sys
import yaml

PATH = Path(__file__).resolve().parents[1]

if __name__ == "__main__":

    sys.path.append(str(PATH))

    solvers_path = PATH / "solvers"
    datasets_path = PATH / "datasets"
    config_path = PATH / "config"

    # find .py files in solvers directory but not hidden files
    filenames = [f for f in solvers_path.iterdir() if f.is_file() and f.suffix == ".py" and not f.name.startswith(".")]

    dataset_list = []

    # find .py files in solvers directory but not hidden files
    filenames_dataset = [f.name for f in datasets_path.iterdir() if f.is_file() and f.suffix == ".py" and not f.name.startswith(".")]

    for name in filenames_dataset:
        spec = importlib.util.spec_from_file_location(name, datasets_path / name)
        foo = importlib.util.module_from_spec(spec)
        sys.modules[name] = foo
        spec.loader.exec_module(foo)
        dataset_list.append(foo.Dataset.name)

    with open(config_path / "best_base_estimators.yml") as stream:
        best_base_estimators = yaml.safe_load(stream)

    for dataset in dataset_list:

        best = best_base_estimators[dataset]["Best"]
        bestSVC = best_base_estimators[dataset]["BestSVC"]

        DD = {}
        DD["dataset"] = [dataset]
        DD["solver"] = []

        for i, name in enumerate(filenames):
            print(name, i)
            spec = importlib.util.spec_from_file_location(name, solvers_path / name)
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

        with open(config_path / "datasets" / f"{dataset}.yml", 'w+') as ff:
            yaml.dump(DD, ff, default_flow_style=False)
