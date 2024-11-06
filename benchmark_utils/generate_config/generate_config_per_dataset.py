from pathlib import Path
import importlib.util
import sys
import yaml

PATH_benchmark_utils = Path(__file__).resolve().parents[1]
PATH_skada_bench = Path(__file__).resolve().parents[2]

sys.path.extend([str(PATH_benchmark_utils), str(PATH_skada_bench)])

if __name__ == "__main__":
    solvers_path = PATH_skada_bench / "solvers"
    datasets_path = PATH_skada_bench / "datasets"
    config_path = PATH_skada_bench / "config"

    filenames = [
        f for f in solvers_path.iterdir()
        if f.is_file() and not f.name.startswith('.') and f.suffix == '.py'
    ]

    dataset_list = []

    filenames_dataset = [
        f for f in datasets_path.iterdir()
        if f.is_file() and not f.name.startswith('.') and f.suffix == '.py'
    ]

    for filepath in filenames_dataset:
        name = filepath.stem  # Remove the .py suffix
        if name.startswith('deep'):
            # We dont want to include the deep datasets
            continue
        spec = importlib.util.spec_from_file_location(name, filepath)
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

        for filepath in filenames:
            name = filepath.stem  # Remove the .py suffix
            if name.startswith('deep'):
                # We dont want to include the deep datasets
                continue
            print(name)
            spec = importlib.util.spec_from_file_location(name, filepath)
            foo = importlib.util.module_from_spec(spec)
            sys.modules[name] = foo
            spec.loader.exec_module(foo)

            if foo.Solver.name == "JDOT_SVC":
                param_grid = foo.Solver.default_param_grid
                param_grid['jdotclassifier__base_estimator__estimator_name'] = [bestSVC]  # noqa: E501

            elif foo.Solver.name == "DASVM":
                param_grid = foo.Solver.default_param_grid
                param_grid['dasvmclassifier__base_estimator__estimator_name'] = [bestSVC]  # noqa: E501

            else:
                param_grid = foo.Solver.default_param_grid
                if isinstance(param_grid, list):
                    for i in range(len(param_grid)):
                        param_grid[i]['finalestimator__estimator_name'] = [best]  # noqa: E501
                else:
                    param_grid['finalestimator__estimator_name'] = [best]

            DD["solver"].append({
                foo.Solver.name: {"param_grid": [param_grid]}
            })

        with open(config_path / "datasets" / f"{dataset}.yml", 'w+') as ff:
            yaml.dump(DD, ff, default_flow_style=False)
