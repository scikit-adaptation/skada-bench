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

    filenames = [f for f in solvers_path.iterdir() if f.is_file() and not f.name.startswith('.') and f.suffix == '.py']

    with open(config_path / "best_base_estimators.yml") as stream:
        best_base_estimators = yaml.safe_load(stream)

    for best in ["LR", "SVC", "XGB"]:
        for dataset in ["Simulated"]:

            DD = {}
            DD["dataset"] = [dataset]
            DD["solver"] = []

            for filepath in filenames:
                name = filepath.stem  # Remove the .py suffix
                print(name)
                spec = importlib.util.spec_from_file_location(name, filepath)
                foo = importlib.util.module_from_spec(spec)
                sys.modules[name] = foo
                spec.loader.exec_module(foo)

                if foo.Solver.name != "JDOT_SVC" and foo.Solver.name != "DASVM":
                    param_grid = foo.Solver.default_param_grid
                    if isinstance(param_grid, list):
                        for i in range(len(param_grid)):
                            param_grid[i]['finalestimator__estimator_name'] = [best]
                    else:
                        param_grid['finalestimator__estimator_name'] = [best]

                    DD["solver"].append({foo.Solver.name: {"param_grid": [param_grid]}})

            with open(config_path / f"{dataset}_{best}.yml", 'w+') as ff:
                yaml.dump(DD, ff, default_flow_style=False)
