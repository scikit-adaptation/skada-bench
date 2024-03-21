from benchopt import BaseDataset, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import pandas as pd
    from sklearn.preprocessing import OneHotEncoder

    from skada.utils import source_target_merge
    import numpy as np


# All datasets must be named `Dataset` and inherit from `BaseDataset`
class Dataset(BaseDataset):

    # Name to select the dataset in the CLI and to display the results.
    name = "Mushrooms"

    install_cmd = 'conda'

    requirements = ['scikit-learn']

    references = [
        "Dai W., Yang Q., Xue G., and Yu Y, "
        "'Boosting for transfer learning' "
        "ICML, (2007)."
    ]

    parameters = {
    }

    def get_data(self):
        # The return arguments of this function are passed as keyword arguments
        # to `Objective.set_data`. This defines the benchmark's
        # API to pass data. It is customizable for each benchmark.

        url = ("https://archive.ics.uci.edu/ml/machine-learning"
               "-databases/mushroom/agaricus-lepiota.data")

        columns = ["target", "cap-shape", "cap-surface", "cap-color",
                   "bruises", "odor", "gill-attachment", "gill-spacing",
                   "gill-size", "gill-color", "stalk-shape", "stalk-root",
                   "stalk-surface-above-ring", "stalk-surface-below-ring",
                   "stalk-color-above-ring", "stalk-color-below-ring",
                   "veil-type", "veil-color", "ring-number", "ring-type",
                   "spore-print-color", "population", "habitat"]

        data = pd.read_csv(url, header=None)
        data.columns = columns
        X = data.drop(["target"], axis=1)
        y = data["target"]

        X_source = X.loc[X["stalk-shape"] == "t"]
        y_source = y.loc[X_source.index].values
        X_target = X.loc[X["stalk-shape"] == "e"]
        y_target = y.loc[X_target.index].values

        ohe = OneHotEncoder(sparse_output=False).fit(X)
        X_source = ohe.transform(X_source)
        X_target = ohe.transform(X_target)

        X, y, sample_domain = source_target_merge(
            X_source, X_target, y_source, y_target)

        # Mapping from letters to binary
        mapping = {'e': 0, 'p': 1}

        # Applying the mapping function
        y = np.array([mapping[val] for val in y])

        return dict(
            X=X,
            y=y,
            sample_domain=sample_domain,
            dataset_name="mushrooms",
        )
