from benchopt import BaseDataset, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    import urllib.request
    from pathlib import Path
    import pickle
    from skada.utils import source_target_merge


# All datasets must be named `Dataset` and inherit from `BaseDataset`
class Dataset(BaseDataset):

    # Name to select the dataset in the CLI and to display the results.
    name = "AmazonReview"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    # Any parameters 'param' defined here is available as `self.param`.
    parameters = {
        "n_samples_source, n_samples_target": [(2000, 2000)],
        "source_target": [
            ("books", "dvd"),
            ("books", "electronics"),
            ("books", "kitchen"),
            ("dvd", "books"),
            ("dvd", "electronics"),
            ("dvd", "kitchen"),
            ("electronics", "books"),
            ("electronics", "dvd"),
            ("electronics", "kitchen"),
            ("kitchen", "books"),
            ("kitchen", "dvd"),
            ("kitchen", "electronics"),
        ],
        "preprocessing": [
            "sentence_transformers"
        ],
        "random_state": [27],
    }


    path_preprocessed_data = "data/amazon_review_preprocessed.pkl"
    url_preprocessed_data = (
        "https://figshare.com/ndownloader/files/"
        "46763626?private_link=2a46003c38fd015cdbde"
    )

    def _get_dataset(self, dataset_name, n_samples=None):
        rng = np.random.RandomState(self.random_state)

        # Check if the preprocessed data is available
        # If not, download them
        path_preprocessed_data = Path(self.path_preprocessed_data)
        if not path_preprocessed_data.exists():
            urllib.request.urlretrieve(
                self.url_preprocessed_data,
                path_preprocessed_data
            )

        # Load preprocessed data
        with open(path_preprocessed_data, "rb") as f:
            data = pickle.load(f)

        # Sample data
        X = data[dataset_name][self.preprocessing]
        y = data[dataset_name]["y"]
        if (n_samples is not None) and (n_samples < len(X)):
            indices = rng.choice(len(X), n_samples, replace=False)
            X, y = X[indices], y[indices]

        return X, y

    def get_data(self):
        # The return arguments of this function are passed as keyword arguments
        # to `Objective.set_data`. This defines the benchmark's
        # API to pass data. It is customizable for each benchmark.

        source, target = self.source_target
        n_samples_source = self.n_samples_source
        n_samples_target = self.n_samples_target
        X_source, y_source = self._get_dataset(source, n_samples_source)
        X_target, y_target = self._get_dataset(target, n_samples_target)

        # Merge source and target data
        X, y, sample_domain = source_target_merge(
            X_source, X_target, y_source, y_target
        )

        return dict(
            X=X,
            y=y,
            sample_domain=sample_domain,
        )
