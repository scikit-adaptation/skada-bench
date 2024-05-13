from benchopt import BaseDataset, safe_import_context


# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import numpy as np
    import pickle

    from skada.utils import source_target_merge


# All datasets must be named `Dataset` and inherit from `BaseDataset`
class Dataset(BaseDataset):

    # Name to select the dataset in the CLI and to display the results.
    name = "mnist_usps"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    # Any parameters 'param' defined here is available as `self.param`.
    parameters = {
        'n_samples_source, n_samples_target': [(10000, 10000)],
        'source_target': [('MNIST', 'USPS'),
                          ('USPS', 'MNIST')]
        'random_state': [27],
    }

    def _get_dataset(self, dataset_name, n_samples=None):
        rng = np.random.RandomState(self.random_state)

        dataset_name = dataset_name.lower()

        # Load preprocessed data
        with open(f'data/digit.pkl', 'rb') as f:
            data = pickle.load(f)

        # Sample data
        X, y = data[dataset_name]['X'], data[dataset_name]['y']
        if (n_samples is not None) and (n_samples < len(X)):
            indices = rng.choice(len(X), n_samples, replace=False)
            X, y = X[indices], y[indices]

        return X, y

    def get_data(self):
        # The return arguments of this function are passed as keyword arguments
        # to `Objective.set_data`. This defines the benchmark's
        # API to pass data. It is customizable for each benchmark.

        # Generate pseudorandom data using `numpy`.
        source, target = self.source_target
        X_source, y_source = self._get_dataset(source, self.n_samples_source)
        X_target, y_target = self._get_dataset(target, self.n_samples_target)

        X, y, sample_domain = source_target_merge(
            X_source, X_target, y_source, y_target)

        return dict(
            X=X,
            y=y,
            sample_domain=sample_domain
        )
