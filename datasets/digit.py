from benchopt import BaseDataset, safe_import_context


# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import numpy as np
    import torchvision
    from torchvision.datasets import MNIST, SVHN, USPS

    from skada.utils import source_target_merge


# All datasets must be named `Dataset` and inherit from `BaseDataset`
class Dataset(BaseDataset):

    # Name to select the dataset in the CLI and to display the results.
    name = "Digit"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    # Any parameters 'param' defined here is available as `self.param`.
    parameters = {
        'n_samples_source, n_samples_target': [(1000, 1000)],
        'source_target': [('MNIST', 'SVHN'),
                          ('MNIST', 'USPS'),
                          ('SVHN', 'USPS'),
                          ('SVHN', 'MNIST'),
                          ('USPS', 'MNIST'),
                          ('USPS', 'SVHN')],
        'random_state': [27],
    }

    def _get_dataset(self, dataset_name, n_samples=None):
        rng = np.random.RandomState(self.random_state)

        dataset_name = dataset_name.lower()
        if dataset_name == 'mnist':
            transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Pad(2),
                torchvision.transforms.Normalize((0.1307,), (0.3081,))
            ])
            dataset = MNIST(
                root='./data/MNIST',
                download=True,
                train=True,
                transform=transform
            )
        elif dataset_name == 'svhn':
            transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Grayscale(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,))
            ])
            dataset = SVHN(
                root='./data/SVHN',
                download=True,
                split='train',
                transform=transform
            )
        elif dataset_name == 'usps':
            transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Pad(8),
                torchvision.transforms.Grayscale(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,))
            ])
            dataset = USPS(
                root='./data/USPS',
                download=True,
                train=True,
                transform=transform
            )
        else:
            raise ValueError(f"Unknown dataset {dataset_name}")

        X, y = list(), list()
        for i in range(len(dataset)):
            X.append(dataset[i][0].numpy().reshape(-1, 32*32))
            y.append(dataset[i][1])
        X = np.concatenate(X, axis=0)
        y = np.array(y)
        if n_samples is None:
            n_samples = len(X)
        indices = rng.choice(len(X), n_samples, replace=False)
        X = X[indices]
        y = y[indices]

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
