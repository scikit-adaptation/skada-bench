from benchopt import BaseDataset, safe_import_context


# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import numpy as np
    import torch
    import torchvision
    from torchvision.datasets import MNIST, USPS

    from skada.utils import source_target_merge


# All datasets must be named `Dataset` and inherit from `BaseDataset`
class Dataset(BaseDataset):
    # Name to select the dataset in the CLI and to display the results.
    name = "mnist_usps"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    # Any parameters 'param' defined here is available as `self.param`.
    parameters = {
        "n_samples_source, n_samples_target": [(None, None)],
        "source_target": [("MNIST", "USPS"), ("USPS", "MNIST")],
        "random_state": [27],
    }

    # Dataset classification variables
    compatible_model_types = ["deep"]

    DATASETS = ["MNIST", "USPS"]

    def _download_data(self):
        preprocessed_data = dict()
        for dataset_name in self.DATASETS:
            print(f"Preprocessing {dataset_name} dataset")

            dataset_name = dataset_name.lower()

            if dataset_name == "mnist":
                transform = torchvision.transforms.Compose(
                    [
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                    ]
                )
                dataset = MNIST(
                    root="./data/MNIST",
                    download=True,
                    train=True,
                    transform=transform,
                )
            elif dataset_name == "usps":
                transform = torchvision.transforms.Compose(
                    [
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Pad(6),
                        torchvision.transforms.Grayscale(),
                        torchvision.transforms.Normalize((0.0806,), (0.2063,)),
                    ]
                )
                dataset = USPS(
                    root="./data/USPS",
                    download=True,
                    train=True,
                    transform=transform,
                )
            else:
                raise ValueError(f"Unknown dataset {dataset_name}")

            # Vectorize the images
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=len(dataset),
                shuffle=False,
            )
            embeddings = list()
            with torch.no_grad():
                for images, _ in dataloader:
                    embeddings.append(images)
            embeddings = torch.cat(embeddings, dim=0)
            embeddings = embeddings.cpu().numpy()

            # Save the preprocessed data
            preprocessed_data[dataset_name] = {
                "X": embeddings,
                "y": np.concatenate([y for _, y in dataloader]),
            }

        return preprocessed_data

    def _get_dataset(self, data, dataset_name, n_samples=None):
        rng = np.random.RandomState(self.random_state)

        dataset_name = dataset_name.lower()

        # Sample data
        X, y = data[dataset_name]["X"], data[dataset_name]["y"]
        if (n_samples is not None) and (n_samples < len(X)):
            indices = rng.choice(len(X), n_samples, replace=False)
            X, y = X[indices], y[indices]

        return X, y

    def get_data(self):
        # The return arguments of this function are passed as keyword arguments
        # to `Objective.set_data`. This defines the benchmark's
        # API to pass data. It is customizable for each benchmark.

        # Generate pseudorandom data using `numpy`.
        data = self._download_data()
        source, target = self.source_target
        X_source, y_source = self._get_dataset(
            data, source, self.n_samples_source
        )
        X_target, y_target = self._get_dataset(
            data, target, self.n_samples_target
        )

        print(f"Mnist mean {X_source.mean()}")
        print(f"Mnist std {X_source.std()}")
        print(f"USPS mean {X_target.mean()}")
        print(f"USPS std {X_target.std()}")

        X, y, sample_domain = source_target_merge(
            X_source, X_target, y_source, y_target
        )

        return dict(
            X=X,
            y=y,
            sample_domain=sample_domain,
        )
