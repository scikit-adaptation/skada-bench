from benchopt import BaseDataset, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from sklearn.preprocessing import LabelEncoder
    from skada.utils import source_target_merge
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    from benchmark_utils.utils import (
        download_and_extract_zipfile, ImageDataset
    )


# All datasets must be named `Dataset` and inherit from `BaseDataset`
class Dataset(BaseDataset):
    # Name to select the dataset in the CLI and to display the results.
    name = "Office31"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    # Any parameters 'param' defined here is available as `self.param`.
    parameters = {
        "source_target": [
            ("dslr", "webcam"),
            ("dslr", "amazon"),
            ("webcam", "dslr"),
            ("webcam", "amazon"),
            ("amazon", "dslr"),
            ("amazon", "webcam"),
        ],
    }

    # Dataset classification variables
    compatible_model_types = ["deep"]

    path_dataset = "data/OFFICE31.zip"
    path_extract = "data/OFFICE31/"
    url_dataset = "https://wjdcloud.blob.core.windows.net/dataset/OFFICE31.zip"

    def _get_dataset(self, domain_select):
        # Define transformations to preprocess the images
        preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # Create a DataLoader for the dataset
        dataset = ImageDataset(
            self.path_extract, transform=preprocess,
            domain_select=domain_select
        )
        dataloader = DataLoader(
            dataset, batch_size=len(dataset), shuffle=False
        )

        images, labels = next(iter(dataloader))
        images = images.numpy()
        labels = np.array(labels)

        return images, labels

    def get_data(self):
        # The return arguments of this function are passed as keyword arguments
        # to `Objective.set_data`. This defines the benchmark's
        # API to pass data. It is customizable for each benchmark.
        download_and_extract_zipfile(
            url_dataset=self.url_dataset,
            path_dataset=self.path_dataset,
            path_extract=self.path_extract,
        )

        source, target = self.source_target
        X_source, y_source = self._get_dataset(source)
        X_target, y_target = self._get_dataset(target)

        # XGBoost only supports labels in [0, num_classes-1]
        le = LabelEncoder()
        le.fit(np.concatenate([y_source, y_target]))
        y_source = le.transform(y_source)
        y_target = le.transform(y_target)

        X, y, sample_domain = source_target_merge(
            X_source, X_target, y_source, y_target
        )

        print(f"Office31 mean {X.mean()}")
        print(f"Office31 std {X.std()}")

        return dict(
            X=X,
            y=y,
            sample_domain=sample_domain,
        )
