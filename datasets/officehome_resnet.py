from benchopt import BaseDataset, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from sklearn.preprocessing import LabelEncoder
    from sklearn.decomposition import PCA
    from skada.utils import source_target_merge
    from skada.datasets import fetch_office_home_all


# All datasets must be named `Dataset` and inherit from `BaseDataset`
class Dataset(BaseDataset):

    # Name to select the dataset in the CLI and to display the results.
    name = "OfficeHomeResnet"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    # Any parameters 'param' defined here is available as `self.param`.
    parameters = {
        "source_target": [
            ("art", "clipart"),
            ("art", "product"),
            ("art", "realworld"),
            ("clipart", "art"),
            ("clipart", "product"),
            ("clipart", "realworld"),
            ("product", "art"),
            ("product", "clipart"),
            ("product", "realworld"),
            ("realworld", "art"),
            ("realworld", "clipart"),
            ("realworld", "product"),
        ],
    }

    mapping = {
        "art": 1,
        "clipart": 2,
        "product": 3,
        "realworld": 4,
    }

    N_COMPONENTS = 50

    def get_data(self):
        # The return arguments of this function are passed as keyword arguments
        # to `Objective.set_data`. This defines the benchmark's
        # API to pass data. It is customizable for each benchmark.

        tmp_folder = "./data/office_home_resnet/"
        dataset = fetch_office_home_all(data_home=tmp_folder)
        X, y, sample_domain = dataset.pack(
            as_sources=["art", "clipart", "product", "realworld"]
        )
        # Apply PCA to reduce the dimensionality of the embeddings
        pca = PCA(n_components=self.N_COMPONENTS)
        X = pca.fit_transform(X)
        source = self.source_target[0]
        target = self.source_target[1]

        X_source, y_source = (
            X[sample_domain == self.mapping[source]],
            y[sample_domain == self.mapping[source]],
        )
        X_target, y_target = (
            X[sample_domain == self.mapping[target]],
            y[sample_domain == self.mapping[target]],
        )

        # XGBoost only supports labels in [0, num_classes-1]
        le = LabelEncoder()
        le.fit(np.concatenate([y_source, y_target]))
        y_source = le.transform(y_source)
        y_target = le.transform(y_target)

        X, y, sample_domain = source_target_merge(
            X_source, X_target, y_source, y_target
        )
        return dict(
            X=X,
            y=y,
            sample_domain=sample_domain,
        )
