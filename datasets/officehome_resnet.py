from benchopt import BaseDataset, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import LabelEncoder
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
        'source_target': [
            ('art', 'clipart'),
            ('art', 'product'),
            ('art', 'realworld'),
            ('clipart', 'art'),
            ('clipart', 'product'),
            ('clipart', 'realworld'),
            ('product', 'art'),
            ('product', 'clipart'),
            ('product', 'realworld'),
            ('realworld', 'art'),
            ('realworld', 'clipart'),
            ('realworld', 'product'),
        ],
        'n_components': [100]
    }


    def get_data(self):
        # The return arguments of this function are passed as keyword arguments
        # to `Objective.set_data`. This defines the benchmark's
        # API to pass data. It is customizable for each benchmark.

        tmp_folder = './data/office_home_resnet/'
        dataset = fetch_office_home_all(
            data_home=tmp_folder
        )

        # Fit PCA on all domains
        domains = dataset.domain_names_.keys()
        X_total = np.concatenate([dataset.get_domain(d)[0] for d in domains])
        pca = PCA(n_components=self.n_components).fit(X_total)

        # Get source and target data and apply PCA
        source = self.source_target[0]
        target = self.source_target[1]

        X_source, y_source = dataset.get_domain(source)
        X_target, y_target = dataset.get_domain(target)
        X_source = pca.transform(X_source)
        X_target = pca.transform(X_target)

        # XGBoost only supports labels in [0, num_classes-1]
        le = LabelEncoder()
        le.fit(np.concatenate([y_source, y_target]))
        y_source = le.transform(y_source)
        y_target = le.transform(y_target)

        X, y, sample_domain = source_target_merge(
            X_source, X_target, y_source, y_target)

        return dict(
            X=X,
            y=y,
            sample_domain=sample_domain,
        )
