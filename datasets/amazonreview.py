from benchopt import BaseDataset, safe_import_context

with safe_import_context() as import_ctx:
    import scipy
    import numpy as np
    from pathlib import Path
    import pickle
    from sklearn.preprocessing import LabelEncoder
    from skada.utils import source_target_merge
    from skada.datasets import fetch_amazon_review_all


# All datasets must be named `Dataset` and inherit from `BaseDataset`
class Dataset(BaseDataset):

    # Name to select the dataset in the CLI and to display the results.
    name = "AmazonReview"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    # Any parameters 'param' defined here is available as `self.param`.
    parameters = {
        "source_target": [
            ("books", "dvd"),
            ("books", "elec"),
            ("books", "kitchen"),
            ("dvd", "books"),
            ("dvd", "elec"),
            ("dvd", "kitchen"),
            ("elec", "books"),
            ("elec", "dvd"),
            ("elec", "kitchen"),
            ("kitchen", "books"),
            ("kitchen", "dvd"),
            ("kitchen", "elec"),
        ],
        "preprocessing": [
            "sentence_transformers"
        ]
    }

    path_preprocessed_data = "data/amazon_review_preprocessed.pkl"
    url_preprocessed_data = "https://figshare.com/ndownloader/files/XXXXX"

    def get_data(self):
        # The return arguments of this function are passed as keyword arguments
        # to `Objective.set_data`. This defines the benchmark's
        # API to pass data. It is customizable for each benchmark.

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
            data_preprocessed = pickle.load(f)

        # Get source and target domains
        source = self.source_target[0]
        target = self.source_target[1]
        X_source = data_preprocessed[source][self.preprocessing]
        X_target = data_preprocessed[target][self.preprocessing]
        y_source = data_preprocessed[source]['y']
        y_target = data_preprocessed[target]['y']

        # Merge source and target data
        X, y, sample_domain = source_target_merge(
            X_source, X_target, y_source, y_target
        )

        return dict(
            X=X,
            y=y,
            sample_domain=sample_domain,
        )
