from benchopt import BaseDataset, safe_import_context
with safe_import_context() as import_ctx:
    import scipy
    import numpy as np
    from sklearn.preprocessing import LabelEncoder
    from skada.utils import source_target_merge


# All datasets must be named `Dataset` and inherit from `BaseDataset`
class Dataset(BaseDataset):

    # Name to select the dataset in the CLI and to display the results.
    name = "AmazonReview"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    # Any parameters 'param' defined here is available as `self.param`.
    parameters = {
        'source_target': [
            ('books', 'dvd'),
            ('books', 'elec'),
            ('books', 'kitchen'),
            ('dvd', 'books'),
            ('dvd', 'elec'),
            ('dvd', 'kitchen'),
            ('elec', 'books'),
            ('elec', 'dvd'),
            ('elec', 'kitchen'),
            ('kitchen', 'books'),
            ('kitchen', 'dvd'),
            ('kitchen', 'elec')
        ],
        "max_features": [5000],
    }

    def get_data(self):
        # The return arguments of this function are passed as keyword arguments
        # to `Objective.set_data`. This defines the benchmark's
        # API to pass data. It is customizable for each benchmark.

        datasets_path = './data/amazon_review/'
        mat_books = scipy.io.loadmat(datasets_path+'books_400.mat')
        mat_dvd = scipy.io.loadmat(datasets_path+'dvd_400.mat')
        mat_elec = scipy.io.loadmat(datasets_path+'elec_400.mat')
        mat_kitchen = scipy.io.loadmat(datasets_path+'kitchen_400.mat')

        mat_dict = {
            'books': mat_books,
            'dvd': mat_dvd,
            'elec': mat_elec,
            'kitchen': mat_kitchen
        }

        source = self.source_target[0]
        target = self.source_target[1]

        # Convert to float32 instead of int8
        X_source = np.array(
            mat_dict[source]['fts'],
            dtype=np.float32
        )
        X_target = np.array(
            mat_dict[target]['fts'],
            dtype=np.float32
        )
        y_source = np.array(
            mat_dict[source]['labels'].flatten(),
            dtype=np.float32
        )
        y_target = np.array(
            mat_dict[target]['labels'].flatten(),
            dtype=np.float32
        )

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
