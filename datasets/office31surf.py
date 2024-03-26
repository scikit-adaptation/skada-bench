from benchopt import BaseDataset, safe_import_context
with safe_import_context() as import_ctx:
    import scipy
    import numpy as np
    from sklearn.preprocessing import LabelEncoder
    from skada.utils import source_target_merge
from sklearn.decomposition import PCA


# All datasets must be named `Dataset` and inherit from `BaseDataset`
class Dataset(BaseDataset):

    # Name to select the dataset in the CLI and to display the results.
    name = "Office31SURF"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    # Any parameters 'param' defined here is available as `self.param`.
    parameters = {
        'source_target': [
            ('dslr', 'caltech'),
            ('dslr', 'webcam'),
            ('dslr', 'amazon'),
            ('caltech', 'dslr'),
            ('caltech', 'webcam'),
            ('caltech', 'amazon'),
            ('webcam', 'dslr'),
            ('webcam', 'caltech'),
            ('webcam', 'amazon'),
            ('amazon', 'dslr'),
            ('amazon', 'caltech'),
            ('amazon', 'webcam')
        ],
    }

    def get_data(self):
        # The return arguments of this function are passed as keyword arguments
        # to `Objective.set_data`. This defines the benchmark's
        # API to pass data. It is customizable for each benchmark.

        datasets_path = './data/OFFICE_31_SURF_DATASET/'
        mat_dslr = scipy.io.loadmat(datasets_path+'dslr_SURF_L10.mat')
        mat_amazon = scipy.io.loadmat(datasets_path+'amazon_SURF_L10.mat')
        mat_webcam = scipy.io.loadmat(datasets_path+'webcam_SURF_L10.mat')
        mat_caltech = scipy.io.loadmat(datasets_path+'Caltech10_SURF_L10.mat')

        mat_dict = {
            'dslr': mat_dslr,
            'amazon': mat_amazon,
            'webcam': mat_webcam,
            'caltech': mat_caltech
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
        
        # PCA explained variance: 0.9532482624053955
        # From 800 reduced to 420
        pca = PCA(n_components=420)
        X = pca.fit_transform(X)

        return dict(
            X=X,
            y=y,
            sample_domain=sample_domain,
        )
