from benchopt import BaseDataset, safe_import_context
with safe_import_context() as import_ctx:
    import scipy
    from skada.utils import source_target_merge


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

        X_source = mat_dict[source]['fts']
        X_target = mat_dict[target]['fts']
        y_source = mat_dict[source]['labels'].flatten()
        y_target = mat_dict[target]['labels'].flatten()

        X, y, sample_domain = source_target_merge(
            X_source, X_target, y_source, y_target)

        return dict(
            X=X,
            y=y,
            sample_domain=sample_domain,
        )
