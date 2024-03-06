from benchopt import BaseDataset, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from skada.utils import source_target_merge
    import pandas as pd


# All datasets must be named `Dataset` and inherit from `BaseDataset`
class Dataset(BaseDataset):

    # Name to select the dataset in the CLI and to display the results.
    name = "CTScan"

    install_cmd = 'conda'

    requirements = ['pandas']

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    # Any parameters 'param' defined here is available as `self.param`.
    parameters = {
        'n_samples_source': [5000],
        'target_id': [46],
        'random_state': [27],
    }

    def get_data(self):
        # The return arguments of this function are passed as keyword arguments
        # to `Objective.set_data`. This defines the benchmark's
        # API to pass data. It is customizable for each benchmark.

        url = ("https://archive.ics.uci.edu/ml/machine-learning-databases"
               "/00206/slice_localization_data.zip")

        df = pd.read_csv(url)

        X = df.drop(["patientId", "reference"], axis=1)
        y = df["reference"]

        source_ids = [i for i in range(97) if i != self.target_id]
        target_ids = [self.target_id]

        X_source = X.loc[df.patientId.isin(source_ids)]
        X_target = X.loc[df.patientId.isin(target_ids)]

        y_source = y.loc[df.patientId.isin(source_ids)]
        y_target = y.loc[df.patientId.isin(target_ids)]

        X_source = X_source.sample(self.n_samples_source,
                                   replace=False,
                                   random_state=self.random_state)
        y_source = y_source.loc[X_source.index]

        X_source = X_source.values
        X_target = X_target.values
        y_source = (y_source.values / 50.) - 1.
        y_target = (y_target.values / 50.) - 1.

        X, y, sample_domain = source_target_merge(
            X_source, X_target, y_source, y_target)

        return dict(
            X=X,
            y=y,
            sample_domain=sample_domain
        )
