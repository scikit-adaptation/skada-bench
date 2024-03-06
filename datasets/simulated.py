from benchopt import BaseDataset
from skada.datasets import make_shifted_datasets


# All datasets must be named `Dataset` and inherit from `BaseDataset`
class Dataset(BaseDataset):

    # Name to select the dataset in the CLI and to display the results.
    name = "Simulated_shifts"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    # Any parameters 'param' defined here is available as `self.param`.
    parameters = {
        'shift': [
            'covariate_shift', 'target_shift', 'concept_drift', 'subspace'
        ],
        'n_samples_source, n_samples_target': [(100, 100)],
        'label': ['binary', 'multiclass',],
        'random_state': list(range(10))
    }

    def get_data(self):
        # The return arguments of this function are passed as keyword arguments
        # to `Objective.set_data`. This defines the benchmark's
        # API to pass data. It is customizable for each benchmark.

        # Generate pseudorandom data using `numpy`.
        X, y, sample_domain = make_shifted_datasets(
            n_samples_source=self.n_samples_source,
            n_samples_target=self.n_samples_target,
            shift=self.shift,
            noise=0.3,
            label=self.label,
            random_state=self.random_state,
        )

        return dict(
            X=X,
            y=y,
            sample_domain=sample_domain,
        )
