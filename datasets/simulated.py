from benchopt import BaseDataset
from skada.datasets import make_shifted_datasets


# All datasets must be named `Dataset` and inherit from `BaseDataset`
class Dataset(BaseDataset):

    # Name to select the dataset in the CLI and to display the results.
    name = "Simulated"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    # Any parameters 'param' defined here is available as `self.param`.
    parameters = {
        'n_samples_source, n_samples_target': [(100, 100)],
        'shift, label': [
            ('covariate_shift', 'binary'),
            ('target_shift', 'binary'),
            ('concept_drift', 'binary'),
            ('subspace', 'binary'),
        ],
        'random_state': list(range(5))
    }

    def get_data(self):
        # The return arguments of this function are passed as keyword arguments
        # to `Objective.set_data`. This defines the benchmark's
        # API to pass data. It is customizable for each benchmark.

        # Generate pseudorandom data using `numpy`.
        if self.shift == "subspace":
            m = 3
            noise = 0.4
        else:
            m = 1
            noise = 0.8
        X, y, sample_domain = make_shifted_datasets(
            n_samples_source=m*self.n_samples_source,
            n_samples_target=m*self.n_samples_target,
            shift=self.shift,
            noise=noise,
            label=self.label,
            random_state=self.random_state,
        )

        return dict(
            X=X,
            y=y,
            sample_domain=sample_domain,
        )
