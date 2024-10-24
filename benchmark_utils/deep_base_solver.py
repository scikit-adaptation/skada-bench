from benchopt import safe_import_context

from benchmark_utils.base_solver import DASolver

with safe_import_context() as import_ctx:
    import torch


class DeepDASolver(DASolver):
    n_jobs = 1

    requirements = ['pip:skada[deep]']

    # For DeepDA solvers, empty test_param_grid
    test_param_grid = {}

    def __init__(self, **kwargs):
        super().__init__(print_infos=False, **kwargs)

        # Set device depending on the gpu/cpu available
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        print(f"n_jobs: {self.n_jobs}")
        print(f"device: {self.device}")

    # Override the DASolver skip method
    def skip(self, X, y, sample_domain, unmasked_y_train, dataset):
        # Check if the dataset name does not start
        # with 'deep' and is not 'Simulated'
        if not (
            dataset.name.startswith('deep') or
            dataset.name == 'Simulated'
        ):
            return True, f"solver does not support the dataset {dataset.name}."

        return False, None