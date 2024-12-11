from benchopt import safe_import_context

from benchmark_utils.base_solver import DASolver

with safe_import_context() as import_ctx:
    import torch
    from skada.metrics import (
        SupervisedScorer, DeepEmbeddedValidation,
        PredictionEntropyScorer, ImportanceWeightedScorer,
        SoftNeighborhoodDensity, MixValScorer,
    )


class DeepDASolver(DASolver):
    n_jobs = 1

    requirements = ['pip:skada[deep]==0.4.0']

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

        self.criterions = {
            'supervised': SupervisedScorer(),
            'prediction_entropy': PredictionEntropyScorer(),
            'importance_weighted': ImportanceWeightedScorer(),
            'soft_neighborhood_density': SoftNeighborhoodDensity(),
            'deep_embedded_validation': DeepEmbeddedValidation(),
            'mix_val_both': MixValScorer(ice_type='both'),
            'mix_val_inter': MixValScorer(ice_type='inter'),
            'mix_val_intra': MixValScorer(ice_type='intra'),
        }

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
