from benchopt import safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from benchmark_utils.base_solver import DASolver
    from benchmark_utils.utils import get_deep_model
    from torch.optim import Adadelta
    from skorch.callbacks import LRScheduler
    from skada.metrics import SupervisedScorer, DeepEmbeddedValidation
    from skada.deep import TargetOnly


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class Solver(DASolver):
    # Name to select the solver in the CLI and to display the results.
    name = 'Deep_NO_DA_TARGET_ONLY'

    def get_estimator(self, n_classes, device, dataset_name, **kwargs):
        # For testing purposes, we use the following criterions:
        self.criterions = {
            'supervised': SupervisedScorer(),
            'deep_embedded_validation': DeepEmbeddedValidation(),
        }

        dataset_name = dataset_name.split("[")[0].lower()

        model = get_deep_model(
            dataset_name, n_classes
        )

        lr_scheduler = LRScheduler(
            policy='StepLR',
            step_every='epoch',
            step_size=1,
            gamma=0.7
        )

        net = TargetOnly(
            **model,
            optimizer=Adadelta,
            layer_name="feature_layer",
            train_split=None,
            device=device,
            callbacks=[lr_scheduler],
        )

        return net

