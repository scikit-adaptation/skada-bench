from benchopt import safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from benchmark_utils.base_solver import DASolver
    from benchmark_utils.utils import get_model_and_batch_size
    from skada.deep import DeepJDOT, DeepJDOTLoss
    from torch.optim import Adadelta
    from skorch.callbacks import LRScheduler
    from skada.metrics import SupervisedScorer, DeepEmbeddedValidation


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class Solver(DASolver):
    # Name to select the solver in the CLI and to display the results.
    name = 'DeepJDOT'

    # List of parameters for the solver. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    default_param_grid = {
        'max_epochs': [14],
        'lr': [1],
        'criterion__adapt_criterion': [DeepJDOTLoss(reg_cl=1e-4, reg_dist=1e-3)],
    }

    def get_estimator(self, n_classes, device, dataset_name, **kwargs):
        # For testing purposes, we use the following criterions:
        self.criterions = {
            'supervised': SupervisedScorer(),
            'deep_embedded_validation': DeepEmbeddedValidation(),
        }

        dataset_name = dataset_name.split("[")[0].lower()

        model, batch_size = get_model_and_batch_size(
            dataset_name, n_classes)
        
        # For DeepJDOT, we override the default batch size
        batch_size = 1000

        lr_scheduler = LRScheduler(
            policy='StepLR',
            step_every='epoch',
            step_size=1,
            gamma=0.7
        )

        net = DeepJDOT(
            model,
            optimizer=Adadelta,
            layer_name="feature_layer",
            batch_size=batch_size,
            train_split=None,
            device=device,
            callbacks=[lr_scheduler],
        )

        return net
