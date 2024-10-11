"""
This script imports and initializes various scoring metrics used for
evaluating domain adaptation techniques.

Adding a New Scorer
-------------------

To add a new scorer to the script:

1. **Import the Scorer**: Add the import statement for the new scorer class.
2. **Update `CRITERIONS` Dictionary**: Add a new entry in the `CRITERIONS`
   dictionary with the scorer's name and its initialized instance.

This method ensures that all scorers are organized and easily accessible
for evaluating domain adaptation methods.
"""
from benchopt import safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from skada.metrics import (
        SupervisedScorer,
        PredictionEntropyScorer,
        ImportanceWeightedScorer,
        SoftNeighborhoodDensity,
        DeepEmbeddedValidation,
        CircularValidation,
        MixValScorer,
    )


CRITERIONS = {
    'supervised': SupervisedScorer(),
    'prediction_entropy': PredictionEntropyScorer(),
    'importance_weighted': ImportanceWeightedScorer(),
    'soft_neighborhood_density': SoftNeighborhoodDensity(),
    'deep_embedded_validation': DeepEmbeddedValidation(),
    'circular_validation': CircularValidation(),
    'mix_val_both': MixValScorer(ice_type='both'),
    'mix_val_inter': MixValScorer(ice_type='inter'),
    'mix_val_intra': MixValScorer(ice_type='intra'),
}
