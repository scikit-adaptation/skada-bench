# How to Contribute to SKADA-Bench

Welcome to the SKADA-Bench project! We appreciate your interest in contributing to our domain adaptation benchmark. This guide will help you add new DA methods, datasets or scorers to SKADA-Bench using the `benchopt` framework.

## Table of Contents
1. [Adding a New Shallow DA Method](#adding-a-new-shallow-da-method)
    - [Example](#example-shallow-da-method)
2. [Adding a New Deep DA Method](#adding-a-new-deep-da-method)
    - [Example](#example-deep-da-method)
3. [Adding a New Dataset](#adding-a-new-dataset)
    - [Example](#example-dataset)
4. [Adding a New Scorer](#adding-a-new-scorer)
    - [Example](#example-scorer)
5. [Contributing to the Repository](#contributing-to-the-repository)
    - [Opening Pull Requests](#opening-pull-requests)
    - [Code Review Process](#code-review-process)
    - [Reporting Issues](#reporting-issues)

## Adding a New Shallow DA Method

To add a new Shallow DA method, follow these steps:

1. **Create a Solver Class:**
   - Navigate to the `solvers` folder.
   - Create a new Python file for your method.
   - Define a class called `Solver` that inherits from `DASolver`.
   - Implement the `get_estimator()` function, which returns a class inheriting from `sklearn.BaseEstimator`.
   - See `_solvers_scorers_registry.py` on how to add the new solver to the benchmark registry. 

### Example Shallow DA Method

Here's an example of how to add a new shallow DA method:

```python
from benchmark_utils.base_solver import DASolver
from sklearn.base import BaseEstimator

class MyShallowDAEstimator(BaseEstimator):
    def __init__(self, param1=10, param2='auto'):
        self.param1 = param1
        self.param2 = param2

    def fit(self, X, y, sample_weight=None):
        # sample_weight<0 are source samples
        # sample_weight>=0 are target samples
        # y contains -1 for masked target samples
        # Your code here: store stuff in self for later predict
        return self

    def predict(self, X):
        # Do prediction on target domain here
        return ypred

    def predict_proba(self, X):
        # Do probabilistic prediction on target domain here
        return proba

class Solver(DASolver):
    name = "My_Shallow_DA_method"
    
    # Param grid to validate
    default_param_grid = {
        'param1': [10, 100],
        'param2': ['auto', 'manual']
    }
    
    def get_estimator(self):
        return MyShallowDAEstimator()
```

## Adding a New Deep DA Method

To add a new Deep DA method, follow these steps:

1. **Create a Deep Solver Class:**
   - Navigate to the `solvers` folder.
   - Create a new Python file for your method.
   - Define a class called `Solver` that inherits from `DeepDASolver`.
   - Implement the `get_estimator()` function, which returns a class inheriting from `torch.nn.Module`.
   - Use the appropriate backbone architecture for your method.
   - See `_solvers_scorers_registry.py` on how to add the new solver to the benchmark registry.

For deep learning methods, you need to define the backbone architecture and set hyperparameters. Here's how to do it:

a. **Define a Backbone:**
   - Use an existing backbone from `benchmark_utils/backbones_architecture.py`, or create a new class that inherits from `nn.Module`.
   - Implement the `forward()` method for your backbone.
   - Ensure that the feature extracting layer is named 'feature_layer'. This is crucial for the proper functioning of the benchmark.

b. **Set Hyperparameters:**
   - In `benchmark_utils/utils.py`, add or modify an entry in the `dataset_configs` dictionary in the `get_params_per_dataset()` function.
   - Specify hyperparameters such as batch size, learning rate, optimizer, and callbacks.

### Example Deep DA Method

Here's an example of how to use a backbone in your solver:

```python
from benchmark_utils.base_solver import DeepDASolver
from benchmark_utils.backbones_architecture import get_backbone
from benchmark_utils.utils import get_params_per_dataset

class Solver(DeepDASolver):
    name = "YourDeepDAMethod"

    def get_estimator(self, n_classes, device, dataset_name, **kwargs):
        dataset_name = dataset_name.split("[")[0].lower()

        params = get_params_per_dataset(dataset_name, n_classes)

        net = YourCustomNet(
            **params,
            layer_name="feature_layer",
            train_split=None,
            device=device,
            warm_start=True,
        )

        return net
```

In this example, `YourCustomNet` would be your custom PyTorch module that implements your deep DA method using the provided backbone and hyperparameters.

## Adding a New Dataset

To add a new dataset, follow these steps:

1. **Create a Dataset Class:**
   - Navigate to the `datasets` folder.
   - Create a new Python file for your dataset.
   - Define a class called `Dataset` that inherits from `BaseDataset`.
   - Implement the `get_data()` function, which returns a dictionary with keys `X`, `y`, and `sample_domain`.

### Example Dataset

Here's an example of how to add a new dataset:

```python
from benchopt import BaseDataset
from sklearn.datasets import make_blobs
import numpy as np

class Dataset(BaseDataset):
    name = "example_dataset"
    
    def get_data(self):
        X_source, y_source = make_blobs(
            n_samples=100, centers=3,
            n_features=2, random_state=0
        )
        X_target, y_target = make_blobs(
            n_samples=100, centers=5,
            n_features=2, random_state=42
        )
        # sample_domain is negative for target samples and positive for source
        sample_domain = np.array([1]*len(X_source) + [-2]*len(X_target))
        
        return dict(
            X=np.concatenate((X_source, X_target), axis=0),
            y=np.concatenate((y_source, y_target)),
            sample_domain=sample_domain
        )
```

## Adding a New Scorer

To add a new scorer, follow these steps:

1. **Create a Scorer Class:**
   - Create a new Python file for your scorer.
   - Define a class for your scorer.
   - Implement the `_score()` function, which returns a `float`.

2. **Add your Scorer to the benchmark:**
   - Navigate to the `benchmark_utils` folder.
   - Open the `scorers` Python file.
   - Import and add your scorer to the `CRITERIONS` dictionnary.
   - See `_solvers_scorers_registry.py` on how to add the new scorer to the benchmark registry. 

### Example Scorer

Here's an example of how to add a new scorer:

```python
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    from skada.metrics import (
        SupervisedScorer,
        PredictionEntropyScorer,
        ImportanceWeightedScorer,
        SoftNeighborhoodDensity,
        DeepEmbeddedValidation,
        CircularValidation,
    )

CRITERIONS = {
    'supervised': SupervisedScorer(),
    'prediction_entropy': PredictionEntropyScorer(),
    'importance_weighted': ImportanceWeightedScorer(),
    'soft_neighborhood_density': SoftNeighborhoodDensity(),
    'deep_embedded_validation': DeepEmbeddedValidation(),
    'circular_validation': CircularValidation()
}
```

## Contributing to the Repository

We welcome contributions to the SKADA-Bench project! Here's how you can contribute by opening Pull Requests (PRs) and participating in the development process.

### Opening Pull Requests

1. **Fork the Repository:**
   - Visit https://github.com/scikit-adaptation/skada-bench
   - Click the "Fork" button in the top-right corner to create your own copy of the repository.

2. **Clone Your Fork:**
   ```
   git clone https://github.com/your-username/skada-bench.git
   cd skada-bench
   ```

3. **Create a New Branch:**
   ```
   git checkout -b feature/your-feature-name
   ```

4. **Make Your Changes:**
   - Add your new DA method, dataset, or scorer as described in the sections above.
   - Make sure to follow the project's coding style and conventions.

5. **Commit Your Changes:**
   ```
   git add .
   git commit -m "Add a brief, descriptive commit message"
   ```

6. **Push to Your Fork:**
   ```
   git push origin feature/your-feature-name
   ```

7. **Open a Pull Request:**
   - Go to https://github.com/scikit-adaptation/skada-bench
   - Click on "Pull requests" and then the "New pull request" button.
   - Select your fork and the branch you created.
   - Fill in the PR template with a clear title and description of your changes.

### Code Review Process

1. **Wait for Review:** Maintainers will review your PR as soon as possible.
2. **Address Feedback:** If changes are requested, make the necessary updates and push them to your branch.
3. **Approval and Merge:** Once approved, a maintainer will merge your PR into the main branch.

### Reporting Issues

If you find a bug or have a suggestion for improvement:

1. Check the [Issues](https://github.com/scikit-adaptation/skada-bench/issues) page to see if it has already been reported.
2. If not, create a new issue with a clear title and detailed description.
3. Use labels to categorize the issue (e.g., bug, enhancement, question).

Thank you for contributing to SKADA-Bench! Your efforts help improve the project for everyone in the domain adaptation community.
