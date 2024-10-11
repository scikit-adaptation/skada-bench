import sys
import os
from pathlib import Path

path_root = Path(os.path.abspath(__file__)).parents[1]
sys.path.append(str(path_root))

import itertools
from sklearn.metrics import accuracy_score
import torch

from datasets.officehome import Dataset
from solvers.deep_no_da_source_only import Solver
from objective import Objective

import argparse

cache_dir = Path("__cache__")
cache_dir.mkdir(exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--debug", action="store_true", help="debug mode")

args = parser.parse_args()
debug = args.debug

TASKS = ["art", "clipart", "product", "realworld"]

results = dict()

all_source_target_pairs = list(itertools.permutations(TASKS, 2))
if debug:
    all_source_target_pairs = all_source_target_pairs[:2]

for source, target in all_source_target_pairs:
    print(f"{source} -> {target}")

    dataset = Dataset()
    dataset.source_target = (source, target)
    solver = Solver()
    objective = Objective()

    # Load data
    data = dataset.get_data()

    # Set the data in the objective
    objective.set_data(**data)

    # Get a cross-validation fold
    cv_fold = next(
        objective.cv.split(objective.X, objective.y, objective.sample_domain)
    )
    X_train, y_train, sample_domain_train, _ = objective.split(
        cv_fold, objective.X, objective.y, objective.sample_domain
    )
    X_test, y_test = objective.X_test, objective.y_test
    sample_domain_test = objective.sample_domain_test

    # Debug mode
    FACTOR = 100
    if debug:
        X_train = X_train[::FACTOR]
        y_train = y_train[::FACTOR]
        sample_domain_train = sample_domain_train[::FACTOR]

    # Get the model from the Solver
    n_classes = len(set(objective.y))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    dataset_name = dataset.name
    estimator = solver.get_estimator(
        n_classes=n_classes, device=device, dataset_name=dataset_name
    )
    hp = {
        "max_epochs": 20,
        "lr": 1,
    }
    estimator = estimator.set_params(**hp)

    # Train the model on the training data
    estimator.fit(X_train, y_train, sample_domain=sample_domain_train)

    # Get the test target set
    X_test_target = X_test[sample_domain_test < 0]
    y_test_target = y_test[sample_domain_test < 0]
    sample_domain_test_target = sample_domain_test[sample_domain_test < 0]

    # Predict on the test target set
    y_pred_test_target = estimator.predict(
        X_test_target, sample_domain=sample_domain_test_target
    )

    # Compute the accuracy on the test set
    accuracy = accuracy_score(y_test_target, y_pred_test_target)
    print(f"Test Target Accuracy: {accuracy:.4f}")

    # Save the results
    results[(source, target)] = accuracy

overall_accuracy = sum(results.values()) / len(results)
print(f"Overall Accuracy: {overall_accuracy:.4f}")
