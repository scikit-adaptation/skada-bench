"""
Script to visualize the impact of hyperparameters on the objective function
for each solver.

The script reads the CSV or Parquet files from the specified directory
and plots the validation curves for each solver.

Example usage:
    # For Binary simulted dataset
    # python visualize_hp_impact.py
    # --directory ../output_choleski/outputs_binary_simulated

    # For Office31Surf dataset
    # python visualize_hp_impact.py
    # --directory ../output_choleski/outputs_office31surf

    # For mnist_usps dataset
    # python visualize_hp_impact.py
    # --directory ../output_choleski/outputs_mnist_usps

    # For 20news dataset
    # python visualize_hp_impact.py
    # --directory ../output_choleski/outputs_20newsgroup

    # For Mushroom dataset
    # python visualize_hp_impact.py
    # --directory ../output_choleski/outputs_mushrooms

    # For Office31Decaf dataset
    # python visualize_hp_impact.py
    # --directory ../output_choleski/outputs_office31decaf


#TODO: Add an argparse to specify the metric to plot
(right now it's fixed to 'mean_test_supervised')
"""

import os
import argparse
import ast
import re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_param_grid(data, output_dir):
    """
    Plot the impact of hyperparameters on the objective function
    for each solver.
    """
    dataset = data['data_name'].iloc[0].split('[')[0]

    num_plots = len(data['solver_name'].unique())

    fig = plt.figure(figsize=(40, 80))
    subfigs = fig.subfigures(num_plots, 1)

    for solver, subfig in zip(data['solver_name'].unique(), subfigs.flat):
        solver_data = data[data['solver_name'] == solver]
        objective_cv_results = solver_data['objective_cv_results']

        objective_cv_results = objective_cv_results.apply(
            lambda x: extract_from_str(x) if isinstance(x, str) else x
        )

        if objective_cv_results.empty:
            print(
                f"No valid 'objective_cv_results' found for solver: {solver}"
            )
            continue
        objective_cv_results = pd.DataFrame(objective_cv_results.tolist())

        # Keep only column 'mean_test_supervised' and
        # columns starting with 'param'
        objective_cv_results = objective_cv_results[
            ['mean_test_supervised', 'params']
        ]

        # Explode the 'mean_test_supervised' and 'params' columns
        objective_cv_results = objective_cv_results.explode(
            ['mean_test_supervised', 'params']
        )

        # Convert 'params' column from dict to columns
        objective_cv_results = pd.concat(
            [
                objective_cv_results.drop(['params'], axis=1),
                objective_cv_results['params'].apply(pd.Series)
            ],
            axis=1
        )

        # Drop col with all NaN
        objective_cv_results = objective_cv_results.dropna(axis=1, how='all')

        # Compute mean of 'mean_test_supervised' column
        # for each group of 'params'
        cols_to_groupby = (
            objective_cv_results.columns.difference(['mean_test_supervised'])
        )
        if len(cols_to_groupby) == 0:
            print(f"No hyperparameters found for solver: {solver}")
            subfig.suptitle(
                'Validation curves, dataset: ' +
                dataset + ", solver: " + solver
            )
            continue
        else:
            # Replace inf values with 0
            objective_cv_results = (
                objective_cv_results.replace([np.inf, -np.inf], 0)
            )
            mean_results = (
                objective_cv_results.groupby(
                    cols_to_groupby.tolist()
                )
                .mean().reset_index()
            )
            std_results = (
                objective_cv_results.groupby(
                    cols_to_groupby.tolist()
                )
                .std().reset_index()
            )

        axes = subfig.subplots(1, len(cols_to_groupby), sharey='row')

        if len(cols_to_groupby) == 1:
            axes = [axes]  # Convert single axis to list for uniform handling

        subfig.suptitle(
            'Validation curves, dataset: ' + dataset + ", solver: " + solver
        )

        plot_single_solver(axes, mean_results, std_results, cols_to_groupby)

    # fig.suptitle('Validation curves, dataset: ' + dataset, fontsize=25, y=0)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, f'{dataset}_validation_curves.png'))


def plot_single_solver(axes, mean_results, std_results, cols_to_groupby):
    """
    Plot the impact of hyperparameters on the objective function
    for a single solver.
    Each subplot corresponds to a different hyperparameter
    """
    axes[0].set_ylabel("Test Supervised Accuracy")

    for idx, param in enumerate(cols_to_groupby):
        grouped_mean_results = (
            mean_results.groupby(param)['mean_test_supervised'].mean()
        )
        grouped_mean_results = grouped_mean_results.reset_index()
        grouped_mean_results['mean_test_supervised'] = (
            grouped_mean_results['mean_test_supervised'].astype(float)
        )

        grouped_std_results = (
            std_results.groupby(param)['mean_test_supervised'].mean()
        )
        grouped_std_results = grouped_std_results.reset_index()
        grouped_std_results['mean_test_supervised'] = (
            grouped_std_results['mean_test_supervised'].astype(float)
        )

        try:
            # Check if the values in grouped_mean_results[param]
            # are numeric and not categorical/boolean
            if grouped_mean_results[param].dtype.kind in 'iufc':
                if len(grouped_mean_results) == 1:
                    axes[idx].scatter(
                        grouped_mean_results[param],
                        grouped_mean_results['mean_test_supervised'],
                        color="darkorange",
                        marker='x',
                    )
                else:
                    axes[idx].scatter(
                        grouped_mean_results[param],
                        grouped_mean_results['mean_test_supervised'],
                        color="darkorange",
                        marker='x',
                    )

                    axes[idx].plot(
                        grouped_mean_results[param],
                        grouped_mean_results['mean_test_supervised'],
                        color="darkorange",
                    )

                    axes[idx].fill_between(
                        grouped_mean_results[param],
                        (grouped_mean_results['mean_test_supervised'] -
                         grouped_std_results['mean_test_supervised']).values,
                        (grouped_mean_results['mean_test_supervised'] +
                         grouped_std_results['mean_test_supervised']).values,
                        alpha=0.2,
                        color="darkorange",
                    )
            else:
                grouped_mean_results[param] = (
                    grouped_mean_results[param].astype(str)
                )
                axes[idx].bar(
                    grouped_mean_results[param],
                    grouped_mean_results['mean_test_supervised'],
                    yerr=grouped_std_results['mean_test_supervised'],
                    color="darkorange",
                )
        except Exception as e:
            print(f"Error: {e}")
        try:
            axes[idx].set_xlabel(param.split('__')[-1])
        except Exception as e:
            print(f"Error: {e}")
        axes[idx].set_ylim(0, 1)
        axes[idx].grid()


def extract_from_str_mean_test_supervised(string):
    """
    Extract the 'mean_test_supervised' array from the string
    """
    start_index = (
        string.find("'mean_test_supervised': array([") +
        len("'mean_test_supervised': array(["))
    end_index = string.find("]), 'std_test_supervised':")

    # Extract the substring
    desired_part = string[start_index:end_index]
    desired_part = desired_part.replace('\n', '')

    # Replace '-inf' with '-np.inf'
    desired_part = desired_part.replace('-inf', '-np.inf')
    return np.array(eval(desired_part))


def extract_from_str_params(string):
    """
    Extract the 'params' dictionary from the string
    """
    start_index = string.find("'params':") + len("'params':")
    end_index = string.find(", 'split0_")

    # Extract the substring
    desired_part = string[start_index:end_index]

    desired_part = desired_part.replace(' ', '')

    try:
        params = ast.literal_eval(desired_part)
    except SyntaxError:
        # If there is a syntax error,
        # return the base_estimator name as the only parameter
        estimators = re.findall(r'base_estimator=([^,()]+)', desired_part)
        # params should be a list of dictionaries
        params = [{'base_estimator': estimator} for estimator in estimators]
    return params


def extract_from_str(string):
    """
    Extract the 'mean_test_supervised' array
    and 'params' dictionary from the string
    """
    mean_test_supervised = extract_from_str_mean_test_supervised(string)
    params = extract_from_str_params(string)

    dict_ = {'mean_test_supervised': mean_test_supervised, 'params': params}

    return dict_


def get_files_from_directory(directory):
    """
    Get all CSV or Parquet files from the specified directory
    """

    df_list = []
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".csv") or filename.endswith(".parquet"):
                file_path = os.path.join(root, filename)
                print(f"Processing file: {file_path}")

                # Read the CSV or Parquet file into a pandas DataFrame
                if file_path.endswith(".csv"):
                    df = pd.read_csv(file_path)
                elif file_path.endswith(".parquet"):
                    df = pd.read_parquet(file_path)

                df_list.append(df)
            else:
                print(f"Ignoring file: {filename} \\"
                      "as it is not a CSV or Parquet file")

    print('\n')
    # Concatenate all DataFrames in the list
    all_df = pd.concat(df_list)

    return all_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize the impact of hyperparameters \\"
                    "on the objective function for each solver"
    )

    parser.add_argument(
        "--directory",
        type=str,
        help="Path to the directory containing CSV or Parquet files",
        default='../output_choleski/outputs_binary_simulated'
    )

    parser.add_argument(
        "--output",
        type=str,
        help="Path to the output directory",
        default='./validation_curves/'
    )

    args = parser.parse_args()

    # Step 1: Load the Data
    # Load the data from the specified directory
    data = get_files_from_directory(args.directory)

    # Plot the impact of hyperparameters on the objective function
    plot_param_grid(data, args.output)
