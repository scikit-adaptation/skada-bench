"""
This script generates LaTeX tables with the performance of all the solvers
on all the datasets. It also generates a table with the delta between the
performance of the supervised scorer and the best scorer for each dataset.

The tables are saved in the 'all_datasets' folder.

Example usage:
    # python generate_all_solvers_vs_all_datasets.py ../output_choleski/
"""

import argparse
import pandas as pd
import numpy as np
import os

# Import functions from generate_table_results.py
# since most of the functions are shared
# TODO: Create a utils.py file and move the shared functions there
from generate_table_results import (
    process_files_in_directory,
    keep_only_best_scorer_per_estimator,
    create_solver_vs_shift_df,
    SHIFT_ACRONYMS,
    beautify_df,
    tabulate_but_better_estimator_index,
    compute_delta_supervised_best_scorer_df,
    compute_relative_perf_df,
)


def create_solver_vs_dataset_df(
    df,
    source_target='target',
    train_test='test',
    metric='accuracy',
    simulated_datasets_params=['binary'],
    shift_name='source_target',
    simulated_shift_name='shift',
    supervised_scorer_only=False,
):
    # Creates a DataFrame with the mean of the acc for each solver/dataset
    # All shifts are aggregated unless the dataset is simulated_shifts

    # Thus we'll have this format:
    # solver | dataset_1 | dataset_2 | ... | dataset_n
    # solver_1 | mean_1 | mean_2 | ... | mean_n
    # solver_2 | mean_1 | mean_2 | ... | mean_n
    # ...

    # Create the df with the right format
    solver_vs_dataset_df = pd.DataFrame(columns=[], index=[])

    # First we get the datasets names
    dataset_names = []

    for index_tuple in df.index:
        dataset_name = index_tuple[0].split('[')[0]
        dataset_names.append(dataset_name)

    dataset_names = np.unique(dataset_names)

    # We work dataset by dataset
    for dataset_name in dataset_names:
        print(f"Processing dataset: {dataset_name}")
        filtered_df = df[
            [
                dataset_name in index_tuple[0]
                for index_tuple in df.index
            ]
        ]
        filtered_columns = [
            col for col in df.columns
            if ((source_target + '_' + metric) == col[0].lower() and
                train_test in col[1].lower())
        ]
        filtered_columns.append('scorer')
        filtered_df = filtered_df.loc[:, filtered_columns]

        if supervised_scorer_only:
            # We keep only the supervised scorer
            filtered_df = filtered_df[filtered_df['scorer'] == 'supervised']
        else:
            # First we remove the "supervised" scorer
            filtered_df = filtered_df[filtered_df['scorer'] != 'supervised']

        if dataset_name == 'Simulated':
            # If the dataset is simulated_shifts we consider each shift
            # as a different dataset
            for dataset_param in simulated_datasets_params:
                filtered_df = filtered_df[
                    [
                        dataset_param in index_tuple[0].lower()
                        for index_tuple in filtered_df.index
                    ]
                ]

            # We keep only the best scorer for each estimator
            other_scorers_df = keep_only_best_scorer_per_estimator(filtered_df)

            other_scorers_df = create_solver_vs_shift_df(
                other_scorers_df,
                simulated_shift_name
            )

            simulated_shifts_col_names = other_scorers_df.columns
            other_scorers_df.columns = simulated_shifts_col_names

            # We add the new columns to the solver_vs_dataset_df
            solver_vs_dataset_df = pd.concat(
                [solver_vs_dataset_df, other_scorers_df],
                axis=1
            )
        else:
            # We keep only the best scorer for each estimator
            other_scorers_df = keep_only_best_scorer_per_estimator(filtered_df)

            other_scorers_df = create_solver_vs_shift_df(
                other_scorers_df,
                shift_name
            )

            # We keep only the column Mean
            col_to_drop = [
                col
                for col in other_scorers_df.columns
                if col != 'Mean'
            ]
            other_scorers_df = other_scorers_df.drop(columns=col_to_drop)

            # We rename the column to the dataset name
            other_scorers_df.columns = [dataset_name]

            # We add the new column to the solver_vs_dataset_df
            solver_vs_dataset_df = pd.concat(
                [solver_vs_dataset_df, other_scorers_df],
                axis=1
            )

    return solver_vs_dataset_df


def generate_latex_table(df, folder, latex_file_name):
    # Beautify the DataFrame
    df_copy = df.copy()
    df_copy = beautify_df(df_copy)

    # Add 'Simulated' before the shifts columns
    df_copy.columns = [
        'Sim. ' + col
        if col in SHIFT_ACRONYMS.values() else col
        for col in df_copy.columns
    ]

    df_str = df_copy.map(lambda x: str(x).replace('+', r'+'))

    # Convert DataFrame to LaTeX table
    latex_table = tabulate_but_better_estimator_index(df_str, latex_file_name)

    latex_file_name = folder + '_' + latex_file_name
    latex_file_name = os.path.join(folder, latex_file_name)
    os.makedirs(folder, exist_ok=True)

    with open(latex_file_name, 'w') as f:
        f.write(latex_table)

    print(f'LaTeX table saved to {latex_file_name}')
    return latex_table


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate LaTeX tables from CSV or Parquet files"
    )
    parser.add_argument(
        "directory",
        type=str,
        help="Path to the directory containing CSV or Parquet files"
    )
    args = parser.parse_args()

    supervised_scorer_only = False

    # Process files in the specified directory + Cleanup the DataFrames
    df = process_files_in_directory(args.directory)

    solver_vs_dataset_df_other_scorers = create_solver_vs_dataset_df(
        df,
        supervised_scorer_only=False
    )

    solver_vs_dataset_df_supervised = create_solver_vs_dataset_df(
        df,
        supervised_scorer_only=True
    )

    solver_vs_dataset_df_delta = compute_delta_supervised_best_scorer_df(
        solver_vs_dataset_df_supervised,
        solver_vs_dataset_df_other_scorers
    )

    solver_vs_dataset_df_relative_perf, _, _ = compute_relative_perf_df(
        solver_vs_dataset_df_supervised,
        normalize=False,
        handle_nan=False,
    )

    solver_vs_dataset_df_relative_perf_normalized, _, _ = \
        compute_relative_perf_df(
            solver_vs_dataset_df_supervised,
            normalize=True,
            handle_nan=False,
        )

    output_folder = 'all_datasets'

    print('\n')
    print('Computing table with all the estimators and all the datasets')
    if supervised_scorer_only:
        solver_vs_dataset_table = generate_latex_table(
            solver_vs_dataset_df_supervised,
            folder=output_folder,
            latex_file_name='solvers_vs_datasets_table.tex'
        )
    else:
        solver_vs_dataset_table = generate_latex_table(
            solver_vs_dataset_df_other_scorers,
            folder=output_folder,
            latex_file_name='solvers_vs_datasets_table.tex'
        )

    print('\n')
    print('Computing table with the delta between the supervised scorer '
          'and the best scorer')
    solver_vs_dataset_delta_table = generate_latex_table(
        solver_vs_dataset_df_delta,
        folder=output_folder,
        latex_file_name='solvers_vs_datasets_delta_table.tex'
    )

    print('\n')
    print('Computing table with the relative performances per estimator '
          '(non-normalized)')
    solver_vs_dataset_relative_perf_table = generate_latex_table(
        solver_vs_dataset_df_relative_perf,
        folder=output_folder,
        latex_file_name='solvers_vs_datasets_relative_perf_table.tex'
    )

    print('\n')
    print('Computing table with the relative performances per estimator '
          '(normalized)')
    solver_vs_dataset_relative_perf_normalized_table = generate_latex_table(
        solver_vs_dataset_df_relative_perf_normalized,
        folder=output_folder,
        latex_file_name='solvers_vs_datasets_'
                        'relative_perf_normalized_table.tex'
    )
