"""
This script processes and cleans output files from Benchopt, converting them
into a more readable CSV format.
Note: The rows with 'best_scorer' as the scorer are the best unsupervised scorers,
except for the NO_DA methods, which use the supervised scorer.

The script performs the following steps:
1. Loads data from a specified directory containing CSV or Parquet files.
2. Cleans and processes the data
3. Exports the cleaned dataframe to a specified output directory.

Usage:
    python clean_benchopt_output_to_readable_format.py --directory <input_directory> --domain <target-source> --output <output_directory>

Arguments:
    --directory: Path to the directory containing the CSV or Parquet files.
                 Default is '../outputs'.
    --domain:    Specify whether to output the results of the 'target' or 'source' domains.
    --output:    Path to the directory where the cleaned CSV file will be saved.
                 Default is './cleaned_outputs'.

Example:
    python clean_benchopt_output_to_readable_format.py --directory ../outputs --domain target --output ./cleaned_outputs
"""

import os
import argparse

import pandas as pd

from generate_table_results import (
    process_files_in_directory,
    keep_only_best_scorer_per_estimator,
    regex_match,
    DA_TECHNIQUES,
    ESTIMATOR_DICT,
)

def clean_benchopt_df(df, domain):
    # We remove '[param_grid=...]' from the dataset name
    df.index = df.index.map(lambda x: (x[0], x[1].split('[param_grid=')[0]))

    # We keep only the columns domain/test + the scorer column
    filtered_columns = [
        col for col in df.columns
        if (
            domain in col[0].lower()
            and 'test' in col[1].lower()
        )
    ]

    filtered_columns.append('scorer')
    df = df.loc[:, filtered_columns]

    # Get df for the best unsupervised scorer
    # First we remove the "supervised" scorer
    best_unsupervised = df[df['scorer'] != 'supervised']
    best_unsupervised = keep_only_best_scorer_per_estimator(
        best_unsupervised,
        specific_col = (domain + '_accuracy', 'test', 'mean'),
    )

    # Remove NO_DA methods from the best_unsupervised df
    no_da_methods = [solver for solver in DA_TECHNIQUES['NO DA']]
    best_unsupervised = best_unsupervised[
        ~best_unsupervised.index.get_level_values(1).isin(no_da_methods)
    ]

    # For the NO_DA methods, we use the results from the supervised scorer
    no_da_methods = [solver for solver in DA_TECHNIQUES['NO DA']]
    no_da_df = df[df.index.get_level_values(1).isin(no_da_methods)]
    no_da_df = no_da_df[no_da_df['scorer'] == 'supervised']

    # Concatenate the two dataframes
    best_scores_df = pd.concat([best_unsupervised, no_da_df])
    best_scores_df['scorer'] = "best_scorer"

    # Add the best scores to the original df
    df = pd.concat([df, best_scores_df])
    

    # Rename the columns by concatenating the tuples with a hyphen, except 'scorer'
    df.columns = [
        '-'.join([col[0], col[2]])
        if isinstance(col, tuple) and len(col) > 2
        else col
        for col in df.columns
    ]

    # Remove the domain in col names since here its implied
    df.columns = [col.replace(domain + '_', '') for col in df.columns]

    # Move dataset name and estimator from index
    df['dataset'] = [index_tuple[0] for index_tuple in df.index]
    df['estimator'] = [index_tuple[1] for index_tuple in df.index]

    df = df.reset_index(drop=True)
    
    # Add type column
    # Create a reverse lookup dictionary
    reverse_lookup = {solver: technique for technique, solvers in DA_TECHNIQUES.items() for solver in solvers}
    df['type'] = df['estimator'].map(reverse_lookup)

    # Set to 'Unknown' for the rest
    df['type'].fillna('Unknown', inplace=True)

    # Rename solvers with ESTIMATOR_DICT
    df['estimator'] = df['estimator'].map(lambda x: ESTIMATOR_DICT.get(x, x))

    # Function to extract shift value
    def extract_shift(dataset):
        if 'shift' in dataset:
            return dataset.split('shift=')[1].strip(']')
        elif 'source_target' in dataset:
            regex = ".*source_target=\('([^']+)', '([^']+)'\).*"
            return regex_match(regex, dataset)
        elif 'subject_id' in dataset:
            return dataset.split('subject_id=')[1].strip(']')
        return None
    
    # Add shift as a new column
    df['shift'] = df['dataset'].apply(extract_shift)

    # Remove the params in the dataset name
    df['dataset'] = df['dataset'].apply(lambda x: x.split('[')[0])

    # Reorganize df
    df = df[
        ['dataset', 'shift', 'estimator', 'scorer', 'type'] +
        [col for col in df.columns if col not in ['dataset', 'shift', 'estimator', 'scorer', 'type']]
    ]
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert the benchopt output files \\"
                    "into a more readable csv"
    )

    parser.add_argument(
        "--directory",
        type=str,
        help="Path to the directory containing CSV or Parquet files",
        default='../outputs'
    )

    parser.add_argument(
        "--domain",
        type=str,
        choices=['target', 'source'],
        help="Specify whether to output the results of the 'target' or 'source' domains.",
        default='target',
    )

    parser.add_argument(
        "--output",
        type=str,
        help="Path to the output directory",
        default='./cleaned_outputs'
    )

    args = parser.parse_args()

    # Step 1: Load the Data
    # Load the data from the specified directory
    df = process_files_in_directory(args.directory)

    print(f"Using {args.domain} domain to generate csv file")
    # Step 2: Clean the dataframe
    df = clean_benchopt_df(df, args.domain)

    # Step 3: Export to CSV
    output_directory = args.output

    os.makedirs(output_directory, exist_ok=True)

    df.to_csv(output_directory + '/readable_csv.csv', index=False)
