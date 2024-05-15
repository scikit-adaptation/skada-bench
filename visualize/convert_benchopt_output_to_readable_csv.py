"""
This script processes and cleans output files from Benchopt, converting them
into a more readable CSV format.

The script performs the following steps:
1. Loads data from a specified directory containing CSV or Parquet files.
2. Cleans and processes the data
3. Exports the cleaned dataframe to a specified output directory.

Usage:
    python clean_benchopt_output.py --directory <input_directory> --output <output_directory>

Arguments:
    --directory: Path to the directory containing the CSV or Parquet files.
                 Default is '../outputs'.
    --output:    Path to the directory where the cleaned CSV file will be saved.
                 Default is './cleaned_outputs'.

Example:
    python clean_benchopt_output.py --directory ../outputs --output ./cleaned_outputs
"""

import os
import argparse

from generate_table_results import (
    generate_df,
    process_files_in_directory,
    DA_TECHNIQUES,
    ESTIMATOR_DICT,
)

def clean_benchopt_df(df):
    # We remove '[param_grid=default]' in each method name
    df.index = df.index.map(lambda x: (x[0], x[1].split('[param_grid=default]')[0]))

    # We keep only the columns target/test + the scorer column
    filtered_columns = [
        col for col in df.columns
        if (
            'target' in col[0].lower()
            and 'test' in col[1].lower()
        )
    ]

    filtered_columns.append('scorer')
    df = df.loc[:, filtered_columns]

    # Rename the columns by concatenating the tuples with a hyphen, except 'scorer'
    df.columns = [
        '-'.join([col[0], col[2]])
        if isinstance(col, tuple) and len(col) > 2
        else col
        for col in df.columns
    ]

    # Remove 'target' in col names since here its implied
    df.columns = [col.replace('target_', '') for col in df.columns]

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
    df['estimator'] = df['estimator'].map(ESTIMATOR_DICT)

    # Remove the params in the dataset name
    df['dataset'] = df['dataset'].apply(lambda x: x.split('[')[0])

    # Reorganize df
    df = df[
        ['dataset', 'estimator', 'scorer', 'type'] +
        [col for col in df.columns if col not in ['dataset', 'estimator', 'scorer', 'type']]
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
        "--output",
        type=str,
        help="Path to the output directory",
        default='./cleaned_outputs'
    )

    args = parser.parse_args()

    # Step 1: Load the Data
    # Load the data from the specified directory
    df = process_files_in_directory(args.directory)

    # Step 2: Clean the dataframe
    df = clean_benchopt_df(df)

    # Step 3: Export to CSV
    output_directory = args.output

    os.makedirs(output_directory, exist_ok=True)

    df.to_csv(output_directory + '/readable_csv.csv', index=False)
