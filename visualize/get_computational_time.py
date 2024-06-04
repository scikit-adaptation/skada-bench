import os
import argparse

import pandas as pd


def process_files_in_directory(directory):
    df_list = []
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".csv") or filename.endswith(".parquet"):
                file_path = os.path.join(root, filename)
                print(f"Processing file: {file_path}")
                if file_path.endswith(".csv"):
                    df = pd.read_csv(file_path)
                elif file_path.endswith(".parquet"):
                    df = pd.read_parquet(file_path)
                else:
                    print(f"Unsupported file format: {file_path}")
                    pass
                df_list.append(df)
            else:
                print(
                    f"Ignoring file: {filename} (not a .csv or .parquet file)"
                )

    print('\n')
    # Concatenate all DataFrames in the list
    if len(df_list) != 0:
        all_df = pd.concat(df_list)

        total_time = all_df['time'].sum()
        hours, minutes, remaining_seconds = convert_seconds(total_time)
        print('Computational time:')
        print(f"Hours: {hours}, Minutes: {minutes}, Seconds: {remaining_seconds:.2f}")


def convert_seconds(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    remaining_seconds = seconds % 60
    return hours, minutes, remaining_seconds



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute comptutational time \\"
                    "for an experiment"
    )

    parser.add_argument(
        "--directory",
        type=str,
        help="Path to the directory containing CSV or Parquet files",
        default='../outputs'
    )

    args = parser.parse_args()

    df = process_files_in_directory(args.directory)

