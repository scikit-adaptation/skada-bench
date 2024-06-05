import os
import argparse

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from generate_table_results import (
    DA_TECHNIQUES,
    ESTIMATOR_DICT,
)


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

        ### Plotting part ###
        all_df['solver_name'] = all_df['solver_name'].map(lambda x: (x.split('[param_grid=')[0]))

        # Add Type column
        # Create a reverse lookup dictionary
        reverse_lookup = {solver: technique for technique, solvers in DA_TECHNIQUES.items() for solver in solvers}
        all_df['Type'] = all_df['solver_name'].map(reverse_lookup)


        # Set to 'Unknown' for the rest
        all_df['Type'].fillna('Unknown', inplace=True)

        all_df = all_df[all_df['Type'] != 'Unknown']


        all_df['Estimator'] = all_df['solver_name'].map(lambda x: ESTIMATOR_DICT.get(x, x))

        all_df = all_df[['Estimator', 'Type', 'time']]

        # Times num_idx because, theres num_idx splits
        groupby_df = all_df.groupby(['Estimator', 'Type']).mean()
        groupby_df = groupby_df.reset_index()

        order = groupby_df.groupby('Type', group_keys=False).apply(lambda x: x.sort_values(by='time'))
        order = order.set_index('Type').loc[
            ['NO DA', 'Reweighting', 'Mapping', 'Subspace', 'Other']
        ].reset_index()


        plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', zorder=-1)

        g = sns.barplot(
            x = 'Estimator',
            y = 'time',
            hue = 'Type',
            data = order,
            palette = 'colorblind',
            dodge=False,
        )

        g.set_yscale("log")

        g.set_xticklabels(
            g.get_xticklabels(), 
            rotation=45, 
            horizontalalignment='right'
        )

        fig = plt.gcf()
        fig.set_size_inches(8, 4)

        plt.tight_layout()
        plt.legend(loc='upper left')
        plt.ylabel("Mean computational time (in sec)")
        
        fig.savefig('estimator_VS_time.png', dpi=100)



def convert_seconds(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    remaining_seconds = seconds % 60
    return hours, minutes, remaining_seconds



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute computational time \\"
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

