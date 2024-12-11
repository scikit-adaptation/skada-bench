import os
import re
import json
import numpy as np
import pandas as pd

SOURCE_TARGET_ACRONYMS = {
    'amazon': 'amz',
    'caltech': 'cal',
    'dslr': 'dsl',
    'webcam': 'web',
    'enlarging': 'enl',
    'tapering': 'tap',
}


def extract_benchopt_file_informations(file_path):
    """
    Extract and process experimentation results from
    the benchopt output file.

    Parameters
    ----------
    file_path : str
        Path to the benchopt results file.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the processed experiment results.
    """
    # Read the CSV or Parquet file into a pandas DataFrame
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    elif file_path.endswith(".parquet"):
        df = pd.read_parquet(file_path)
    else:
        print(f"Unsupported file format: {file_path}")
        return

    # Remove 'random_state' from 'data_name' if present
    regex_random_state = r'random_state=\d+'
    df['data_name'] = \
        df['data_name'].str.replace(regex_random_state, '', regex=True)

    # Exclude specific columns and select only the metrics columns
    exclude_columns = [
        'objective_name',
        'objective_value',
        'objective_cv_results',
    ]
    metrics_columns = [
        col
        for col in df.columns
        if col.startswith('objective_') and col not in exclude_columns
    ]

    def extract_mean_test_cv(text):
        # This pattern matches the key and the array,
        # including multiline arrays
        pattern = r"('mean_test_\w+': array\(\[.*?\]\))"
        result = {}
        try:
            matches = re.findall(pattern, text, re.DOTALL)

            for match in matches:
                key_pattern = r"'(mean_test_\w+)'"
                key_match = re.search(key_pattern, match)
                if key_match:
                    key = key_match.group(1)
                    array_pattern = r"array\((\[.*?\])\)"
                    array_match = re.search(array_pattern, match, re.DOTALL)
                    if array_match:
                        array_str = array_match.group(1).replace('\n', '')
                        array = np.fromstring(array_str.strip('[]'), sep=', ')

                        # We take only the max value in the list
                        # Since its the one that has been considered
                        # To chosse the right hps
                        result[key] = max(array.tolist())
        except Exception:
            pass

        return result

    df['cv_score'] = df['objective_cv_results'].apply(
        lambda x: extract_mean_test_cv(x)
    )
    df_exploded = pd.json_normalize(df['cv_score'])

    if df_exploded.size != 0:
        df = pd.concat([df.drop(columns=['cv_score']), df_exploded], axis=1)

    # Compute mean and standard deviation for each metric
    def identity(x):
        return json.dumps(x.tolist())

    # Group by 'data_name' and 'solver_name'
    grouped_df = df.groupby(['data_name', 'solver_name'])

    # Aggregate the specified metrics with mean, std, and identity
    agg_metrics = grouped_df[metrics_columns].agg([np.mean, np.std, identity])

    if df_exploded.size != 0:
        agg_mean_cv = grouped_df[df_exploded.columns].agg([identity])

    grouped_df = agg_metrics

    # Rename columns --> Remove 'objective_'
    new_column_names = []

    # Extract the part before 'train' or 'test'
    for index_label in grouped_df.columns.levels[0]:
        new_column_name = index_label.replace('objective_', '')
        new_column_names.append(new_column_name)

    # Assign modified column names to DataFrame
    grouped_df.columns = \
        grouped_df.columns.set_levels(new_column_names, level=0)

    # regex pattern to match 'train' or 'test'
    pattern = r'(.*?)_(train|test)_(.*)'

    new_column = []
    scorer_names = []
    train_test_names = []

    # Extract the part before 'train' or 'test' --> The scorer name
    for col_label in grouped_df.columns.levels[0]:
        matches = re.match(pattern, col_label)
        scorer_name = matches.group(1)
        train_test_name = matches.group(2)
        new_label = matches.group(3)

        new_column.append(new_label)
        scorer_names.append(scorer_name)
        train_test_names.append(train_test_name)

    # New columns: (scorer, train_test, metric)
    new_columns = [
        (scorer, train_test, metric)
        for scorer in np.unique(new_column)
        for train_test in np.unique(train_test_names)
        for metric in grouped_df.columns.levels[1]
    ]

    # Create a new DataFrame with the new columns + column 'scorer'
    new_df = pd.DataFrame(
        np.nan,
        index=np.tile(grouped_df.index, len(np.unique(scorer_names))),
        columns=new_columns
    )
    new_df['scorer'] = np.repeat(np.unique(scorer_names), len(grouped_df))

    # Fill the new DataFrame with the values from the grouped DataFrame
    for index, row in grouped_df.iterrows():
        for col in new_columns:
            for (_, row_agg), index_agg in zip(
                list(new_df.iloc[np.where(index == new_df.index)].iterrows()),
                np.where(index == new_df.index)[0]
            ):
                scorer = row_agg['scorer']
                old_col = [scorer + '_' + col[1] + '_' + col[0], col[2]]

                # Select the row with the good scorer,
                # the good train_test and the good metric
                new_df.iloc[
                    index_agg,
                    new_df.columns.get_loc((col[0], col[1], col[2]))
                ] = row[old_col[0]][old_col[1]]

    if df_exploded.size != 0:
        agg_mean_cv = agg_mean_cv.reset_index()
        agg_mean_cv.columns = agg_mean_cv.columns.droplevel(1)

        df_melted = agg_mean_cv.melt(
            id_vars=['data_name', 'solver_name'],
            var_name='scorer', value_name='cv_score'
        )
        df_melted['scorer'] = df_melted['scorer'].str.replace('mean_test_', '')

        new_df = new_df.reset_index(names=['index'])
        new_df[['data_name', 'solver_name']] = pd.DataFrame(
            new_df['index'].tolist(), index=new_df.index
        )
        new_df = new_df.drop(columns=['index'])

        merged_df = pd.merge(
            new_df, df_melted, on=['scorer', 'data_name', 'solver_name'],
            how='outer'
        )

        merged_df = merged_df.set_index(['data_name', 'solver_name'])
    else:
        merged_df = new_df

    return merged_df


def retrieve_cv(file_path):
    """
    Read and extract the cv results from the benchopt output file.

    Parameters
    ----------
    file_path : str
        Path to the CSV or Parquet file containing cross-validation results.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the processed cross-validation results.
    """
    # Read the CSV or Parquet file into a pandas DataFrame
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    elif file_path.endswith(".parquet"):
        df = pd.read_parquet(file_path)
    else:
        print(f"Unsupported file format: {file_path}")
        return

    # Keep only useful columns
    df = df[['solver_name', 'data_name', 'idx_rep', 'objective_cv_results']]

    def extract_test_cv(text):
        # This pattern matches both 'mean_test_*' and 'split*_test_*' keys
        pattern = r"('split\d+_test_\w+': array\(\[.*?\]\))"
        result = {}
        try:
            matches = re.findall(pattern, text, re.DOTALL)

            for match in matches:
                key_pattern = r"'(split\d+_test_\w+)'"
                key_match = re.search(key_pattern, match)
                if key_match:
                    key = key_match.group(1)
                    array_pattern = r"array\((\[.*?\])\)"
                    array_match = re.search(array_pattern, match, re.DOTALL)
                    if array_match:
                        array_str = array_match.group(1).replace('\n', '')
                        array = np.fromstring(array_str.strip('[]'), sep=', ')

                        # Store all values in the list
                        result[key] = array.tolist()
        except Exception:
            pass

        return result

    df['cv_score'] = df['objective_cv_results'].apply(
        lambda x: extract_test_cv(x)
    )
    df_exploded = pd.json_normalize(df['cv_score'])

    if df_exploded.size != 0:
        df = pd.concat([df.drop(columns=['cv_score']), df_exploded], axis=1)

    # Rename columns --> Remove 'objective_'
    new_column_names = []

    # Extract the part before 'train' or 'test'
    for index_label in df.columns:
        new_column_name = index_label.replace('objective_', '')
        new_column_names.append(new_column_name)

    # Assign modified column names to DataFrame
    df.columns = new_column_names

    # Function to extract split and scorer from column names

    def extract_split_and_scorer(col_name):
        match = re.match(r'split(\d+)_test_(.*)', col_name)
        if match:
            split = match.group(1)
            scorer = match.group(2)
            return split, scorer
        return None, None

    # Prepare a list to store the new rows
    new_rows = []

    # Determine columns that do not match the pattern and are not to be removed
    non_split_columns = [
        col for col in df.columns
        if not re.match(r'split\d+_test_.*', col)
    ]

    # Iterate over each row in the DataFrame
    for _, row in df.iterrows():
        # Iterate over each column in the row
        for col in df.columns:
            split, scorer = extract_split_and_scorer(col)
            if split is not None and scorer is not None:
                # Convert string representation of list to actual list
                results = row[col]

                # Check if results is a list-like object
                if isinstance(results, (list, np.ndarray)):
                    # Create a new row for each value in the results list
                    for grid_split, result in enumerate(results):
                        new_row = {key: row[key] for key in non_split_columns}
                        new_row.update({
                            'results': result,
                            'split': split,
                            'scorer': scorer,
                            'grid_split': grid_split
                        })
                        new_rows.append(new_row)
                else:
                    # Handle single value case
                    new_row = {key: row[key] for key in non_split_columns}
                    new_row.update({
                        'results': results,
                        'split': split,
                        'scorer': scorer,
                        'grid_split': 0
                    })
                    new_rows.append(new_row)

    # Create a new DataFrame from the new rows
    new_df = pd.DataFrame(new_rows)

    return new_df


def process_files_in_directory(directory, processing_type='extract_benchopt'):
    """
    Process all CSV and Parquet files in a directory using the specified processing function.

    Parameters
    ----------
    directory : str
        Path to the directory containing the files to process
    processing_type : str, optional
        Type of processing to apply. Can be either 'retrieve_cv' or 'extract_benchopt'
        (default is 'extract_benchopt')

    Returns
    -------
    pandas.DataFrame
        Concatenated DataFrame containing all processed results

    Raises
    ------
    ValueError
        If processing_type is not one of 'retrieve_cv' or 'extract_benchopt'
    """
    if processing_type not in ['retrieve_cv', 'extract_benchopt']:
        raise ValueError(
            "processing_type must be either 'retrieve_cv' or 'extract_benchopt'"
        )

    df_list = []
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".csv") or filename.endswith(".parquet"):
                file_path = os.path.join(root, filename)
                print(f"Processing file: {file_path}")

                if processing_type == 'retrieve_cv':
                    df = retrieve_cv(file_path)
                else:  # extract_benchopt
                    df = extract_benchopt_file_informations(file_path)

                df_list.append(df)
            else:
                print(
                    f"Ignoring file: {filename} (not a .csv or .parquet file)"
                )

    print('\n')
    # Concatenate all DataFrames in the list
    all_df = pd.concat(df_list)

    return all_df


def keep_only_best_scorer_per_estimator(df, specific_col=None):
    df_copy = df.copy()
    df_copy['estimator'] = [index_tuple[1] for index_tuple in df.index]
    df_copy['dataset'] = [index_tuple[0] for index_tuple in df.index]
    df_copy = df_copy.reset_index(drop=True)

    mean_index = None
    if not specific_col:
        for i in range(0, len(df.columns)):
            if df.columns[i][2] == 'mean':
                mean_index = i
                break
    else:
        mean_index = df.columns.get_loc(specific_col)

    if mean_index is None:
        raise ValueError('No mean column found')

    max_indices = df_copy.groupby(
        [df_copy['dataset'], 'estimator']
    )[[df_copy.columns[mean_index]]].idxmax()
    max_indices = max_indices.dropna()
    max_rows = df_copy.iloc[max_indices.values.flatten()]

    max_rows = max_rows.set_index(['dataset', 'estimator'])

    return max_rows


def regex_match(regex, string):
    result = re.match(regex, string)
    new_string = string
    if result:
        source = result.group(1)
        target = result.group(2)

        if source in SOURCE_TARGET_ACRONYMS.keys():
            source = SOURCE_TARGET_ACRONYMS[source]

        if target in SOURCE_TARGET_ACRONYMS.keys():
            target = SOURCE_TARGET_ACRONYMS[target]

        if '_rank' in string:
            new_string = f"{source}" + r'$\rightarrow$' + f"{target} rank"
        else:
            new_string = f"{source}" + r'$\rightarrow$' + f"{target}"

    return new_string
