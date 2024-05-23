"""
Script to generate LaTeX tables from CSV or Parquet files
# This script generates LaTeX tables from the results of the experiments

Example usage:
    # For Binary simulted dataset
    # python generate_table_results.py
    # ../output_choleski/outputs_binary_simulated
    # --dataset simulated --shift_name shift --dataset_params binary

    # For Office31Surf dataset
    # python generate_table_results.py ../output_choleski/outputs_office31surf
    # --dataset office31surf

    # For mnist_usps dataset
    # python generate_table_results.py ../output_choleski/outputs_mnist_usps
    # --dataset mnist_usps

    # For 20news dataset
    # python generate_table_results.py ../output_choleski/outputs_20newsgroup
    # --dataset 20newsgroup

    # For Mushroom dataset
    # python generate_table_results.py ../output_choleski/outputs_mushrooms
    # --dataset mushrooms

    # For Office31Decaf dataset
    # python generate_table_results.py ../output_choleski/outputs_office31decaf
    # --dataset office31decaf
"""

import argparse
import os
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

SCORER_DICT = {
    'supervised_scorer': 'SS',
    'importance_weighted': 'IWG',
    'soft_neighborhood_density': 'SND',
    'deep_embedded_validation': 'DV',
    'circular_validation': 'CircV',
    'prediction_entropy': 'PE',
}

DA_TECHNIQUES = {
    'NO DA': [
        'NO_DA_SOURCE_ONLY',
        'NO_DA_TARGET_ONLY'
    ],
    'Reweighting': [
        'gaussian_reweight',
        'KLIEP',
        'discriminator_reweight',
        'KMM',
        "TarS",
        'density_reweight',
        'nearest_neighbor_reweight',
    ],
    'Mapping': [
        'CORAL',
        'MMDSConS',
        'linear_ot_mapping',
        'entropic_ot_mapping',
        'ot_mapping',
        'class_regularizer_ot_mapping',
    ],
    'Subspace': [
        'transfer_component_analysis',
        'subspace_alignment',
        'transfer_subspace_learning',
        'joint_distribution_adaptation',
        'conditional_transferable_components',
        'transfer_joint_matching',
        'PCA',
    ],
    'Other': [
        'JDOT_SVC',
        'DASVM',
        'OTLabelProp'
    ],
}

ESTIMATOR_DICT = {
    'CORAL': 'CORAL',
    'JDOT_SVC': 'JDOT',
    'KLIEP': 'KLIEP',
    'PCA': 'JPCA',
    'discriminator_reweight': 'Disc. RW',
    'entropic_ot_mapping': 'EntOT',
    'gaussian_reweight': 'Gauss. RW',
    'linear_ot_mapping': 'LinOT',
    'ot_mapping': 'MapOT',
    'subspace_alignment': 'SA',
    'transfer_component_analysis': 'TCA',
    'transfer_joint_matching': 'TJM',
    'transfer_subspace_learning': 'TSL',
    'joint_distribution_adaptation': 'JDA',
    'conditional_transferable_components': 'CTC',
    'class_regularizer_ot_mapping': 'ClassRegOT',
    'MMDSConS': 'MMD-LS',
    'TarS': 'MMDTarS',
    'KMM': 'KMM',
    'NO_DA_SOURCE_ONLY': 'Train Src',
    'NO_DA_TARGET_ONLY': 'Train Tgt',
    'DASVM': 'DASVM',
    'density_reweight': 'Dens. RW',
    'nearest_neighbor_reweight': 'NN RW',
    'OTLabelProp': 'OTLabelProp'
}

SHIFT_ACRONYMS = {
    'covariate_shift': 'Cov. shift',
    'target_shift': 'Targ. shift',
    'concept_drift': 'Con. drift',
    'subspace': 'Subspace',
}

SOURCE_TARGET_ACRONYMS = {
    'amazon': 'amz',
    'caltech': 'cal',
    'dslr': 'dsl',
    'webcam': 'web',
    'enlarging': 'enl',
    'tapering': 'tap',
}

REGEX_SOURCE_TARGET = r"\('(.+)', '(.+)'\)"

TABLE_DESCRIPTIONS = {
    'supervised_scorer_table.tex': (
        'Supervised scorer table. '
        'Color-code: the darker the result, the better. '
        'Bold value: best value per shift.'
    ),
    'other_scorers_table.tex': (
        'Best realistic scorer table. '
        'Color-code: the darker the result, the better. '
        'Bold value: best value per shift.'
    ),
    'delta_table.tex': (
        'Performance loss between supervised and best scorer table '
        '(the delta is computed as best scorer accuracy '
        '- supervised accuracy). '
        'Color-code: the darker the result, the better. '
        'Bold value: best value per shift.'
    ),
    'best_scorer_table.tex': 'Best scorer per estimator table',
    'best_scorer_per_shift_per_estimator_table.tex': (
        'Best scorer per dataset table'
    ),
    'solvers_vs_datasets_table.tex': (
        'Performance of all the estimators on all the datasets table. '
        'Color-code: the darker the result, the better. '
        'Bold value: best value per dataset.'
    ),
    'solvers_vs_datasets_delta_table.tex': (
        'Performance loss of all the estimators on all the datasets table '
        'between the supervised and best scorer table.'
        '(the delta is computed as best estimator accuracy '
        '- supervised accuracy). '
        'Color-code: the darker the result, the better. '
        'Bold value: best value per dataset.'
    ),
    'solvers_vs_datasets_relative_perf_table.tex': (
        'Relative performance of all the estimators on all the datasets table.'
        ' (train on source accuracy - train with estimator accuracy). '
        'Color-code: the darker the result, the better. '
        'Bold value: best value per dataset.'
    ),
    'solvers_vs_datasets_relative_perf_normalized_table.tex': (
        'Normalized relative performance of all the estimators on'
        'all the datasets table. '
        '(train with estimator accuracy - train on source accuracy) '
        '/ (train on target accuracy - train on source accuracy). '
        'Color-code: the darker the result, the better. '
        'Bold value: best value per dataset.'
    ),
    }

TABLES_TO_PUT_IN_BOLD = [
    'supervised_scorer_table.tex',
    'other_scorers_table.tex',
    'delta_table.tex',
    'solvers_vs_datasets_table.tex',
    'solvers_vs_datasets_delta_table.tex',
    'solvers_vs_datasets_relative_perf_table.tex',
    'solvers_vs_datasets_relative_perf_normalized_table.tex',
]

ADD_PLUS_SIGN_TABLES = [
    'delta_table.tex',
    'solvers_vs_datasets_relative_perf_table.tex',
    'solvers_vs_datasets_relative_perf_normalized_table.tex',
]


def generate_df(file_path):
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

    # Group by 'data_name' and 'solver_name'
    grouped_df = df.groupby(['data_name', 'solver_name'])

    # Exclude specific columns and select only the metrics columns
    exclude_columns = [
        'objective_name',
        'objective_cv_results',
        'objective_value'
    ]
    metrics_columns = [
        col
        for col in df.columns
        if col.startswith('objective_') and col not in exclude_columns
    ]

    # Compute mean and standard deviation for each metric
    grouped_df = grouped_df[metrics_columns].agg(['mean', 'std'])

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

    # multi_index = \
    #    pd.MultiIndex.from_tuples(new_df.columns.values[:-1]).append(pd.Index(['scorer']))
    # new_df.columns = multi_index
    return new_df


def tabulate_but_better_estimator_index(df, latex_file_name):
    # Remove values equal to 'nan'
    df = df.replace('nan', np.nan)
    df = df.replace('None', np.nan)

    def shade_of_grey(mean_value, df_value, min_value=0, max_value=1):
        intensity_range = (40, 100)

        if mean_value == 'nan' or np.isnan(mean_value):
            # Return the nan value as is but in the lightest grey
            return (
                '\\textcolor{greyshade!%d}{%s}' %
                (intensity_range[0], df_value)
            )
        else:
            intensity = int(
                intensity_range[0] +
                (intensity_range[1] - intensity_range[0]) *
                (mean_value - min_value) /
                (max_value - min_value)
            )
            return '\\textcolor{greyshade!%d}{%s}' % (intensity, df_value)

    def shade_of_green_red(
        mean_value,
        df_value,
        min_value=0,
        max_value=1,
        train_on_source_mean=0.5,
        train_on_source_std=0,
        is_delta_table=False,
        skip_std=False,
    ):
        # If is_delta_table, we want green > 0
        # red < 0 and transparent = 0
        if is_delta_table:
            train_on_source_mean = 0
            train_on_source_std = 0
        
        if skip_std:
            train_on_source_std = 0

        # Intensity range for the green and red colors
        intensity_range = (10, 90)
        green_threshold = train_on_source_mean

        if mean_value == 'nan' or np.isnan(mean_value):
            # Return the nan value
            return df_value
        elif mean_value > (green_threshold + train_on_source_std):
            green_min = green_threshold
            green_max = max_value
            if green_max - green_min == 0:
                # To avoid division by zero
                intensity = intensity_range[1]
            else:
                intensity = int(
                    intensity_range[0] +
                    (intensity_range[1] - intensity_range[0]) *
                    (mean_value - green_min) /
                    (green_max - green_min)
                )
            return '\\cellcolor{green!%d}{%s}' % (intensity, df_value)
        elif mean_value < (green_threshold - train_on_source_std):
            # No color if value = 0 for the delta table
            if is_delta_table and mean_value == 0:
                return df_value

            red_min = min_value
            red_max = green_threshold
            if red_min - red_max == 0:
                # To avoid division by zero
                intensity = intensity_range[1]
            else:
                intensity = int(
                    intensity_range[0] +
                    (intensity_range[1] - intensity_range[0]) *
                    (mean_value - red_max) /
                    (red_min - red_max)
                )
            return '\\cellcolor{red!%d}{%s}' % (intensity, df_value)
        else:
            return df_value

    # Put in bold the best value for each shift
    if latex_file_name in TABLES_TO_PUT_IN_BOLD:
        # Otherwise we don't have shift columns but more a best_scorer
        # column or something like that
        for shift in df.columns:
            filtered_da_techniques = {
                key: value
                for key, value in DA_TECHNIQUES.items()
                if key != 'NO DA'
            }
            
            # Add a '+' sign to the values that are not negative
            if latex_file_name in ADD_PLUS_SIGN_TABLES:
                for value_index in df[shift].index:
                    df.loc[value_index][shift] = add_plus_sign(
                        df.loc[value_index][shift]
                    )

            means = df[shift].apply(
                lambda x: float(x.split(' ± ')[0])
                if not pd.isna(x) else x
            )

            max_index = means.dropna().idxmax()

            # We add a shade of green and red to the values
            # depending on their value
            for value_index in df[shift].index:
                # df.loc[value_index][shift] = shade_of_grey(
                #     means.loc[value_index],
                #     df.loc[value_index][shift],
                #     min_value=means.min(),
                #     max_value=means.max()
                # )
                if latex_file_name in [
                    'solvers_vs_datasets_relative_perf_table.tex',
                    'solvers_vs_datasets_relative_perf_normalized_table.tex',
                ]:
                    # We don't have the train_on_source value for
                    # solvers_vs_datasets_relative_perf_* tables
                    train_on_source_mean = 0
                    train_on_source_std = 0
                else:
                    train_on_source_mean = means.loc[
                        ESTIMATOR_DICT.get(
                            'NO_DA_SOURCE_ONLY',
                            'NO_DA_SOURCE_ONLY'
                        )
                    ]

                    if not latex_file_name == 'delta_table.tex':
                        train_on_source_std = (
                            df[shift].loc[
                                ESTIMATOR_DICT.get(
                                    'NO_DA_SOURCE_ONLY',
                                    'NO_DA_SOURCE_ONLY'
                                )
                            ]
                        )
                        if pd.isna(train_on_source_std):
                            train_on_source_std = 0
                        else:
                            train_on_source_std = float(
                                train_on_source_std.split(' ± ')[1]
                            )
                    else:
                        train_on_source_std = 0
                
                # Since in the mean col we have huge stds
                # We should not consider it when
                # coloring our tables
                skip_std = True if shift == 'Mean' else False

                df.loc[value_index][shift] = shade_of_green_red(
                    means.loc[value_index],
                    df.loc[value_index][shift],
                    min_value=means.min(),
                    max_value=means.max(),
                    train_on_source_mean=train_on_source_mean,
                    train_on_source_std=train_on_source_std,
                    is_delta_table=latex_file_name == 'delta_table.tex',
                    skip_std=skip_std,
                )

            # We put the best value in bold
            formatted_max_value = "\\textbf{" + df.loc[max_index][shift] + "}"

            df.loc[max_index][shift] = formatted_max_value

    index_is_estimators = any(
        value in df.index
        for value in ESTIMATOR_DICT.values()
    )

    def escape_underscore(s):
        if type(s) is str or type(s) is np.str_:
            return s.replace('_', r'\_').replace('->', r'$\rightarrow$')
        elif type(s) is list or type(s) is np.ndarray:
            return [escape_underscore(x) for x in s]

    table_name = latex_file_name.split('.')[0]
    table_name = escape_underscore(table_name)

    latex_file = ''
    latex_file += "\\begin{table}[H]" + '\n'
    latex_file += "\\centering" + '\n'
    latex_file += "\\renewcommand{\\arraystretch}{1.5}" + '\n'
    # latex_file += "\\makebox[1 \\textwidth][c]{" + '\n'
    # latex_file += "\\resizebox{1.6 \\textwidth}{!}{" + '\n'
    latex_file += (
        "\\begin{tabular}" + '{c|l' +
        ('|c' * len(df.columns)) + "|}" + '\n'
    )

    if index_is_estimators:
        # latex_file += '\\hline' + '\n'

        def column_rotate(col_name_list):
            # \mcrot{1}{|c}{60}{\textbf{Con\_drift}}
            return [
                "\\mcrot{1}{|c|}{60}{\\textbf{" + col_name + "}}"
                for col_name in col_name_list
            ]

        latex_file += (
            '& & ' + ' & '
            .join(column_rotate(escape_underscore(df.columns.values))) +
            r'\\' + '\n'
        )

        current_da_type = None
        for key, estimators in DA_TECHNIQUES.items():
            # First check if this category is present in the DataFrame
            if not any(
                ESTIMATOR_DICT.get(estimator, estimator) in df.index
                for estimator in estimators
            ):
                continue

            if key != current_da_type:
                latex_file += '\\hline\\hline' + '\n'
                # \multirow{2}{*}{{\rotatebox{90}{\textbf{NO DA}}}}
                latex_file += (
                    "\\multirow{" + str(len(DA_TECHNIQUES[key])) +
                    "}{*}{{\\rotatebox{90}{\\textbf{" + key + "}}}}"
                )
                current_da_type = key

            for estimator in estimators:
                estimator = ESTIMATOR_DICT.get(estimator, estimator)
                if estimator in df.index:
                    latex_file += " & " + f"{escape_underscore(estimator)} & "
                    for col in df.columns:
                        latex_file += (
                            f"{escape_underscore(df.loc[estimator, col]) } & "
                        )
                    latex_file = latex_file[:-2] + r'\\' + '\n'

    else:
        # Here we have the Best scorer per shift per estimator table
        latex_file += '\\hline' + '\n'

        column_names = [ESTIMATOR_DICT.get(col, col) for col in df.columns]
        column_names = [escape_underscore(col) for col in column_names]
        latex_file += '&' + ' & '.join(column_names) + r'\\' + '\n'

        for index in df.index:
            latex_file += f"{escape_underscore(index)} & "
            for col in df.columns:
                latex_file += f"{escape_underscore(df.loc[index, col])} & "
            latex_file = latex_file[:-2] + r'\\' + '\n'

    latex_file += '\\hline' + '\n'
    latex_file += '\\end{tabular}' + '\n'
    # latex_file += '}' + '\n'
    # latex_file += '}' + '\n'
    latex_file += "\\caption{{{}}}".format(
        TABLE_DESCRIPTIONS.get(latex_file_name, table_name)
    ) + '\n'
    latex_file += "\\end{table}"

    return latex_file


def generate_latex_table(
    df,
    folder='other_datasets',
    latex_file_name='table_latex.tex'
):
    # Beautify the DataFrame
    df_copy = df.copy()
    df_copy = beautify_df(df_copy)

    df_str = df_copy.applymap(lambda x: str(x).replace('+', r'+'))

    # Convert DataFrame to LaTeX table
    latex_table = tabulate_but_better_estimator_index(df_str, latex_file_name)

    latex_file_name = folder + '_' + latex_file_name
    latex_file_name = os.path.join(folder, latex_file_name)
    os.makedirs(folder, exist_ok=True)

    with open(latex_file_name, 'w') as f:
        f.write(latex_table)

    print(f'LaTeX table saved to {latex_file_name}')
    return latex_table


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


def add_plus_sign(x):
    # For the delta table we add a '+' sign to the values that are not negative
    if x is np.nan:
        return x
    elif x.startswith('-'):
        return x
    else:
        return "+" + x


def beautify_df(df):
    df.index = [
        ESTIMATOR_DICT.get(estimator, estimator) for estimator in df.index
    ]

    is_simulated_dataset = False
    new_columns = []  # New list to hold modified column names
    for column in df.columns:
        new_column = column
        if column in SHIFT_ACRONYMS.keys():
            new_column = SHIFT_ACRONYMS[column]
            is_simulated_dataset = True

        if column in SHIFT_ACRONYMS.values():
            # The case where the col names are already cleaned
            is_simulated_dataset = True

        new_column = regex_match(REGEX_SOURCE_TARGET, new_column)
        # Append modified column name to the list
        new_columns.append(new_column)
    # Assign the new list of column names to the DataFrame's columns
    df.columns = new_columns

    # We sort the columns based on the SHIFT_ACRONYMS keys
    if is_simulated_dataset:
        # Filter out columns present in SHIFT_ACRONYMS.values()
        columns_to_sort = [
            col for col in df.columns if col in SHIFT_ACRONYMS.values()
        ]

        # Sort columns based on their index in SHIFT_ACRONYMS.values()
        sorted_columns = sorted(
            columns_to_sort,
            key=lambda x: list(SHIFT_ACRONYMS.values()).index(x)
        )

        # Get columns not present in SHIFT_ACRONYMS.values()
        remaining_columns = [
            col for col in df.columns if col not in columns_to_sort
        ]

        # Concatenate sorted columns with remaining columns
        final_columns = sorted_columns + remaining_columns

        # Reindex dataframe with sorted columns at the beginning
        # and remaining columns at the end
        df = df.reindex(columns=final_columns)

    new_indexes = []  # New list to hold modified index names
    for index in df.index:
        new_index = index
        if index in SHIFT_ACRONYMS.keys():
            new_index = SHIFT_ACRONYMS[index]

        new_index = regex_match(REGEX_SOURCE_TARGET, new_index)
        # Append modified index name to the list
        new_indexes.append(new_index)
    # Assign the new list of index names to the DataFrame's index
    df.index = new_indexes

    # Change name of scorers
    if 'best_scorer' in df.columns:
        df['best_scorer'] = df['best_scorer'].apply(
            lambda x: ', '.join(
                [SCORER_DICT[scorer] for scorer in x]
            )
        )

    return df


def generate_all_tables(
    df,
    dataset='simulated_shifts',
    dataset_params=['binary'],
    metric='Accuracy',
    train_test='test',
    source_target='target',
    shift_name='shift',
    exlude_solvers=[],
    supervised_scorer_only=False,
):

    dataset = dataset.lower()
    metric = metric.lower()
    train_test = train_test.lower()
    source_target = source_target.lower()
    shift_name = shift_name.lower()
    dataset_params = [param.lower() for param in dataset_params]

    exlude_solvers = [solver.lower() for solver in exlude_solvers]

    # We remove '[param_grid=default]' in each method name
    df.index = df.index.map(lambda x: (x[0], x[1].split('[param_grid=default]')[0]))

    # We keep only the rows with the dataset in the index
    filtered_df = df[
        [dataset in index_tuple[0].lower() for index_tuple in df.index]
    ]

    # We remove the rows with the exlude_solvers in the index
    for exlude_solver in exlude_solvers:
        filtered_df = filtered_df[
            [
                exlude_solver not in index_tuple[1].lower()
                for index_tuple in filtered_df.index
            ]
        ]

    # We keep only the rows with the dataset_params in the index
    for dataset_param in dataset_params:
        filtered_df = filtered_df[
            [
                dataset_param in index_tuple[0].lower()
                for index_tuple in filtered_df.index
            ]
        ]

    # We keep only the columns with the metric + the scorer column
    filtered_columns = [
        col for col in df.columns
        if (
            (source_target + '_' + metric) == col[0].lower()
            and train_test in col[1].lower()
        )
    ]
    filtered_columns.append('scorer')
    filtered_df = filtered_df.loc[:, filtered_columns]

    ###########################################
    # Keep only the best "source_target + '_' + metric"
    # scorer for each estimator
    ###########################################

    # We keep only the "superivsed" scorer
    supervised_scorer_df = filtered_df[filtered_df['scorer'] == 'supervised']

    # First we remove the "supervised" scorer
    filtered_df = filtered_df[filtered_df['scorer'] != 'supervised']

    # We keep only the best unsupervised scorer for each estimator
    # in best_unsupervised_df. Except for the NO_DA methods,
    # where we keep the supervised scorer
    best_unsupervised_df = keep_only_best_scorer_per_estimator(filtered_df)

    # Remove NO_DA methods from the best_unsupervised df
    no_da_methods = [solver for solver in DA_TECHNIQUES['NO DA']]
    best_unsupervised_df = best_unsupervised_df[
        ~best_unsupervised_df.index.get_level_values(1).isin(no_da_methods)
    ]

    # For the NO_DA methods, we use the results from the supervised scorer
    no_da_methods = [solver for solver in DA_TECHNIQUES['NO DA']]
    no_da_df = supervised_scorer_df[supervised_scorer_df.index.get_level_values(1).isin(no_da_methods)]

    # Concatenate the two dataframes
    best_scores_df = pd.concat([best_unsupervised_df, no_da_df])
    best_scores_df['scorer'] = "best_scorer"

    ###########################################
    supervised_scorer_df = create_solver_vs_shift_df(
        supervised_scorer_df,
        shift_name
    )
    # supervised_scorer_df = compute_avg_ranking(supervised_scorer_df)

    best_scores_df = create_solver_vs_shift_df(best_scores_df, shift_name)
    # best_unsupervised_df = compute_avg_ranking(best_unsupervised_df)

    all_latex_tables = {}

    table_prefix = '_'.join([dataset] + dataset_params)
    if supervised_scorer_only:
        print('Computing supervised scorer only DataFrame')
        supervised_scorer_table = generate_latex_table(
            supervised_scorer_df,
            folder=table_prefix,
            latex_file_name='supervised_scorer_table.tex'
        )
        all_latex_tables['Supervised scorer'] = supervised_scorer_table
    else:
        print('Computing best unsupervised scorer DataFrame')
        best_unsupervised_table = generate_latex_table(
            best_scores_df,
            folder=table_prefix,
            latex_file_name='other_scorers_table.tex'
        )
        all_latex_tables['Best unsupervised scorer'] = best_unsupervised_table

    print('\n')
    print('Computing delta between the supervised scorer and the best scorer:')
    diff_df = compute_delta_supervised_best_scorer_df(
        supervised_scorer_df,
        best_scores_df
    )
    delta_table = generate_latex_table(
        diff_df,
        folder=table_prefix,
        latex_file_name='delta_table.tex'
    )
    all_latex_tables['Delta between supervised and best scorer'] = delta_table

    print('\n')
    print('Computing best scorer for each estimator:')
    best_scorer_for_estimator = compute_best_scorer(filtered_df)
    best_scorer_table = generate_latex_table(
        best_scorer_for_estimator,
        folder=table_prefix,
        latex_file_name='best_scorer_table.tex'
    )
    all_latex_tables['Best scorer per estimator'] = best_scorer_table

    print('\n')
    print('Computing best scorer per shift per estimator')
    best_scorer_per_shift_per_estimator = (
        compute_best_scorer_per_estimator_per_shift(filtered_df, shift_name)
    )
    best_scorer_per_shift_per_estimator_table = generate_latex_table(
        best_scorer_per_shift_per_estimator,
        folder=table_prefix,
        latex_file_name='best_scorer_per_shift_per_estimator_table.tex'
    )
    all_latex_tables['Best scorer per dataset'] = (
        best_scorer_per_shift_per_estimator_table
    )

    generate_latex_all_tables(all_latex_tables, table_prefix)

    # Plot the relative performances per estimator
    plot_relative_performances_per_estimator(
        best_scores_df,
        folder=table_prefix,
        plot_file_name='relative_performances_per_estimator.jpg',
        normalize=False
    )

    # Normalized version of the plot
    plot_relative_performances_per_estimator(
        best_scores_df,
        folder=table_prefix,
        plot_file_name='relative_performances_per_estimator_normalized.jpg',
        normalize=True
    )

    plot_accuracy_vs_shifts(
        best_scores_df,
        folder=table_prefix,
        plot_file_name='accuracy_vs_shift.jpg'
    )

    print('\n')
    print('Parameters:')
    print(f"Dataset: {dataset}")
    print(f"Dataset parameters: {dataset_params}")
    print(f"Metric: {metric}")
    print(f"Train/test: {train_test}")
    print(f"Source/target: {source_target}")
    print(f"Shift name: {shift_name}")
    print(f"Exclude solvers: {exlude_solvers}")
    print(f"Supervised scorer only: {supervised_scorer_only}")

    return best_unsupervised_df


def generate_latex_all_tables(all_latex_tables, dataset_name):
    latex_file = ''

    section_name = dataset_name.replace('_', '\\_')
    latex_file += "\\section{{{}}}".format(section_name)

    latex_file += '\n'
    for table_name, table in all_latex_tables.items():
        latex_file += "\\subsection{{{}}}".format(table_name)
        latex_file += '\n\n'
        latex_file += table
        latex_file += '\n\n'

    folder = dataset_name

    latex_file_name = dataset_name + '_all_tables.tex'
    latex_file_name = os.path.join(folder, latex_file_name)
    os.makedirs(folder, exist_ok=True)

    with open(latex_file_name, 'w') as f:
        f.write(latex_file)

    print(f'LaTeX all tables saved to {latex_file_name}')


def keep_only_best_scorer_per_estimator(df, specific_col = None):
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
    
    max_indices = df_copy.groupby([df_copy['dataset'], 'estimator'])[[df_copy.columns[mean_index]]].idxmax()
    max_rows = df_copy.iloc[max_indices.values.flatten()]

    max_rows = max_rows.set_index(['dataset', 'estimator'])

    return max_rows


def compute_delta_supervised_best_scorer_df(supervised_df, best_scorer_df):
    ###########################################
    ###########################################

    ###########################################
    # Compute the delta between the supervised
    # scorer and the best scorer
    ###########################################
    mask_exlude_columns = (
        supervised_df.columns.str.lower().str.contains('rank')
    )
    columns_to_drop = supervised_df.columns[mask_exlude_columns]

    supervised_df = supervised_df.drop(columns=columns_to_drop)
    best_scorer_df = best_scorer_df.drop(columns=columns_to_drop)

    # Create a new DataFrame to hold the percentage difference
    diff_df = pd.DataFrame(
        columns=supervised_df.columns,
        index=supervised_df.index
    )

    # Iterate over each row and column to calculate percentage difference
    for col in supervised_df.columns:
        for idx in supervised_df.index:
            if idx not in best_scorer_df.index:
                diff_df.at[idx, col] = None
            elif (
                supervised_df.at[idx, col] is np.nan or
                best_scorer_df.at[idx, col] is np.nan
            ):
                diff_df.at[idx, col] = None
            else:
                val1 = float(supervised_df.at[idx, col].split(' ±')[0])
                val2 = float(best_scorer_df.at[idx, col].split(' ±')[0])

                # We loose diff point in accuracy when switching
                # from supervised_df to best_scorer_df
                diff = val2 - val1
                # if diff > 0:
                #     diff = f"+{diff:.2f}"
                # else:
                #     diff = f"{diff:.2f}"
                diff_df.at[idx, col] = round(diff, 2)

    ###########################################
    ###########################################
    return diff_df


def compute_best_scorer(df):
    ###########################################
    ###########################################

    ###########################################
    # Get the best scorer for each estimator
    ###########################################

    df['estimator'] = [index_tuple[1] for index_tuple in df.index]
    df = df.reset_index(drop=True)
    for col in df.columns:
        if col[2] == 'mean':
            # Drop the column with the 'std' metric
            df = df.drop(columns=(col[0], col[1], 'std'))
            break

    # Now df looks like this:
    # (target_accuracy, test, mean) | scorer | estimator
    # Group by 'scorer' and 'estimator', calculate
    # the mean of the metric for each group
    grouped = df.groupby(['scorer', 'estimator']).mean()

    best_scorers = []
    for estimator, group in grouped.groupby('estimator'):
        max_mean = group.max().iloc[0]
        best_scorers.append(
            list(
                group[group[df.columns[0]] == max_mean]
                .index.get_level_values('scorer')
            )
        )

    best_scorer_for_estimator = pd.DataFrame(
        {
            'estimator': grouped.index.levels[1],
            'best_scorer': best_scorers,
        }
    )
    best_scorer_for_estimator = (
        best_scorer_for_estimator.set_index('estimator')
    )

    return best_scorer_for_estimator
    ###########################################
    ###########################################


def compute_best_scorer_per_estimator_per_shift(df, shift_name):
    ###########################################
    ###########################################

    ###########################################
    # Get the best scorer per shift for each estimator
    ###########################################

    # We extract the shift
    df['shift'] = None
    for index_tuple in df.index:
        shift = index_tuple[0].split(shift_name + '=')[1].split(']')[0]
        df.loc[[index_tuple], 'shift'] = shift

    df['estimator'] = [index_tuple[1] for index_tuple in df.index]
    df = df.reset_index(drop=True)
    for col in df.columns:
        if col[2] == 'mean':
            # Drop the column with the 'std' metric
            df = df.drop(columns=(col[0], col[1], 'std'))
            break

    # Now df looks like this:
    # (target_accuracy, test, mean) | scorer | shift | estimator
    # Group by 'shift' and 'estimator', calculate the scorer
    # where (target_accuracy, test, mean) is max
    grouped = df.groupby(['shift', 'estimator'])

    estimator_list = np.unique(df['estimator'])
    shift_list = np.unique(df['shift'])
    index = pd.Index(shift_list, name='Shift')

    best_scorer_per_shift_per_estimator = pd.DataFrame(
        index=estimator_list,
        columns=index
    )

    for (shift, estimator), group in grouped:
        max_mean = group.max().iloc[0]
        best_scorer = group[group[df.columns[0]] == max_mean]['scorer'].values
        best_scorer = np.unique(best_scorer)
        best_scorer = ', '.join(
            [SCORER_DICT[scorer]
             for scorer in best_scorer]
        )
        best_scorer_per_shift_per_estimator[shift][estimator] = best_scorer

    return best_scorer_per_shift_per_estimator


def create_solver_vs_shift_df(filtered_df, shift_name):
    # We create a new DataFrame with the right format
    indexes_name = []
    columns_name = []
    for index_tuple in filtered_df.index:
        indexes_name.append(index_tuple[1])
        columns_name.append(
            index_tuple[0].split(shift_name + '=')[1].split(']')[0]
        )

    indexes_name = np.unique(indexes_name)
    columns_name = np.unique(columns_name)

    output_df = pd.DataFrame(columns=columns_name, index=indexes_name)

    # We fill the new DataFrame
    for idx, row in filtered_df.iterrows():
        estimator = idx[1]
        shift = idx[0].split(shift_name + '=')[1].split(']')[0]

        mean = None
        std = None
        for metric_index, row_value in enumerate(row.index):
            # Check if the third element of the tuple is 'mean' or 'std'
            if row_value[2] == 'mean':
                mean = row.iloc[metric_index]
            elif row_value[2] == 'std':
                std = row.iloc[metric_index]

        value = f"{mean:.2f} ± {std:.2f}"
        output_df.loc[estimator, shift] = value

    # Add the MEAN column
    ###########################################
    if shift_name != 'shift':
        # Extracting the values and converting them to float
        output_df_means_only = output_df.applymap(
            lambda x: float(x.split(' ± ')[0])
            if x is not np.nan else np.nan
        )

        # Calculating the mean + std of each row
        mean_values = output_df_means_only.mean(axis=1)
        std_values = output_df_means_only.std(axis=1)

        # Creating the 'mean ± std' string
        mean_with_std = [
            f"{mean:.2f} ± {std:.2f}"
            for mean, std in zip(mean_values, std_values)
        ]

        # Adding the 'Mean' column to the dataframe
        output_df['Mean'] = mean_with_std
    else:
        # For the simulated shifts we don't have a 'Mean' column
        pass
    ###########################################

    return output_df


def compute_avg_ranking(output_df):
    ###########################################
    # Categorize each estimator
    # to put them next to each other
    ###########################################

    # Sort the DataFrame by our custom order
    method_indices = {
        method: index_1*100 + index_2
        for index_1, (keys, methods) in enumerate(DA_TECHNIQUES.items())
        for index_2, method in enumerate(methods)
    }
    sorted_index = sorted(
        output_df.index,
        key=lambda x: method_indices.get(x, float('inf'))
    )
    output_df = output_df.reindex(sorted_index)

    ###########################################
    ###########################################

    ###########################################
    # Create a ranking column
    # it corresponds to the mean of the
    # ranking of each estimator for each shift
    ###########################################

    # Find Nans in the DataFrame
    rows_with_nan = output_df[output_df.isna().any(axis=1)]
    print(f"Rows with NaNs: {rows_with_nan.index}")
    print("Dropping them to compute the ranking")
    output_df = output_df.dropna(axis=0)

    output_df['Avg_Ranking'] = None
    number_of_shifts = len(output_df.columns) - 1
    for shift in output_df.columns[:-1]:
        output_df[shift+'_rank'] = (
            output_df[shift].str.split(' ±').str[0].astype(float)
        )
        output_df[shift+'_rank'] = (
            output_df[shift+'_rank'].rank(ascending=False, method='min')
        )
        output_df['Avg_Ranking'] = (
            output_df['Avg_Ranking'].add(
                output_df[shift+'_rank'],
                fill_value=0
            )
        )

    output_df['Avg_Ranking'] = output_df['Avg_Ranking'] / number_of_shifts
    ###########################################
    ###########################################

    return output_df


def process_files_in_directory(directory):
    df_list = []
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".csv") or filename.endswith(".parquet"):
                file_path = os.path.join(root, filename)
                print(f"Processing file: {file_path}")
                df = generate_df(file_path)
                df_list.append(df)
            else:
                print(
                    f"Ignoring file: {filename} (not a .csv or .parquet file)"
                )

    print('\n')
    # Concatenate all DataFrames in the list
    all_df = pd.concat(df_list)

    return all_df


def compute_relative_perf_df(df, normalize=False, handle_nan=True):
    """
    Compute the relative performance of each solver
    compared to NO_DA_TARGET_ONLY as a baseline.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to process.
    normalize : bool, optional
        Whether to normalize the values or not, by default False.
    handle_nan : bool, optional
        Whether to set NaN values to 0 or not. Useful to then plot
        the results but not good for a table, by default True.
    """
    # Compute the relative performance of each solver
    # compared to NO_DA_TARGET_ONLY as a baseline

    df = df.applymap(lambda x: '0.0 ± 0.0' if pd.isna(x) else x)
    df = df.apply(lambda x: x.str.split(' ± ').apply(lambda x: float(x[0])))    

    if 'NO_DA_SOURCE_ONLY' not in df.index:
        Warning('NO_DA_SOURCE_ONLY not found in the index \
            so we cannot plot the relative performances')
        return

    baseline_source = df.loc['NO_DA_SOURCE_ONLY']
    baseline_target = df.loc['NO_DA_TARGET_ONLY']
    df = df.drop(index=['NO_DA_SOURCE_ONLY', 'NO_DA_TARGET_ONLY'])

    # Compute relative performances
    for index, row in df.iterrows():
        if normalize:
            df.loc[index] = normalizer(row, baseline_source, baseline_target)
        else:
            df.loc[index] = round(row - baseline_source, 2)

    if not handle_nan:
        # We suppose that its impossible to have an accuracy of 0
        # Thus each time the acc is equal to 0 means that beforehand
        # It was a nan values
        # Not a great way to do it, but cleary the easiest one
        if normalize:
            zero_expected_value = normalizer(
                0,
                baseline_source,
                baseline_target
            )
        else:
            zero_expected_value = round(0 - baseline_source, 2)

        # Set all values equal to zero_expected_value to Nan
        df = df.mask(df.eq(zero_expected_value))

    # Now df corresponds to the relative performances

    # Sort the DataFrame by our custom order
    method_indices = {
        method: index_1*100 + index_2
        for index_1, (keys, methods) in enumerate(DA_TECHNIQUES.items())
        for index_2, method in enumerate(methods)
    }
    sorted_index = sorted(
        df.index,
        key=lambda x: method_indices.get(x, float('inf'))
    )
    df = df.reindex(sorted_index)

    new_columns = []  # New list to hold modified column names
    for column in df.columns:
        new_column = column
        if column in SHIFT_ACRONYMS.keys():
            new_column = SHIFT_ACRONYMS[column]

        new_column = regex_match(REGEX_SOURCE_TARGET, new_column)
        # Append modified column name to the list
        new_columns.append(new_column)
    # Assign the new list of column names to the DataFrame's columns
    df.columns = new_columns

    return df, baseline_source, baseline_target


def plot_accuracy_vs_shifts(df, folder, plot_file_name):
    # Define colormaps
    # Shape of the dict
    # cmap | number of estimator | current estimator color
    da_techniques_cmap = {
        'NO DA': [plt.cm.Reds, len(DA_TECHNIQUES['NO DA']), 0.5],
        'Reweighting': [plt.cm.Greens, len(DA_TECHNIQUES['Reweighting']), 0.5],
        'Subspace': [plt.cm.Blues, len(DA_TECHNIQUES['Subspace']), 0.5],
        'Mapping': [plt.cm.Purples, len(DA_TECHNIQUES['Mapping']), 0.5],
        'Other': [plt.cm.Oranges, len(DA_TECHNIQUES['Other']), 0.5],
    }

    # Sort the DataFrame by our custom order
    method_indices = {
        method: index_1*100 + index_2
        for index_1, (keys, methods) in enumerate(DA_TECHNIQUES.items())
        for index_2, method in enumerate(methods)
    }
    sorted_index = sorted(
        df.index,
        key=lambda x: method_indices.get(x, float('inf'))
    )
    df = df.reindex(sorted_index)

    # Split means and stds
    df = df.applymap(lambda x: '0.0 ± 0.0' if pd.isna(x) else x)
    means = df.apply(lambda x: x.str.split(' ± ').apply(lambda x: float(x[0])))
    stds = df.apply(lambda x: x.str.split(' ± ').apply(lambda x: float(x[1])))

    # Scatter plot
    plt.figure(figsize=(20, 16))

    if 'NO_DA_SOURCE_ONLY' not in df.index:
        Warning('NO_DA_SOURCE_ONLY not found in the index \
            so we cannot plot the horizontal train on source line')
    else:
        baseline_source = means.loc['NO_DA_SOURCE_ONLY']
        df = df.drop(index=['NO_DA_SOURCE_ONLY'])
        means = means.drop(index=['NO_DA_SOURCE_ONLY'])
        stds = stds.drop(index=['NO_DA_SOURCE_ONLY'])

    if 'NO_DA_TARGET_ONLY' not in df.index:
        Warning('NO_DA_TARGET_ONLY not found in the index \
            so we cannot plot the horizontal train on source line')
    else:
        baseline_target = means.loc['NO_DA_TARGET_ONLY']
        df = df.drop(index=['NO_DA_TARGET_ONLY'])
        means = means.drop(index=['NO_DA_TARGET_ONLY'])
        stds = stds.drop(index=['NO_DA_TARGET_ONLY'])

    new_columns = []  # New list to hold modified column names
    for column in df.columns:
        new_column = column
        if column in SHIFT_ACRONYMS.keys():
            new_column = SHIFT_ACRONYMS[column]

        new_column = regex_match(REGEX_SOURCE_TARGET, new_column)

        # Append modified column name to the list
        new_columns.append(new_column)
    # Assign the new list of column names to the DataFrame's columns
    df.columns = new_columns

    shifts = df.columns

    x = np.arange(len(shifts))  # the label locations
    width = 0.05  # the width of the bars
    multiplier = 0

    for estimator in means.index:
        color_map_key = None
        for key, value in DA_TECHNIQUES.items():
            if estimator in value:
                color_map_key = key
                break

        if color_map_key is None:
            color = 'gray'  # Default color if no matching key is found
        else:
            color_map = da_techniques_cmap[color_map_key][0]
            color = color_map(
                da_techniques_cmap[color_map_key][2] /
                da_techniques_cmap[color_map_key][1]
            )
            da_techniques_cmap[color_map_key][2] += 1

        offset = width * multiplier
        rects = plt.bar(
            x + offset,
            means.loc[estimator],
            width,
            label=ESTIMATOR_DICT.get(estimator, estimator),
            color=color,
        )
        plt.errorbar(
            x + offset,  # x position at the center of each bar
            means.loc[estimator],
            yerr=stds.loc[estimator],
            fmt='none',  # no marker on the error bars
            color=color,
        )
        plt.bar_label(rects, padding=10, labels=['']*len(x))
        multiplier += 1

    # Plot a vertical line between the shifts
    for i in range(1, len(shifts)):
        plt.axvline(
            x=i - 2.5*width,
            color='black',
            linewidth=3,
            linestyle='--'
        )

    # Plot the horizontal line for the baselines
    for i in range(0, len(shifts)):
        if baseline_source is not None:
            plt.axhline(
                y=baseline_source.iloc[i],
                xmin=normalizer(
                    i - 2.5*width,
                    -width,
                    len(shifts)-1 + offset + width,
                ),
                xmax=normalizer(
                    i + 1 - 2.5*width,
                    -width,
                    len(shifts)-1 + offset + width,
                ),
                color='red',
                linestyle='--',
                linewidth=1,
                label='Baseline (NO_DA_SRC)'
            )

        if baseline_target is not None:
            plt.axhline(
                y=baseline_target.iloc[i],
                xmin=normalizer(
                    i - 2.5*width,
                    -width,
                    len(shifts)-1 + offset + width,
                ),
                xmax=normalizer(
                    i + 1 - 2.5*width,
                    -width,
                    len(shifts)-1 + offset + width,
                ),
                color='green',
                linestyle='--',
                linewidth=1,
                label='Baseline (NO_DA_TGT)'
            )

    plt.ylabel('Accuracy')
    plt.title('Accuracy with respect to shift')
    plt.xlabel('Shift')
    plt.xticks(x + offset/2, shifts)
    plt.xlim(-width, len(shifts)-1 + offset + width)

    # To display the legend with only unique labels
    handles, labels = plt.gca().get_legend_handles_labels()
    labels, ids = np.unique(labels, return_index=True)
    handles = [handles[i] for i in ids]

    plt.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.2, 1))

    plt.yticks(np.arange(0, 1.1, 0.1))

    plt.tight_layout()

    plot_file_name = folder + '_' + plot_file_name
    plot_file_name = os.path.join(folder, plot_file_name)
    os.makedirs(folder, exist_ok=True)

    # Save the plot as a JPEG image
    plt.savefig(plot_file_name, format='jpeg')
    print(f'Plot Accuracy vs Shift saved to {plot_file_name}')
    plt.close()


def plot_relative_performances_per_estimator(
    df,
    folder,
    plot_file_name,
    normalize
):
    # Define colormaps
    # Shape of the dict
    # cmap | number of estimator | current estimator color
    da_techniques_cmap = {
        'NO DA': [plt.cm.Reds, len(DA_TECHNIQUES['NO DA']), 0.5],
        'Reweighting': [plt.cm.Greens, len(DA_TECHNIQUES['Reweighting']), 0.5],
        'Subspace': [plt.cm.Blues, len(DA_TECHNIQUES['Subspace']), 0.5],
        'Mapping': [plt.cm.Purples, len(DA_TECHNIQUES['Mapping']), 0.5],
        'Other': [plt.cm.Oranges, len(DA_TECHNIQUES['Other']), 0.5],
    }

    # Compute the relative performance of each solver
    df, baseline_source, baseline_target = compute_relative_perf_df(
        df,
        normalize=normalize
    )

    # Scatter plot
    # Plotting
    plt.figure(figsize=(20, 16))

    # Plotting the source baseline
    plt.axhline(
        y=0,
        color='red',
        linestyle='--',
        linewidth=1,
        label='Baseline (NO_DA_SRC)',
    )

    if normalize:
        plt.axhline(
            y=1,
            color='green',
            linestyle='--',
            linewidth=1,
            label='Baseline (NO_DA_TGT)',
        )

    shifts = df.columns

    x = np.arange(len(shifts))  # the label locations
    width = 0.05  # the width of the bars
    multiplier = 0

    for estimator in df.index:
        color_map_key = None
        for key, value in DA_TECHNIQUES.items():
            if estimator in value:
                color_map_key = key
                break

        if color_map_key is None:
            color = 'gray'  # Default color if no matching key is found
        else:
            color_map = da_techniques_cmap[color_map_key][0]
            color = color_map(
                da_techniques_cmap[color_map_key][2] /
                da_techniques_cmap[color_map_key][1]
            )
            da_techniques_cmap[color_map_key][2] += 1

        offset = width * multiplier

        rects = plt.bar(
            x + offset,
            df.loc[estimator],
            width,
            label=ESTIMATOR_DICT.get(estimator, estimator),
            color=color,
        )
        plt.bar_label(rects, padding=10, labels=['']*len(x))
        multiplier += 1

    if not normalize:
        # Plotting the target baseline
        baseline_target = baseline_target - baseline_source
        for i in range(0, len(shifts)):
            plt.axhline(
                y=baseline_target.iloc[i],
                xmin=normalizer(
                    i - 2.5*width, -width, len(shifts)-1 + offset + width
                ),
                xmax=normalizer(
                    i + 1 - 2.5*width, -width, len(shifts)-1 + offset + width
                ),
                color='green',
                linestyle='--',
                linewidth=1,
                label='Baseline (NO_DA_TGT)'
            )

    # Plot a vertical line between the shifts
    for i in range(1, len(shifts)):
        plt.axvline(
            x=i - 2.5*width, color='black', linewidth=3, linestyle='--'
        )

    if normalize:
        plt.ylabel('Normalized Relative performance')
        plt.title('Relative normalized performance gain with respect to shift')
    else:
        plt.ylabel('Relative performance')
        plt.title('Relative performance gain with respect to shift')
    plt.xlabel('Shift')
    plt.xticks(x + offset/2, shifts)
    plt.xlim(-width, len(shifts)-1 + offset + width)

    # To display the legend with only unique labels
    handles, labels = plt.gca().get_legend_handles_labels()
    labels, ids = np.unique(labels, return_index=True)
    handles = [handles[i] for i in ids]

    plt.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.2, 1))
    plt.tight_layout()

    plot_file_name = folder + '_' + plot_file_name
    plot_file_name = os.path.join(folder, plot_file_name)
    os.makedirs(folder, exist_ok=True)

    # Save the plot as a JPEG image
    plt.savefig(plot_file_name, format='jpeg')
    print(f'Plot Relative performance vs Shift saved to {plot_file_name}')
    plt.close()


def normalizer(row, min_values, max_values):
    # Function to normalize the values
    if isinstance(row, pd.Series):
        denominator = pd.concat(
            [max_values - min_values, min_values - max_values], axis=1
        ).max(axis=1)
        numerator = pd.concat(
            [row - min_values, row - max_values], axis=1
        ).max(axis=1)
        return round(numerator / denominator, 2)
    else:
        # We suppose that row is a float
        return round((row - min_values) / (max_values - min_values), 2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate LaTeX tables from CSV or Parquet files"
    )
    parser.add_argument(
        "directory",
        type=str,
        help="Path to the directory containing CSV or Parquet files"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset name to select the results",
        default='simulated_shifts'
    )
    parser.add_argument(
        "--dataset_params",
        nargs="+",
        help="Dataset parameters to select the results",
        default=[]
    )
    parser.add_argument(
        "--shift_name",
        type=str,
        help="Name of the shift parameter",
        default='source_target'
    )
    parser.add_argument(
        "--supervised_scorer_only",
        action="store_true",
        help="Keep only the supervised scorer"
    )
    parser.add_argument(
        "--exlude_solvers",
        nargs="+",
        help="Exclude specific solvers",
        default=[]
    )
    args = parser.parse_args()

    # Process files in the specified directory + Cleanup the DataFrames
    df = process_files_in_directory(args.directory)

    best_unsupervised_df = generate_all_tables(
        df,
        exlude_solvers=args.exlude_solvers,
        supervised_scorer_only=args.supervised_scorer_only,
        dataset=args.dataset,
        dataset_params=args.dataset_params,
        shift_name=args.shift_name
    )
