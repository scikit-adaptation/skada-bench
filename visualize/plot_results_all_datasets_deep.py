# %%
import numpy as np
import glob
import pandas as pd
import json
import argparse
from _solvers_scorers_registry import DEEP_ESTIMATOR_DICT, DEEP_DATASET_DICT


def shade_of_color_pvalue(
    df_value,
    min_value=0,
    mean_value=0,
    max_value=1,
    color_threshold=0.05,
):
    # Intensity range for the green and red colors
    intensity_range = (10, 60)

    if df_value == "nan" or np.isnan(df_value):
        # Return the nan value
        return df_value
    else:
        if df_value > mean_value:
            color_min = mean_value
            color_max = max_value
            if color_max - color_min == 0:
                # To avoid division by zero
                intensity = intensity_range[0]
            else:
                intensity = int(
                    intensity_range[0]
                    + (intensity_range[1] - intensity_range[0])
                    * (df_value - color_min)
                    / (color_max - color_min)
                )
            return "\\cellcolor{green_color!%d}{%s}" % (intensity, df_value)
        else:
            return df_value


def generate_table(csv_folder, scorer_selection="unsupervised"):
    # Load the data
    csv_files = glob.glob(f"{csv_folder}/*.csv")
    df = pd.concat([pd.read_csv(f) for f in csv_files])

    # %%
    df["target_accuracy-test-identity"] = (
        df["target_accuracy-test-identity"].apply(lambda x: json.loads(x))
    )

    df["nb_splits"] = df["target_accuracy-test-identity"].apply(
        lambda x: len(x)
    )
    df_target = df.query(
        'estimator == "deep_no_da_target_only" & scorer == "supervised"'
    )
    df_source = df.query(
        'estimator == "deep_no_da_source_only" & scorer == "supervised"'
    )
    df = df.merge(
        df_target[["shift", "target_accuracy-test-mean",
                   "target_accuracy-test-std"]],
        on="shift",
        suffixes=("", "_target"),
    )
    df = df.merge(
        df_source[
            [
                "shift",
                "target_accuracy-test-mean",
                "target_accuracy-test-std",
                "target_accuracy-test-identity",
            ]
        ],
        on="shift",
        suffixes=("", "_source"),
    )
    # remove duplicates
    df = df.drop_duplicates(subset=["dataset", "scorer", "estimator", "shift"])

    df["rank"] = df.groupby(["dataset", "scorer", "shift"])[
        "target_accuracy-test-mean"
    ].rank(ascending=False)
    df_rank = df.groupby(["estimator"])["rank"].mean().reset_index()

    df_mean = (
        df.groupby(["dataset", "type", "scorer", "estimator"])
        .agg(
            {
                "target_accuracy-test-mean": lambda x: x.mean(skipna=True),
                "target_accuracy-test-std": lambda x: x.mean(skipna=True),
                "rank": lambda x: x.mean(skipna=True),
            }
        )
        .reset_index()
    )
    df_source_mean = df_mean.query(
        "estimator == 'deep_no_da_source_only' & scorer == 'supervised'"
    )
    df_target_mean = df_mean.query(
        "estimator == 'deep_no_da_target_only' & scorer == 'supervised'"
    )
    df_mean = df_mean.query("estimator != 'deep_no_da_source_only'")
    df_mean = df_mean.query("estimator != 'deep_no_da_target_only'")

    if scorer_selection == "supervised":
        df_tot = df_mean.query("scorer == 'supervised'")

    elif scorer_selection == "unsupervised":
        df_mean = df_mean.query(
            "scorer != 'supervised' & scorer != 'best_scorer'"
        )

        best_scorers = (
            df_mean.groupby(["estimator", "scorer"])[
                "target_accuracy-test-mean"
            ]
            .mean()
            .reset_index()
        )

        idx_best_scorer = best_scorers.groupby(["estimator"])[
            "target_accuracy-test-mean"
        ].idxmax()
        best_scorers = best_scorers.loc[idx_best_scorer]

        df_tot = df_mean.merge(
            best_scorers[
                [
                    "estimator",
                    "scorer",
                ]
            ],
            on=[
                "estimator",
            ],
            suffixes=("", "_best"),
        )
        df_tot = df_tot.query("scorer == scorer_best")

    df_tot = pd.concat(
        [df_tot, df_source_mean, df_target_mean], axis=0
    ).reset_index()
    df_tab = df_tot.pivot(
        index="dataset",
        columns=["estimator"],
        values="target_accuracy-test-mean",
    )

    columns = DEEP_ESTIMATOR_DICT.keys()
    df_tab = df_tab.reindex(
        columns=columns,
        level=0,
    )

    df_tab = df_tab.T.merge(df_rank, on="estimator")
    # %%
    if scorer_selection == "unsupervised":
        df_best_scorer = df_tot[["estimator", "scorer_best"]].drop_duplicates()
        df_tab = df_tab.merge(
            df_best_scorer, on="estimator"
        )
        df_tab = df_tab[df_tot["scorer"] ==
                        df_tot["scorer_best"]].reset_index()

    df_tab = df_tab.set_index(["estimator"])
    df_tab = df_tab.round(2)
    # df_tab = df_tab[df_tab.columns[1:]]

    # add the colorcell
    for i, col in enumerate(
        df_tab.columns[:-2 if scorer_selection == "unsupervised" else -1]
    ):
        max_value = df_tab.loc[df_tab[col].index[1], col]
        mean_value = df_tab.loc[df_tab[col].index[0], col]
        min_value = df_tab[col].min()
        for idx in df_tab.index[2:]:
            # get the value
            if df_tab.loc[idx, col] == "nan" or np.isnan(df_tab.loc[idx, col]):
                continue
            value = df_tab.loc[idx, col]
            # get the color
            color = shade_of_color_pvalue(
                value,
                min_value=min_value,
                mean_value=mean_value,
                max_value=max_value,
            )
            df_tab.loc[idx, col] = color
        df_tab.loc[df_tab.index[1], col] = (
            "\\cellcolor{{green_color!{}}}{{{}}}".format(
                60,
                df_tab.loc[df_tab.index[1], col],
            )
        )

    if scorer_selection == "supervised":
        columns = [dataset for dataset in DEEP_DATASET_DICT.keys()]
        columns += ["rank"]
        df_tab = df_tab.reindex(
            columns=columns,
        )
    else:
        columns = [dataset for dataset in DEEP_DATASET_DICT.keys()]
        columns += ["scorer"]
        columns += ["rank"]
        df_tab = df_tab.rename(
            columns={"scorer_best": "scorer"},
        )
        df_tab = df_tab.reindex(
            columns=columns,
        )

    # rename columns
    df_tab = df_tab.rename(
        columns=DEEP_DATASET_DICT,
        index=DEEP_ESTIMATOR_DICT,
    )
    # apply mcrot on columns names
    df_tab.columns = pd.MultiIndex.from_tuples(
        [(f"\\mcrot{{1}}{{l}}{{45}}{{{col}}}", "") for col in df_tab.columns],
        names=["", ""],
    )

    df_tab = df_tab.fillna("\\color{gray!90}NA")

    # convert to latex
    column_format = "|l||" + len(df_tab.columns)*"r" + "|"

    lat_tab = df_tab.to_latex(
        escape=False,
        multicolumn_format="c",
        multirow=True,
        # column_format="|l||rr||r||rr|",
        column_format=column_format,
    )
    lat_tab = lat_tab.replace("\type & estimator &  &  &  &  \\", "")
    lat_tab = lat_tab.replace("toprule", "hline")
    lat_tab = lat_tab.replace("midrule", "hline")
    lat_tab = lat_tab.replace(r"\multirow[t]", r"\multirow")
    lat_tab = lat_tab.replace("bottomrule", "hline")
    lat_tab = lat_tab.replace("circular_validation", "CircV")
    lat_tab = lat_tab.replace("prediction_entropy", "PE")
    lat_tab = lat_tab.replace("importance_weighted", "IW")
    lat_tab = lat_tab.replace("soft_neighborhood_density", "SND")
    lat_tab = lat_tab.replace("deep_embedded_validation", "DEV")
    lat_tab = lat_tab.replace("mix_val_inter", "MixValInter")
    lat_tab = lat_tab.replace("mix_val_both", "MixValBoth")
    lat_tab = lat_tab.replace("mix_val_intra", "MixValIntra")

    # save to txt file
    with open(f"table_results_all_dataset_{scorer_selection}.txt", "w") as f:
        f.write(lat_tab)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate main table for all datasets",
    )

    parser.add_argument(
        "--csv-folder",
        type=str,
        help="Path to the csv folder containing results for real data",
        required=True,
    )

    parser.add_argument(
        "--scorer-selection",
        type=str,
        choices=["unsupervised", "supervised"],
        required=True
    )

    args = parser.parse_args()
    df = generate_table(
        args.csv_folder, args.scorer_selection)
