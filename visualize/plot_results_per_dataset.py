# %%
import numpy as np
import pandas as pd
import json
import argparse


def shade_of_color(
    df_value,
    final_value,
    min_value=0,
    max_value=1,
):
    # Intensity range for the green and red colors
    intensity_range = (30, 70)

    if df_value == "nan" or np.isnan(df_value):
        # Return the nan value
        return final_value
    elif df_value > min_value:
        color_min = min_value
        color_max = max_value
        if color_max - color_min == 0:
            # To avoid division by zero
            intensity = intensity_range[1]
        else:
            intensity = int(
                intensity_range[0]
                + (intensity_range[1] - intensity_range[0])
                * (df_value - color_min)
                / (color_max - color_min)
            )
        return "\\cellcolor{green_color!%d}{%s}" % (intensity, final_value)
    else:
        return final_value


def generate_table_results(
    dataset,
    csv_file,
    csv_file_simulated,
):
    df_base = pd.read_csv(csv_file)
    df_base = df_base.query("scorer != 'supervised' & scorer != 'best_scorer'")
    df_base = df_base.query("estimator != 'NO_DA_SOURCE_ONLY_BASE_ESTIM'")
    df_best_scorer = (
        df_base.groupby(["estimator", "scorer"])[
            "target_accuracy-test-mean"
        ]
        .mean()
        .reset_index()
    )

    idx_best_scorer = df_best_scorer.groupby(["estimator"])[
        "target_accuracy-test-mean"
    ].idxmax()
    df_best_scorer = df_best_scorer.loc[idx_best_scorer]

    if dataset == "simulated":
        df = pd.read_csv(csv_file_simulated)
        print(df)
    else:
        df = pd.read_csv(csv_file)
        df = df.query("dataset == @dataset")

    df = df.query("estimator != 'NO_DA_SOURCE_ONLY_BASE_ESTIM'")

    df["target_accuracy-test-identity"] = df["target_accuracy-test-identity"].apply(
        lambda x: json.loads(x)
    )

    df["nb_splits"] = df["target_accuracy-test-identity"].apply(lambda x: len(x))

    df_target = df.query('estimator == "Train Tgt" & scorer == "supervised"')
    df_source = df.query(
        'estimator == "Train Src" & scorer !=' '"supervised" & scorer != "best_scorer"'
    )
    idx_source_best_scorer = df_source.groupby(["shift"])[
        "target_accuracy-test-mean"
    ].idxmax()
    df_source = df_source.loc[idx_source_best_scorer]

    df = df.merge(
        df_target[["shift", "target_accuracy-test-mean", "target_accuracy-test-std"]],
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
    # remove rows where the source is better than the target
    df = df[
        df["target_accuracy-test-mean_source"] < df["target_accuracy-test-mean_target"]
    ].reset_index()
    # check if nb_splits is 5 and 25 for the simulated dataset
    df = df.query("nb_splits == 5 | nb_splits == 25")

    # remove duplicates
    df = df.drop_duplicates(subset=["dataset", "scorer", "estimator", "shift"])

    df["rank"] = df.groupby(["dataset", "scorer", "shift"])[
        "target_accuracy-test-mean"
    ].rank(ascending=False)

    df_dataset = df.merge(
        df_best_scorer[
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
    df_dataset = df_dataset[
        df_dataset["scorer"] == df_dataset["scorer_best"]
    ].reset_index()
    df_rank = df_dataset.groupby(["estimator"])["rank"].mean().reset_index()
    # add mean \pm std column
    df_dataset["acc_std"] = (
        df_dataset["target_accuracy-test-mean"].round(2).astype(str)
        + " $\pm$ "
        + df_dataset["target_accuracy-test-std"].round(2).astype(str)
    )

    # create the table
    df_tab = df_dataset.pivot(
        index="shift", columns=["type", "estimator"], values="target_accuracy-test-mean"
    )
    df_tab_acc_std = df_dataset.pivot(
        index="shift", columns=["type", "estimator"], values="acc_std"
    )
    df_tab = df_tab.reindex(
        columns=["NO DA", "Reweighting", "Mapping", "Subspace", "Other"], level=0
    )
    df_tab_acc_std = df_tab_acc_std.reindex(
        columns=["NO DA", "Reweighting", "Mapping", "Subspace", "Other"], level=0
    )

    df_tab = df_tab.reindex(
        columns=[
            "Train Src",
            "Train Tgt",
            "Dens. RW",
            "Disc. RW",
            "Gauss. RW",
            "KLIEP",
            "KMM",
            "NN RW",
            "MMDTarS",
            "CORAL",
            "MapOT",
            "EntOT",
            "ClassRegOT",
            "LinOT",
            "MMD-LS",
            "JPCA",
            "SA",
            "TCA",
            "TSL",
            "JDOT",
            "OTLabelProp",
            "DASVM",
        ],
        level=1,
    )
    df_tab_acc_std = df_tab_acc_std.reindex(
        columns=[
            "Train Src",
            "Train Tgt",
            "Dens. RW",
            "Disc. RW",
            "Gauss. RW",
            "KLIEP",
            "KMM",
            "NN RW",
            "MMDTarS",
            "CORAL",
            "MapOT",
            "EntOT",
            "ClassRegOT",
            "LinOT",
            "MMD-LS",
            "JPCA",
            "SA",
            "TCA",
            "TSL",
            "JDOT",
            "OTLabelProp",
            "DASVM",
        ],
        level=1,
    )
    df_tab = df_tab.T.rename(
        index={
            "NO DA": r"\rotatebox[origin=c]{90}{}",
            "Reweighting": r"\rotatebox[origin=c]{90}{Reweighting}",
            "Mapping": r"\rotatebox[origin=c]{90}{Mapping}",
            "Subspace": r"\rotatebox[origin=c]{90}{Subspace}",
            "Other": r"\rotatebox[origin=c]{90}{Other}",
        }
    )

    df_tab_acc_std = df_tab_acc_std.T.rename(
        index={
            "NO DA": r"\rotatebox[origin=c]{90}{}",
            "Reweighting": r"\rotatebox[origin=c]{90}{Reweighting}",
            "Mapping": r"\rotatebox[origin=c]{90}{Mapping}",
            "Subspace": r"\rotatebox[origin=c]{90}{Subspace}",
            "Other": r"\rotatebox[origin=c]{90}{Other}",
        }
    )

    if df_dataset["dataset"].values[0] == "Simulated":
        df_tab = df_tab.reindex(
            columns=[
                "covariate_shift",
                "target_shift",
                "concept_drift",
                "subspace",
            ],
        )
        df_tab = df_tab.rename(
            columns={
                "covariate_shift": "\\underline{Cov. shift}",
                "target_shift": "\\underline{Tar. shift}",
                "concept_drift": "\\underline{Cond. shift}",
                "subspace": "\\underline{Sub. shift}",
            }
        )
        df_tab_acc_std = df_tab_acc_std.reindex(
            columns=[
                "covariate_shift",
                "target_shift",
                "concept_drift",
                "subspace",
            ],
        )
        df_tab_acc_std = df_tab_acc_std.rename(
            columns={
                "covariate_shift": "\\underline{Cov. shift}",
                "target_shift": "\\underline{Tar. shift}",
                "concept_drift": "\\underline{Cond. shift}",
                "subspace": "\\underline{Sub. shift}",
            }
        )
    df_mean_dataset = (
        df_dataset.groupby(["estimator", "scorer"])[
            "target_accuracy-test-mean", "target_accuracy-test-std"
        ]
        .mean()
        .reset_index()
    )
    df_mean_dataset["Mean"] = df_mean_dataset["target_accuracy-test-mean"]
    # rename accuracy to mean
    df_tab = df_tab.reset_index().merge(
        df_mean_dataset[["estimator", "Mean"]], on="estimator"
    )
    df_mean_dataset["Mean"] = (
        df_mean_dataset["target_accuracy-test-mean"].round(2).astype(str)
        + " $\pm$ "
        + df_mean_dataset["target_accuracy-test-std"].round(2).astype(str)
    )
    df_tab_acc_std = df_tab_acc_std.reset_index().merge(
        df_mean_dataset[["estimator", "Mean"]],
        on="estimator",
    )
    df_tab = df_tab.merge(df_rank, on="estimator")

    df_tab = df_tab.set_index(["type", "estimator"])
    df_tab = df_tab.round(2)

    df_tab_acc_std = df_tab_acc_std.set_index(["type", "estimator"])

    for i, col in enumerate(df_tab.columns[:-1]):
        max_value = df_tab.loc[df_tab[col].index[1], col]
        min_value = df_tab.loc[df_tab[col].index[0], col]
        for idx in df_tab.index[2:]:
            # get the value
            if df_tab.loc[idx, col] == "nan" or np.isnan(df_tab.loc[idx, col]):
                continue
            value = df_tab.loc[idx, col]
            final_value = df_tab_acc_std.loc[idx, col]
            # get the color
            color = shade_of_color(
                value,
                final_value,
                min_value=min_value,
                max_value=max_value,
            )
            df_tab.loc[idx, col] = color
        df_tab.loc[df_tab.index[1], col] = "\\cellcolor{green_color!%d}{%s}" % (
            70,
            df_tab_acc_std.loc[df_tab.index[1], col],
        )
        df_tab.loc[df_tab.index[0], col] = df_tab_acc_std.loc[df_tab.index[0], col]

        # apply mcrot on the columns
        df_tab = df_tab.rename(
            columns={
                col: "\mcrot{1}{l}{45}{" + str(col) + "}",
            }
        )
    df_tab = df_tab.rename(
        columns={
            "rank": "\mcrot{1}{l}{45}{Rank}",
        }
    )

    df_tab = df_tab.fillna("\\color{gray!90}NA")

    # convert to latex
    lat_tab = df_tab.to_latex(
        escape=False,
        multicolumn_format="c",
        multirow=True,
        # remove [t] in multirow
        column_format="|l|l||" + "r|" * (len(df_tab.columns) - 1) + "|r|",
        # round value
        # float_format="%.2f",
        # put the name of multirow in vertical
    )
    lat_tab = lat_tab.replace("toprule", "hline")
    lat_tab = lat_tab.replace("midrule", "hline")
    lat_tab = lat_tab.replace(
        "cline{1-" + f"{2 + len(df_tab.columns)}" + "}", "hline\hline"
    )
    lat_tab = lat_tab.replace("\multirow[t]", "\multirow")
    lat_tab = lat_tab.replace("bottomrule", "hline")

    # save in txt file
    with open(f"table_results_{df['dataset'].values[0]}.txt", "w") as f:
        f.write(lat_tab)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate table results per dataset")

    parser.add_argument("--dataset", type=str, help="dataset name", default="simulated")

    parser.add_argument(
        "--csv-file",
        type=str,
        help="Path to the csv file containing results for real data",
        default="./readable_csv/results_all_datasets_experiments.csv",
    )

    parser.add_argument(
        "--csv-file-simulated",
        type=str,
        help="Path to the csv file containing results for simulated data",
        default="./readable_csv/simulated_31_05_readable.csv",
    )

    args = parser.parse_args()

    df = generate_table_results(
        args.dataset, args.csv_file, args.csv_file_simulated
    )
