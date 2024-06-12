# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse


def generate_boxplot(csv_file):
    df = pd.read_csv(csv_file)
    df_rm_supervised = df.query("scorer != 'supervised' & scorer != 'best_scorer'")
    df_rm_supervised = df_rm_supervised.query("estimator != 'NO_DA_SOURCE_ONLY_BASE_ESTIM'")

    df_best_scorer = (
        df_rm_supervised.groupby(["estimator", "scorer"])[
            "target_accuracy-test-mean"
        ]
        .mean()
        .reset_index()
    )

    idx_best_scorer = df_best_scorer.groupby(["estimator"])[
        "target_accuracy-test-mean"
    ].idxmax()
    df_best_scorer = df_best_scorer.loc[idx_best_scorer]

    df_target = df.query('estimator == "Train Tgt" & scorer == "supervised"')
    df_source = df.query('estimator == "Train Src" & scorer == "best_scorer"')
    df = df.merge(
        df_target[["shift", "target_accuracy-test-mean", "target_accuracy-test-std"]],
        on="shift",
        suffixes=("", "_target"),
    )
    df = df.merge(
        df_source[["shift", "target_accuracy-test-mean", "target_accuracy-test-std"]],
        on="shift",
        suffixes=("", "_source"),
    )
    df["accn"] = (
        df["target_accuracy-test-mean"] - df["target_accuracy-test-mean_source"]
    ) / (df["target_accuracy-test-mean_target"] - df["target_accuracy-test-mean_source"])

    df["stdn"] = df["target_accuracy-test-std"] / np.abs(
        (df["target_accuracy-test-mean_target"] - df["target_accuracy-test-mean_source"])
    )
    # remove rows where the source is better than the target
    df = df[
        df["target_accuracy-test-mean_source"] < df["target_accuracy-test-mean_target"]
    ].reset_index()

    # filtering
    df = df.query("estimator != 'NO_DA_SOURCE_ONLY_BASE_ESTIM'")

    df_supervised = df.query('scorer == "supervised"')
    df_tot = df.merge(
        df_supervised[
            [
                "estimator",
                "dataset",
                "target_accuracy-test-mean",
                "shift"
            ]
        ],
        on=["estimator", "dataset", "shift"],
        suffixes=("", "_supervised"),
    )

    df_tot["Delta"] = (
        df_tot["target_accuracy-test-mean"] - df_tot["target_accuracy-test-mean_supervised"]
    )

    df_tot = df_tot.merge(
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

    df_tot = df_tot[df_tot["scorer"] == df_tot["scorer_best"]].reset_index()

    fig = plt.figure(figsize=(12, 2.5))

    order = [
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
    ]
    axis = sns.boxplot(
        data=df_tot,
        x="estimator",
        y="Delta",
        hue="type",
        showfliers=False,
        width=0.5,
        palette="colorblind",
        dodge=False,
        order=order,
    )
    axis.get_legend().remove()
    for patch in axis.patches:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, 0.6))
    sns.stripplot(
        data=df_tot,
        x="estimator",
        y="Delta",
        hue="type",
        legend=False,
        dodge=False,
        palette="colorblind",
        order=order,
        alpha=0.4,
        size=4,
        # color="silver",
        edgecolor="dimgray",
        linewidth=0.5,
    )
    plt.grid(axis="y", linestyle="--")
    plt.grid(axis="x", linestyle="--")
    plt.yticks(np.arange(-0.1, 0.11, 0.05))
    plt.ylim(-0.1, 0.1)
    plt.axhline(0, color="black", linestyle="--")
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.xlabel("", fontsize=10)
    plt.ylabel(r"$\Delta$ACC w.r.t. to supervised", fontsize=10)
    fig.savefig("boxplot.pdf", bbox_inches="tight",)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute scatter plot"
    )

    parser.add_argument(
        "--csv-file",
        type=str,
        help="Path to the csv file containing results for real data",
        default='./readable_csv/results_all_datasets_experiments.csv'
    )

    args = parser.parse_args()

    df = generate_boxplot(args.csv_file)
