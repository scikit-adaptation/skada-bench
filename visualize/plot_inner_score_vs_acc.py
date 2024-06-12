# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import argparse


def generate_scatter(csv_file):
    df = pd.read_csv(csv_file)

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

    df["target_accuracy-test-identity"] = df["target_accuracy-test-identity"].apply(lambda x: json.loads(x))
    df["cv_score"] = df["cv_score"].apply(lambda x: json.loads(x))

    df_filtered = df.query("estimator != 'Train Tgt'")
    df_filtered = df_filtered.query("estimator != 'Train Src'")
    df_grouped = df_filtered.groupby(["dataset", "scorer", "estimator", "shift"])
    cv_score = []
    acc = []
    scorer = []
    estimator = []
    dataset = []
    shift = []
    type = []
    for idx, df_ in df_grouped:
        acc_list = df_["target_accuracy-test-identity"].values[0]
        score_list = df_["cv_score"].values[0]
        for acc_, score_ in zip(acc_list, score_list):
            acc.append(acc_)
            cv_score.append(score_)
            scorer.append(df_["scorer"].values[0])
            estimator.append(df_["estimator"].values[0])
            dataset.append(df_["dataset"].values[0])
            shift.append(df_["shift"].values[0])
            type.append(df_["type"].values[0])

    df_final = pd.DataFrame(
        {
            "dataset": dataset,
            "scorer": scorer,
            "estimator": estimator,
            "acc": acc,
            "cv_score": cv_score,
            "shift": shift,
            "type": type,
        }
    )

    fig, axes = plt.subplots(2, 3, figsize=(8.7, 5.5), sharex=True,)
    name_scorer = ["Supervised", "IW", "SND", "DEV", "PE", "CircV", ]
    scorers = [
        "supervised",
        "importance_weighted",
        "soft_neighborhood_density",
        "deep_embedded_validation",
        "prediction_entropy",
        "circular_validation",
    ]

    # iterate over axes
    for i, ax in enumerate(axes.flat):
        df_scorer = df_final.query(f'scorer == "{scorers[i]}"')
        df_scorer.rename(
            columns={"type": "Method type", "dataset": "Dataset"},
            inplace=True
        )
        if i == 0:
            sns.scatterplot(
                data=df_scorer,
                x="acc",
                y="cv_score",
                hue="Method type",
                ax=ax,
                alpha=0.3,
                palette="colorblind",
                edgecolor="gray",
                linewidth=0,
                marker=".",
            )
            ax.legend(fontsize=7)
        else:
            sns.scatterplot(
                data=df_scorer,
                x="acc",
                y="cv_score",
                hue="Method type",
                # style="Dataset",
                ax=ax,
                legend=False,
                alpha=0.3,
                edgecolor="gray",
                palette="colorblind",
                linewidth=0,
                marker=".",
            )
        ax.set_xlim(0, 1)
        if scorers[i] == "deep_embedded_validation":
            ax.set_ylim(-2, 0.1)
        corr = df_scorer[["acc", "cv_score"]].corr().values[0, 1]
        ax.set_title(rf"{name_scorer[i]} - $\rho$ = {corr:.2f}")
        ax.set_xlabel("")
        ax.set_ylabel("")
    axes[0, 0].set_ylabel("Cross-validation score")
    axes[1, 0].set_ylabel("Cross-validation score")
    axes[1, 0].set_xlabel("Target accuracy")
    axes[1, 1].set_xlabel("Target accuracy")
    axes[1, 2].set_xlabel("Target accuracy")
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.2)
    fig.savefig("scatter_scorers_inner.pdf", bbox_inches="tight")

    fig, axes = plt.subplots(4, 6, figsize=(12, 7.5), sharex=True,)
    name_scorer = ["Supervised", "IW", "SND", "DEV", "PE", "CircV"]
    scorers = [
        "supervised",
        "importance_weighted",
        "soft_neighborhood_density",
        "deep_embedded_validation",
        "prediction_entropy",
        "circular_validation",
    ]
    types = [
        "Reweighting",
        "Mapping",
        "Subspace",
        "Other",
    ]

    for i, scorer in enumerate(scorers):
        for j, type in enumerate(types):
            df_scorer_type = df_final.query(
                f'scorer == "{scorer}" & type == "{type}"'
            )
            if i < 4 or j != 2:
                sns.scatterplot(
                    data=df_scorer_type,
                    x="acc",
                    y="cv_score",
                    ax=axes[j, i],
                    alpha=0.3,
                    palette="colorblind",
                    edgecolor="gray",
                    linewidth=0,
                    marker=".",
                )
            else:
                sns.scatterplot(
                    data=df_scorer_type,
                    x="acc",
                    y="cv_score",
                    ax=axes[j, i],
                    legend=False,
                    alpha=0.3,
                    edgecolor="gray",
                    palette="colorblind",
                    linewidth=0,
                    marker=".",
                )
                axes[j, i].legend(fontsize=6)

            axes[j, i].set_xlabel("")
            axes[j, i].set_ylabel("")

            if j == 0:
                axes[j, i].set_title(f"{name_scorer[i]}")
            if i == 0:
                axes[j, i].set_ylabel(f"{type}\n Cross-validation score")
            if scorers[i] == "deep_embedded_validation":
                axes[j, i].set_ylim(-2, 0.1)
            axes[j, i].set_xlabel("Target accuracy")

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.4)
    fig.savefig("scatter_scorer_inner_big_tables.pdf", bbox_inches="tight")


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

    df = generate_scatter(args.csv_file)
