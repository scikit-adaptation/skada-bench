# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import numpy as np

# %%
files = glob.glob("./readable_csv/*readable.csv")
df = pd.concat([pd.read_csv(file) for file in files])

# df_simulated = pd.read_csv("./readable_csv/simulated_readable_csv.csv")
# df_simulated["dataset"] = df_simulated["shift"]

# df = pd.concat([df, df_simulated])
# %%
df_target = df.query('estimator == "Train Tgt" & scorer == "supervised"')
df_source = df.query('estimator == "Train Src" & scorer == "best_scorer"')
df = df.merge(
    df_target[["shift", "accuracy-mean", "accuracy-std"]],
    on="shift",
    suffixes=("", "_target"),
)
df = df.merge(
    df_source[["shift", "accuracy-mean", "accuracy-std"]],
    on="shift",
    suffixes=("", "_source"),
)
df["accn"] = (df["accuracy-mean"] - df["accuracy-mean_source"]) / (
    df["accuracy-mean_target"] - df["accuracy-mean_source"]
)

df["stdn"] = df["accuracy-std"] / np.abs(
    (df["accuracy-mean_target"] - df["accuracy-mean_source"])
)
# remove rows where the source is better than the target
df = df[df["accuracy-mean_source"] < df["accuracy-mean_target"]].reset_index()

# filtering
df = df.query("estimator != 'NO_DA_SOURCE_ONLY_BASE_ESTIM'")

# %%

df_mean = (
    df.groupby(["dataset", "type", "scorer", "estimator"])["accuracy-mean",]
    .mean()
    .reset_index()
)
# %%
# remove estimators
df_mean = df_mean.query("estimator != 'Train Tgt'")
df_mean = df_mean.query("estimator != 'Train Src'")

df_supervised = df_mean.query('scorer == "supervised"')
df_tot = df_mean.merge(
    df_supervised[["estimator", "dataset", "accuracy-mean",]],
    on=["estimator", "dataset"],
    suffixes=("", "_supervised"),
)
df_tot = df_tot.query("scorer != 'supervised' & scorer != 'best_scorer'")

# %%
fig, axes = plt.subplots(1, 5, figsize=(11, 1.9), sharey=True)
name_scorer = ["IW", "SND", "DEV", "PE", "CircV"]
scorers = [
    "importance_weighted",
    "soft_neighborhood_density",
    "deep_embedded_validation",
    "prediction_entropy",
    "circular_validation",
]

for i, scorer in enumerate(scorers):
    df_scorer = df_tot.query(f'scorer == "{scorer}"')
    df_scorer.rename(columns={"type": "Method type", "dataset": "Dataset"}, inplace=True)
    if i < 4:
        sns.scatterplot(
            data=df_scorer,
            x="accuracy-mean_supervised",
            y="accuracy-mean",
            hue="Method type",
            style="Dataset",
            ax=axes[i],
            legend=False,
            alpha=0.7,
            s=70,
            edgecolor="gray",
        )
    else:
        sns.scatterplot(
            data=df_scorer,
            x="accuracy-mean_supervised",
            y="accuracy-mean",
            hue="Method type",
            style="Dataset",
            ax=axes[i],
            alpha=0.7,
            s=70,
            # change line color
            edgecolor="gray",
        )
        # change legend position,
        axes[i].legend(loc="upper left", bbox_to_anchor=(1.1, 1.2), fontsize=8)
    # get lims of the plot
    lims = [
        np.min([axes[i].get_xlim(), axes[i].get_ylim()]),
        np.max([axes[i].get_xlim(), axes[i].get_ylim()]),
    ]
    # plot the diagonal
    axes[i].plot(lims, lims, "k--", alpha=0.75, zorder=0)
    axes[i].set_xlim(0, 1)
    axes[i].set_ylim(0, 1)
    axes[i].set_aspect("equal")
    axes[i].set_title(f"{name_scorer[i]}")
    axes[i].set_xlabel("Supervised scorer")
    axes[i].set_ylabel("")
axes[0].set_ylabel("Unsupervised scorer")
fig.savefig("scatter_scorers.pdf", bbox_inches="tight")
# %%
fig, axes = plt.subplots(4, 5, figsize=(12, 9), sharex=True, sharey=True)
name_scorer = ["IW", "SND", "DEV", "PE", "CircV"]
scorers = [
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
        df_scorer_type = df_tot.query(f'scorer == "{scorer}" & type == "{type}"')
        if i < 4 or j != 2:
            sns.scatterplot(
                data=df_scorer_type,
                x="accuracy-mean_supervised",
                y="accuracy-mean",
                # hue="type",
                hue="dataset",
                ax=axes[j, i],
                legend=False,
                alpha=0.7,
                s=80,
                edgecolor="gray",
            )
        else:
            sns.scatterplot(
                data=df_scorer_type,
                x="accuracy-mean_supervised",
                y="accuracy-mean",
                # hue="type",
                hue="dataset",
                ax=axes[j, i],
                alpha=0.7,
                s=80,
                # change line color
                edgecolor="gray",
                legend=True,
            )
            axes[j, i].legend(fontsize=6)
        if j == 0:
            axes[j, i].set_title(f"{name_scorer[i]}")
        if i == 0:
            axes[j, i].set_ylabel(f"{type}\n Unsupervised Scorer")
        # plot the diagonal
        axes[j, i].plot([0, 1], [0, 1], "k--", alpha=0.75, zorder=0)
        axes[j, i].set_xlim(0, 1)
        axes[j, i].set_ylim(0, 1)
        axes[j, i].set_aspect("equal")
        axes[j, i].set_xlabel("Supervised Scorer")
fig.savefig("scatter_scorer_4x4.pdf", bbox_inches="tight")
# %%
