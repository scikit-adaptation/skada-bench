# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# %%
df = pd.read_csv("simulated_readable_csv.csv")

df_target = df.query('estimator == "Train Tgt" & scorer == "supervised"')
df_source = df.query(
    'estimator == "Train Src" & scorer != "supervised" '
    '& scorer != "best_scorer"'
)
idx_source_best_scorer = df_source.groupby(["shift"])["accuracy-mean"].idxmax()
df_source = df_source.loc[idx_source_best_scorer]

df_source_target = pd.concat(
    [
        df_source,
        df_target,
    ],
    axis=0,
)
df_source_target = df_source_target.sort_values(["type"])
df_source_target_tab = df_source_target.pivot(
    index="shift", columns=["type", "estimator"], values="accuracy-mean"
)

df_other_ = df.query('estimator != "Train Tgt" and estimator != "Train Src"')
df_other = df_other_.query(
    'scorer != "best_scorer" and scorer != "supervised"'
)
idx_best_scorer = df_other.groupby(
    ["estimator", "shift"]
)["accuracy-mean"].idxmax()
df_best_scorer = df_other.loc[idx_best_scorer]

df_supervised = df_other_.query('scorer == "supervised"')
df_tot = df_other.merge(
    df_supervised[["estimator", "shift", "accuracy-mean", "accuracy-std"]],
    on=["estimator", "shift"],
    suffixes=("", "_supervised"),
)
# %%
fig, axes = plt.subplots(1, 4, figsize=(12, 2.7))
name_shift = ["Cov. shift", "Tar. shift", "Cond. shift", "Sub. shift"]
shifts = [
    "covariate_shift",
    "target_shift",
    "concept_drift",
    "subspace",
]

for i, shift in enumerate(shifts):
    df_shift = df_tot.query(f'shift == "{shift}"')
    if i < 3:
        sns.scatterplot(
            data=df_shift,
            x="accuracy-mean",
            y="accuracy-mean_supervised",
            hue="scorer",
            style="type",
            ax=axes[i],
            legend=False,
            alpha=0.7,
            s=100,
            edgecolor="gray",
        )
    else:
        sns.scatterplot(
            data=df_shift,
            x="accuracy-mean",
            y="accuracy-mean_supervised",
            hue="scorer",
            style="type",
            ax=axes[i],
            alpha=0.7,
            s=100,
            # change line color
            edgecolor="gray",
        )
        # change legend position,
        axes[i].legend(loc="upper left", bbox_to_anchor=(1.05, 1.05))
    # get lims of the plot
    lims = [
        np.min([axes[i].get_xlim(), axes[i].get_ylim()]),
        np.max([axes[i].get_xlim(), axes[i].get_ylim()]),
    ]
    # plot the diagonal
    axes[i].plot(lims, lims, "k--", alpha=0.75, zorder=0)
    axes[i].set_title(f"Shift {name_shift[i]}")
    axes[i].set_xlabel("DA Scorer")
    axes[i].set_ylabel("")
axes[0].set_ylabel("Supervised Scorer")
fig.savefig("scorers.pdf", bbox_inches="tight")
# %%
