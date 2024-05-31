# %%
import numpy as np
import glob
import pandas as pd
import hiplot as hip


def shade_of_color(
    df_value,
    min_value=0,
    max_value=1,
    std=0,
    color_threshold=0,
    is_delta_table=False,
):
    # If is_delta_table, we want green > 0
    # red < 0 and transparent = 0

    # Intensity range for the green and red colors
    intensity_range = (10, 75)

    if df_value == "nan" or np.isnan(df_value):
        # Return the nan value
        return df_value
    elif df_value > (color_threshold + std):
        color_min = color_threshold
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
        return "\\cellcolor{good_color!%d}{%s}" % (intensity, df_value)
    elif df_value < (color_threshold - std):
        # No color if value = 0 for the delta table
        if is_delta_table and df_value == 0:
            return df_value

        red_min = min_value
        red_max = color_threshold
        if red_min - red_max == 0:
            # To avoid division by zero
            intensity = intensity_range[1]
        else:
            intensity = int(
                intensity_range[0]
                + (intensity_range[1] - intensity_range[0])
                * (df_value - red_max)
                / (red_min - red_max)
            )
        return "\\cellcolor{bad_color!%d}{%s}" % (intensity, df_value)
    else:
        return df_value


# %%
files = glob.glob("./readable_csv/*readable.csv")
df = pd.concat([pd.read_csv(file) for file in files])

df_simulated = pd.read_csv("./readable_csv/simulated_readable_csv.csv")
df_simulated["dataset"] = df_simulated["shift"]

df = pd.concat([df, df_simulated])

df = df.query("estimator != 'NO_DA_SOURCE_ONLY_BASE_ESTIM'")

# %%
df_target = df.query('estimator == "Train Tgt" & scorer == "supervised"')
df_source = df.query(
    'estimator == "Train Src" & scorer != "supervised" & scorer != "best_scorer"'
)
idx_source_best_scorer = df_source.groupby(["shift"])["target_accuracy-test-mean"].idxmax()
df_source = df_source.loc[idx_source_best_scorer]

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

# remove rows where the source is better than the target
df = df[df["target_accuracy-test-mean_source"] < df["target_accuracy-test-mean_target"]].reset_index()
df = (
    df.groupby(["dataset", "type", "scorer", "shift", "estimator"]).mean().reset_index()
)
# %%
df["rank"] = df.groupby(["dataset", "scorer", "shift"])["target_accuracy-test-mean"].rank(
    ascending=False
)

# %%
import scipy.stats as stats
df_filtered = df.query("dataset != 'covariate_shift' & dataset != 'target_shift' & dataset != 'concept_drift' & dataset != 'subspace'")
df_filtered = df_filtered.query("estimator != 'Train Tgt'")
df_filtered = df_filtered.query("estimator != 'Train Src'")
df_grouped = df_filtered.groupby(["scorer", "estimator"])
wilco = []
scorer = []
estimator = []
for idx, df_ in df_grouped:
    # test de wilcoxon
    wilco.append(stats.wilcoxon(df_["target_accuracy-test-mean"], df_["target_accuracy-test-mean_source"], alternative="greater"))
    scorer.append(df_["scorer"].values[0])
    estimator.append(df_["estimator"].values[0])

df_wilco = pd.DataFrame(
    {
        "scorer": scorer,
        "estimator": estimator,
        "pvalue": [w[1] for w in wilco],
        "stat": [w[0] for w in wilco],
    }
)

def mapped_pvalue(pvalue):
    if pvalue < 10e-4:
        return "****"
    elif pvalue < 10e-3:
        return "***"
    elif pvalue < 10e-2:
        return "**"
    elif pvalue < 5*10e-2:
        return "*"
    else:
        return "ns"
    
df_wilco["pvalue"] = df_wilco["pvalue"].apply(mapped_pvalue)
# %%
df_mean = (
    df.groupby(["dataset", "type", "scorer", "estimator"])[
        "target_accuracy-test-mean", "target_accuracy-test-std", "rank"
    ]
    .mean()
    .reset_index()
)

df_source_mean = df_mean.query(
    "estimator == 'Train Src' & scorer == 'circular_validation'"
)
df_target_mean = df_mean.query("estimator == 'Train Tgt' & scorer == 'supervised'")
# %%
# remove estimators
# df_mean = df_mean.query("estimator != 'Train Tgt'")
# df_mean = df_mean.query("estimator != 'Train Src'")
df_mean = df_mean.query("scorer != 'supervised' & scorer != 'best_scorer'")

df_mean_dataset = df_mean.query(
    "dataset != 'covariate_shift' & dataset != 'target_shift' & dataset != 'concept_drift' & dataset != 'subspace'"
)
df_mean_dataset = (
    df_mean_dataset.groupby(["estimator", "scorer"])["target_accuracy-test-mean"]
    .mean()
    .reset_index()
)

idx_best_scorer = df_mean_dataset.groupby(["estimator"])["target_accuracy-test-mean"].idxmax()
df_mean_dataset = df_mean_dataset.loc[idx_best_scorer]

df_tot = df_mean.merge(
    df_mean_dataset[
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

df_wilco = df_wilco[["estimator", "scorer", "pvalue"]].merge(
    df_mean_dataset[["estimator", "scorer"]],
    on=["estimator",],
    suffixes=("", "_best"),
)

df_wilco = df_wilco[df_wilco["scorer"] == df_wilco["scorer_best"]].reset_index()
# %%
df_tot_ = df_tot.query(
    "dataset != 'covariate_shift' & dataset != 'target_shift' & dataset != 'concept_drift' & dataset != 'subspace'"
)
df_rank = df_tot_.groupby(["estimator"])["rank"].mean().reset_index()
# %%
df_tot = df_tot.query("estimator != 'Train Tgt'")
df_tot = df_tot.query("estimator != 'Train Src'")
df_tot = pd.concat([df_tot, df_source_mean, df_target_mean], axis=0).reset_index()
# %%
std_source = df_source_mean["target_accuracy-test-std"].values

# %%
# create the table
df_tab = df_tot.pivot(
    index="dataset", columns=["type", "estimator"], values="target_accuracy-test-mean"
)

df_tab = df_tab.reindex(
    columns=["NO DA", "Reweighting", "Mapping", "Subspace", "Other"], level=0
)
df_tab = df_tab.T.rename(
    index={
        "NO DA": r"\rotatebox[origin=c]{90}{NA}",
        "Reweighting": r"\rotatebox[origin=c]{90}{Reweighting}",
        "Mapping": r"\rotatebox[origin=c]{90}{Mapping}",
        "Subspace": r"\rotatebox[origin=c]{90}{Subspace}",
        "Other": r"\rotatebox[origin=c]{90}{Other}",
    }
)

df_tab = df_tab.reset_index().merge(df_rank, on="estimator")
df_tab = df_tab.reset_index().merge(df_mean_dataset[["estimator", "scorer"]], on="estimator")
# df_tab = df_tab.reset_index().merge(df_wilco[["estimator", "pvalue"]], on=["estimator"])
df_tab = df_tab.set_index(["type", "estimator"])
df_tab = df_tab.round(2)
df_tab = df_tab[df_tab.columns[1:]]
# %%
# add the colorcell
for i, col in enumerate(df_tab.columns[:-2]):
    max_value = df_tab.loc[df_tab[col].index[1], col]
    color_threshold = df_tab.loc[df_tab[col].index[0], col]
    min_value = df_tab[col].min()
    std_value = std_source[i]
    for idx in df_tab.index[2:]:
        # get the value
        if df_tab.loc[idx, col] == "nan" or np.isnan(df_tab.loc[idx, col]):
            continue
        value = df_tab.loc[idx, col]
        # get the color
        color = shade_of_color(
            value,
            std=std_value,
            min_value=min_value,
            max_value=max_value,
            color_threshold=color_threshold,
        )
        df_tab.loc[idx, col] = color
    df_tab.loc[df_tab.index[1], col] = "\\cellcolor{good_color!%d}{%s}" % (
        90,
        df_tab.loc[df_tab.index[1], col],
    )


df_tab = df_tab.reindex(
    columns=[
        "covariate_shift",
        "target_shift",
        "concept_drift",
        "subspace",
        "Office31",
        "OfficeHomeResnet",
        "mnist_usps",
        # "mnist_usps_10k",
        "20NewsGroups",
        "AmazonReview",
        "Mushrooms",
        "Phishing",
        "BCI",
        "scorer",
        "rank",
        "pvalue",
    ],
)
df_tab = df_tab.rename(
    columns={
        "covariate_shift": "\mcrot{1}{l}{45}{Cov. shift}",
        "target_shift": "\mcrot{1}{l}{45}{Tar. shift}",
        "concept_drift": "\mcrot{1}{l}{45}{Cond. shift}",
        "subspace": "\mcrot{1}{l}{45}{Sub. shift}",
        "Office31": "\mcrot{1}{l}{45}{Office31}",
        "OfficeHomeResnet": "\mcrot{1}{l}{45}{OfficeHome}",
        # "mnist_usps_10k": "\mcrot{1}{l}{45}{MNIST/USPS(10k)}",
        "mnist_usps": "\mcrot{1}{l}{45}{MNIST/USPS}",
        "20NewsGroups": "\mcrot{1}{l}{45}{20NewsGroups}",
        "AmazonReview": "\mcrot{1}{l}{45}{AmazonReview}",
        "Mushrooms": "\mcrot{1}{l}{45}{Mushrooms}",
        "Phishing": "\mcrot{1}{l}{45}{Phishing}",
        "BCI": "\mcrot{1}{l}{45}{BCI}",
        "scorer": "\mcrot{1}{l}{45}{Selected Scorer}",
        "rank": "\mcrot{1}{l}{45}{Rank}",
        "pvalue": "\mcrot{1}{l}{45}{Wilcoxon}",
    }
)

# convert to latex
lat_tab = df_tab.to_latex(
    escape=False,
    multicolumn_format="c",
    multirow=True,
    # remove [t] in multirow
    column_format="|l|l||rrrr||rrr|rr|rr|r||rrr|",
    # round value
    # float_format="%.2f",
    # put the name of multirow in vertical
)
lat_tab = lat_tab.replace("\type & estimator &  &  &  &  \\", "")
lat_tab = lat_tab.replace("toprule", "hline")
lat_tab = lat_tab.replace("midrule", "hline")
lat_tab = lat_tab.replace("cline{1-17}", "hline\hline")
lat_tab = lat_tab.replace("\multirow[t]", "\multirow")
lat_tab = lat_tab.replace("bottomrule", "hline")
lat_tab = lat_tab.replace("mnist_usps", "MNIST/USPS")
lat_tab = lat_tab.replace("OfficeHomeResnet", "OfficeHome")
lat_tab = lat_tab.replace("circular_validation", "CircV")
lat_tab = lat_tab.replace("prediction_entropy", "PE")
lat_tab = lat_tab.replace("importance_weighted", "IW")

# %%
print(lat_tab)

# %%
# # %%
# df = df.merge(
#     df_target[["shift", "target_accuracy-test-mean", "target_accuracy-test-std"]],
#     on="shift",
#     suffixes=("", "_target"),
# )
# df = df.merge(
#     df_source[["shift", "target_accuracy-test-mean", "target_accuracy-test-std"]],
#     on="shift",
#     suffixes=("", "_source"),
# )
# # %%
# df["accn"] = (df["target_accuracy-test-mean"] - df["target_accuracy-test-mean_source"]) / (
#     df["target_accuracy-test-mean_target"] - df["target_accuracy-test-mean_source"]
# )

# df["stdn"] = df["target_accuracy-test-std"] / np.abs(
#     (df["target_accuracy-test-mean_target"] - df["target_accuracy-test-mean_source"])
# )
# # %%
# # remove rows where the source is better than the target
# df = df[df["target_accuracy-test-mean_source"] < df["target_accuracy-test-mean_target"]].reset_index()

# # %%
# # filtering
# df = df.query("estimator != 'NO_DA_SOURCE_ONLY_BASE_ESTIM'")

# # %%
# # Visualization with Hiplot

# ddf = df[["accn", "dataset", "type", "scorer", "estimator"]]
# ddf = ddf.query("estimator != 'Train Tgt'")
# ddf = ddf.query("estimator != 'Train Src'")
# hip.Experiment.from_dataframe(ddf).display()

# # %%

# df_mean = (
#     df.groupby(["dataset", "type", "scorer", "estimator"])["accn", "stdn"]
#     .mean()
#     .reset_index()
# )
# std_source = df_mean.query("estimator == 'Train Src' & scorer == 'best_scorer'")[
#     "stdn"
# ].values
# # %%
# # remove estimators
# df_mean = df_mean.query("estimator != 'Train Tgt'")
# df_mean = df_mean.query("estimator != 'Train Src'")
# df_mean = df_mean.query("scorer == 'best_scorer'")

# # %%
# # create the table
# df_tab = df_mean.pivot(index="dataset", columns=["type", "estimator"], values="accn")
# df_tab = df_tab.round(2)

# df_tab = df_tab.reindex(
#     columns=["NO DA", "Reweighting", "Mapping", "Subspace", "Other"], level=0
# )

# df_tab = df_tab.T.rename(
#     index={
#         "NO DA": r"\rotatebox[origin=c]{90}{NA}",
#         "Reweighting": r"\rotatebox[origin=c]{90}{Reweighting}",
#         "Mapping": r"\rotatebox[origin=c]{90}{Mapping}",
#         "Subspace": r"\rotatebox[origin=c]{90}{Subspace}",
#         "Other": r"\rotatebox[origin=c]{90}{Other}",
#     }
# )

# # add the column rank

# # add the colorcell
# for i, col in enumerate(df_tab.columns):
#     max_value = 1
#     color_threshold = 0
#     min_value = df_tab[col].min()
#     std_value = std_source[i]
#     for idx in df_tab.index[2:]:
#         # get the value
#         if df_tab.loc[idx, col] == "nan" or np.isnan(df_tab.loc[idx, col]):
#             continue
#         value = df_tab.loc[idx, col]
#         # get the color
#         color = shade_of_color(
#             value,
#             std=std_value,
#             min_value=min_value,
#             max_value=max_value,
#             color_threshold=color_threshold,
#         )
#         df_tab.loc[idx, col] = color

# df_tab = df_tab.reindex(
#     columns=[
#         "Office31",
#         # "OfficeHome",
#         "mnist_usps",
#         "20NewsGroups",
#         "AmazonReview",
#         "Mushrooms",
#         "Phishing",
#         "BCI",
#     ],
# )
# df_tab = df_tab.rename(
#     columns={
#         "Office31": "\mcrot{1}{l}{45}{Office31}",
#         "mnist_usps": "\mcrot{1}{l}{45}{MNIST/USPS}",
#         "20NewsGroups": "\mcrot{1}{l}{45}{20NewsGroups}",
#         "AmazonReview": "\mcrot{1}{l}{45}{AmazonReview}",
#         "Mushrooms": "\mcrot{1}{l}{45}{Mushrooms}",
#         "Phishing": "\mcrot{1}{l}{45}{Phishing}",
#         "BCI": "\mcrot{1}{l}{45}{BCI}",
#     }
# )
# # convert to latex
# lat_tab = df_tab.to_latex(
#     escape=False,
#     multicolumn_format="c",
#     multirow=True,
#     column_format="|l|l" + "|r" * 7 + "|",
# )
# lat_tab = lat_tab.replace("\type & estimator &  &  &  &  \\", "")
# lat_tab = lat_tab.replace("toprule", "hline")
# lat_tab = lat_tab.replace("midrule", "hline")
# lat_tab = lat_tab.replace("cline{1-9}", "hline\hline")
# lat_tab = lat_tab.replace("\multirow[t]", "\multirow")
# lat_tab = lat_tab.replace("bottomrule", "hline")
# lat_tab = lat_tab.replace("mnist_usps", "MNIST/USPS")

# # %%
# print(lat_tab)
