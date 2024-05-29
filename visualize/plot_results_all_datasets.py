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
    intensity_range = (10, 90)

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

# %%
# filtering
df = df.query("estimator != 'NO_DA_SOURCE_ONLY_BASE_ESTIM'")

# %%
# Visualization with Hiplot

ddf = df[["accn", "dataset", "type", "scorer", "estimator"]]
ddf = ddf.query("estimator != 'Train Tgt'")
ddf = ddf.query("estimator != 'Train Src'")
hip.Experiment.from_dataframe(ddf).display()

# %%

df_mean = (
    df.groupby(["dataset", "type", "scorer", "estimator"])["accn", "stdn"]
    .mean()
    .reset_index()
)
std_source = df_mean.query("estimator == 'Train Src' & scorer == 'best_scorer'")[
    "stdn"
].values
# %%
# remove estimators
df_mean = df_mean.query("estimator != 'Train Tgt'")
df_mean = df_mean.query("estimator != 'Train Src'")
df_mean = df_mean.query("scorer == 'best_scorer'")

# %%
# create the table
df_tab = df_mean.pivot(index="dataset", columns=["type", "estimator"], values="accn")
df_tab = df_tab.round(2)

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

# add the column rank

# add the colorcell
for i, col in enumerate(df_tab.columns):
    max_value = 1
    color_threshold = 0
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

df_tab = df_tab.reindex(
    columns=[
        "Office31",
        # "OfficeHome",
        "mnist_usps",
        "20NewsGroups",
        "AmazonReview",
        "Mushrooms",
        "Phishing",
        "BCI",
    ],
)
df_tab = df_tab.rename(
    columns={
        "Office31": "\mcrot{1}{l}{45}{Office31}",
        "mnist_usps": "\mcrot{1}{l}{45}{MNIST/USPS}",
        "20NewsGroups": "\mcrot{1}{l}{45}{20NewsGroups}",
        "AmazonReview": "\mcrot{1}{l}{45}{AmazonReview}",
        "Mushrooms": "\mcrot{1}{l}{45}{Mushrooms}",
        "Phishing": "\mcrot{1}{l}{45}{Phishing}",
        "BCI": "\mcrot{1}{l}{45}{BCI}",
    }
)
# convert to latex
lat_tab = df_tab.to_latex(
    escape=False,
    multicolumn_format="c",
    multirow=True,
    column_format="|l|l" + "|r" * 7 + "|",
)
lat_tab = lat_tab.replace("\type & estimator &  &  &  &  \\", "")
lat_tab = lat_tab.replace("toprule", "hline")
lat_tab = lat_tab.replace("midrule", "hline")
lat_tab = lat_tab.replace("cline{1-9}", "hline\hline")
lat_tab = lat_tab.replace("\multirow[t]", "\multirow")
lat_tab = lat_tab.replace("bottomrule", "hline")
lat_tab = lat_tab.replace("mnist_usps", "MNIST/USPS")

# %%
print(lat_tab)

# %%
df_mean = (
    df.groupby(["dataset", "type", "scorer", "estimator"])[
        "accuracy-mean", "accuracy-std"
    ]
    .mean()
    .reset_index()
)
std_source = df_mean.query("estimator == 'Train Src' & scorer == 'best_scorer'")[
    "accuracy-std"
].values

# remove estimators
# df_mean = df_mean.query("estimator != 'Train Tgt'")
# df_mean = df_mean.query("estimator != 'Train Src'")
df_mean = df_mean.query("scorer == 'best_scorer'")

# %%
# create the table
df_tab = df_mean.pivot(
    index="dataset", columns=["type", "estimator"], values="accuracy-mean"
)
df_tab = df_tab.round(2)

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
# %%
# add the colorcell
for i, col in enumerate(df_tab.columns):
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
        # "OfficeHome",
        "mnist_usps",
        "20NewsGroups",
        "AmazonReview",
        "Mushrooms",
        "Phishing",
        "BCI",
    ],
)
df_tab = df_tab.rename(
    columns={
        "covariate_shift": "\mcrot{1}{l}{45}{Cov. shift}",
        "target_shift": "\mcrot{1}{l}{45}{Tar. shift}",
        "concept_drift": "\mcrot{1}{l}{45}{Cond. shift}",
        "subspace": "\mcrot{1}{l}{45}{Sub. shift}",
        "Office31": "\mcrot{1}{l}{45}{Office31}",
        "mnist_usps": "\mcrot{1}{l}{45}{MNIST/USPS}",
        "20NewsGroups": "\mcrot{1}{l}{45}{20NewsGroups}",
        "AmazonReview": "\mcrot{1}{l}{45}{AmazonReview}",
        "Mushrooms": "\mcrot{1}{l}{45}{Mushrooms}",
        "Phishing": "\mcrot{1}{l}{45}{Phishing}",
        "BCI": "\mcrot{1}{l}{45}{BCI}",
    }
)
# convert to latex
lat_tab = df_tab.to_latex(
    escape=False,
    multicolumn_format="c",
    multirow=True,
    # remove [t] in multirow
    column_format="|l|l||rrrr||rr|rr|rr|r|",
    # round value
    # float_format="%.2f",
    #  put the name of multirow in vertical
)
lat_tab = lat_tab.replace("\type & estimator &  &  &  &  \\", "")
lat_tab = lat_tab.replace("toprule", "hline")
lat_tab = lat_tab.replace("midrule", "hline")
lat_tab = lat_tab.replace("cline{1-13}", "hline\hline")
lat_tab = lat_tab.replace("\multirow[t]", "\multirow")
lat_tab = lat_tab.replace("bottomrule", "hline")
lat_tab = lat_tab.replace("mnist_usps", "MNIST/USPS")

# %%
print(lat_tab)

# %%
