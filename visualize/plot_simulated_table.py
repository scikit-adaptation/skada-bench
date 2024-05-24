# %%
import pandas as pd
import numpy as np


# %%
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


def create_latex_table(df, std, normalized=False):

    # Rename the variables and add rotation
    df = df.rename(
        index={
            "concept_drift": "Cond. shift",
            "covariate_shift": "Cov. shift",
            "subspace": "Sub. shift",
            "target_shift": "Tar. shift",
        }
    )

    df = df.reindex(
        columns=["NO DA", "Reweighting", "Mapping", "Subspace", "Other"],
        level=0
    )
    df = df.T.rename(
        index={
            "NO DA": r"\rotatebox[origin=c]{90}{NA}",
            "Reweighting": r"\rotatebox[origin=c]{90}{Reweighting}",
            "Mapping": r"\rotatebox[origin=c]{90}{Mapping}",
            "Subspace": r"\rotatebox[origin=c]{90}{Subspace}",
            "Other": r"\rotatebox[origin=c]{90}{Other}",
        }
    )
    df = df.reindex(
        columns=[
            "Cov. shift",
            "Tar. shift",
            "Cond. shift",
            "Sub. shift",
        ],
        level=0,
    )

    # add the colorcell
    for i, col in enumerate(df.columns):
        if normalized:
            max_value = 1
            color_threshold = 0
        else:
            max_value = df[col].loc[df.index[1]]
            color_threshold = df[col].loc[df.index[0]]
        min_value = df[col].min()
        std_value = std[i]
        for idx in df.index[2:]:
            # get the value
            value = df.loc[idx, col]
            # get the color
            color = shade_of_color(
                value,
                std=std_value,
                min_value=min_value,
                max_value=max_value,
                color_threshold=color_threshold,
            )
            df.loc[idx, col] = color
        df.loc[df.index[1], col] = "\\cellcolor{good_color!%d}{%s}" % (
            90,
            df.loc[df.index[1], col],
        )

    # convert to latex
    lat_tab = df.to_latex(
        escape=False,
        multicolumn_format="c",
        multirow=True,
        # remove [t] in multirow
        column_format="|l|l" + "|c" * 4 + "|",
        # round value
        float_format="%.2f",
        #  put the name of multirow in vertical
    )
    lat_tab = lat_tab.replace("\type & estimator &  &  &  &  \\", "")
    lat_tab = lat_tab.replace("toprule", "hline")
    lat_tab = lat_tab.replace("midrule", "hline")
    lat_tab = lat_tab.replace("cline{1-6}", "hline\hline")
    lat_tab = lat_tab.replace("\multirow[t]", "\multirow")
    lat_tab = lat_tab.replace("\\hline\n\\bottomrule", "")

    return lat_tab


# %%
df = pd.read_csv("simulated_readable_csv.csv")
# %%
# get the source and target df
df_target = df.query('estimator == "Train Tgt" & scorer == "supervised"')
df_source = df.query(
    'estimator == "Train Src" & scorer != "supervised" & scorer != "best_scorer"'
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

# %%
df_other_ = df.query('estimator != "Train Tgt" and estimator != "Train Src"')
df_other = df_other_.query('scorer != "best_scorer" and scorer != "supervised"')
# %%
# get the indices of  the line where we have the best scorer per estimator and shift
idx_best_scorer = df_other.groupby(["estimator", "shift"])["accuracy-mean"].idxmax()
df_best_scorer = df_other.loc[idx_best_scorer]

# %%

std_values = df_source["accuracy-std"].values
std_values = std_values[
    [1, 3, 0, 2]
]  # WARNING: reorder the std values, but need to be change in the future

# %%
# plot the tab with the accuracy-mean
df_tab = df_best_scorer.pivot(
    index="shift", columns=["type", "estimator"], values="accuracy-mean"
)

df_tot_tab = pd.concat([df_source_target_tab, df_tab], axis=1)
df_tot_tab = df_tot_tab.round(2)

lat_tab = create_latex_table(
    df_tot_tab,
    std_values,
)
# %%
print(lat_tab)
