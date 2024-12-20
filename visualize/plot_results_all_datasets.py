# %%
import json
import argparse

import glob
import numpy as np
import pandas as pd
import scipy.stats as stats


def shade_of_color_pvalue(
    df_value,
    pvalue,
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
        if pvalue < color_threshold:
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
                return "\\cellcolor{good_color!%d}{%s}" % (intensity, df_value)
            else:
                red_min = min_value
                red_max = mean_value
                if red_min - red_max == 0:
                    # To avoid division by zero
                    intensity = intensity_range[0]
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


def generate_table(
    csv_folder,
    output_folder,
    csv_folder_simulated,
    scorer_selection="unsupervised",
    score="accuracy",
    output_format="latex",
):
    csv_files = glob.glob(f"{csv_folder}/*.csv")
    df = pd.concat([pd.read_csv(f) for f in csv_files])

    csv_files_simulated = glob.glob(f"{csv_folder_simulated}/*.csv")
    df_simulated = pd.concat([pd.read_csv(f) for f in csv_files_simulated])

    df_simulated["dataset"] = df_simulated["shift"]

    df = pd.concat([df, df_simulated])

    df = df.query("estimator != 'NO_DA_SOURCE_ONLY_BASE_ESTIM'")
    df[f"target_{score}-test-identity"] = df[
        f"target_{score}-test-identity"
    ].apply(
        lambda x: json.loads(x)
    )

    df["nb_splits"] = df[
        f"target_{score}-test-identity"
    ].apply(lambda x: len(x))

    df_target = df.query('estimator == "Train Tgt" & scorer == "supervised"')
    df_source = df.query('estimator == "Train Src" & scorer == "supervised" ')

    df = df.merge(
        df_target[
            ["shift", f"target_{score}-test-mean", f"target_{score}-test-std"]
        ],
        on="shift",
        suffixes=("", "_target"),
    )
    df = df.merge(
        df_source[
            [
                "shift",
                f"target_{score}-test-mean",
                f"target_{score}-test-std",
                f"target_{score}-test-identity",
            ]
        ],
        on="shift",
        suffixes=("", "_source"),
    )
    # # remove rows where the source is better than the target
    # df = df[
    #     df[f"target_{score}-test-mean_source"]
    #     < df[f"target_{score}-test-mean_target"]
    # ].reset_index()
    # df = df.query("nb_splits == 5 | nb_splits == 25")

    # remove duplicates
    df = df.drop_duplicates(subset=["dataset", "scorer", "estimator", "shift"])

    # # count the number of shifts
    # df_shift = df.groupby(["dataset", "scorer", "estimator"])
    # df_shift = df_shift.agg({"shift": "count"}).reset_index()
    # df_shift["nb_shift"] = df_shift["shift"]
    # nb_shifts_per_dataset = {
    #     "Office31": int(0.8 * 5),
    #     "OfficeHomeResnet": int(0.8 * 12),
    #     "mnist_usps": 2,
    #     "20NewsGroups": int(0.8 * 6),
    #     "AmazonReview": int(0.8 * 11),
    #     "Mushrooms": int(0.8 * 2),
    #     "Phishing": int(0.8 * 2),
    #     "BCI": int(0.8 * 9),
    #     "covariate_shift": 1,
    #     "target_shift": 1,
    #     "concept_drift": 1,
    #     "subspace": 1,
    # }

    # df_shift["nb_shift_max"] = df_shift["dataset"].apply(
    #     lambda x: nb_shifts_per_dataset[x]
    # )

    # df = df.merge(
    #     df_shift[[
    #         "dataset", "scorer", "estimator", "nb_shift", "nb_shift_max"
    #     ]], on=["dataset", "scorer", "estimator"],
    # )
    # df = df[df["nb_shift"] >= df["nb_shift_max"]]

    df_filtered = df.query("estimator != 'Train Tgt'")
    df_filtered = df_filtered.query("estimator != 'Train Src'")
    df_grouped = df_filtered.groupby(["dataset", "scorer", "estimator"])
    wilco = []
    scorer = []
    estimator = []
    dataset = []
    for idx, df_ in df_grouped:
        # test de wilcoxon
        acc_da = np.concatenate(df_[f"target_{score}-test-identity"].values)
        acc_source = np.concatenate(
            df_[f"target_{score}-test-identity_source"].values
        )
        try:
            wilco.append(
                stats.wilcoxon(
                    acc_da,
                    acc_source,
                )[1]
            )
            scorer.append(df_["scorer"].values[0])
            estimator.append(df_["estimator"].values[0])
            dataset.append(df_["dataset"].values[0])
        except ValueError:
            wilco.append(1)
            scorer.append(df_["scorer"].values[0])
            estimator.append(df_["estimator"].values[0])
            dataset.append(df_["dataset"].values[0])

    df_wilco = pd.DataFrame(
        {
            "scorer": scorer,
            "estimator": estimator,
            "pvalue": wilco,
            "dataset": dataset,
        }
    )

    df["rank"] = df.groupby(["dataset", "scorer", "shift"])[
        f"target_{score}-test-mean"
    ].rank(ascending=False)

    df_no_simutaled = df.query(
        "dataset != 'covariate_shift' & dataset != 'target_shift' & "
        "dataset != 'concept_drift' & dataset != 'subspace'"
    )
    df_rank = df_no_simutaled.groupby(
        ["estimator"]
    )["rank"].mean().reset_index()

    df_mean = (
        df.groupby(["dataset", "type", "scorer", "estimator"])
        .agg(
            {
                f"target_{score}-test-mean": lambda x: x.mean(skipna=False),
                f"target_{score}-test-std": lambda x: x.mean(skipna=False),
                "rank": lambda x: x.mean(skipna=False),
            }
        )
        .reset_index()
    )

    df_source_mean = df_mean.query(
        "estimator == 'Train Src' & scorer == 'supervised'"
    )
    df_target_mean = df_mean.query(
        "estimator == 'Train Tgt' & scorer == 'supervised'"
    )

    df_mean = df_mean.query("estimator != 'Train Src'")
    df_mean = df_mean.query("estimator != 'Train Tgt'")

    if scorer_selection == "supervised":
        df_tot = df_mean.query("scorer == 'supervised'")
        df_wilco = df_wilco.query("scorer == 'supervised'")

    elif scorer_selection == "unsupervised":
        df_mean = df_mean.query(
            "scorer != 'supervised' & scorer != 'best_scorer'"
        )
        best_scorers = df_mean.query(
            "dataset != 'covariate_shift' & dataset != 'target_shift' "
            "& dataset != 'concept_drift' & dataset != 'subspace'"
        )
        best_scorers = (
            best_scorers.groupby(
                ["estimator", "scorer"]
            )[f"target_{score}-test-mean"]
            .mean()
            .reset_index()
        )
        idx_best_scorer = best_scorers.groupby(["estimator"])[
            f"target_{score}-test-mean"
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

        df_tot = df_tot.query("scorer == scorer_best").reset_index()

        df_wilco = df_wilco[
            ["dataset", "estimator", "scorer", "pvalue"]
        ].merge(
            best_scorers[["estimator", "scorer"]],
            on=[
                "estimator",
            ],
            suffixes=("", "_best"),
        )

        df_wilco = df_wilco.query("scorer == scorer_best").reset_index()

    # %%
    df_tot = pd.concat(
        [df_tot, df_source_mean, df_target_mean], axis=0
    ).reset_index()

    df_tab = df_tot.pivot(
        index="dataset",
        columns=["type", "estimator"],
        values=f"target_{score}-test-mean",
    )
    df_tab = df_tab.reindex(
        columns=["NO DA", "Reweighting", "Mapping", "Subspace", "Other"],
        level=0
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

    if output_format == "latex":
        df_tab = df_tab.T.rename(
            index={
                "NO DA": r"\rotatebox[origin=c]{90}{}",
                "Reweighting": r"\rotatebox[origin=c]{90}{Reweighting}",
                "Mapping": r"\rotatebox[origin=c]{90}{Mapping}",
                "Subspace": r"\rotatebox[origin=c]{90}{Subspace}",
                "Other": r"\rotatebox[origin=c]{90}{Other}",
            }
        )
    else:
        df_tab = df_tab.T

    df_tab = df_tab.reset_index().merge(df_rank, on="estimator")

    if scorer_selection == "unsupervised":
        df_best_scorer = df_tot[["estimator", "scorer_best"]].drop_duplicates()
        df_tab = df_tab.merge(df_best_scorer, on="estimator")
        df_tab = df_tab[
            df_tot["scorer"] == df_tot["scorer_best"]
        ].reset_index()

    df_tab = df_tab.set_index(["type", "estimator"])
    df_tab = df_tab.round(2)
    # remove columns index
    if scorer_selection == "unsupervised":
        df_tab = df_tab[df_tab.columns[1:]]
    
    if output_format == "latex":
        # add the colorcell
        for i, col in enumerate(
            df_tab.columns[: -2 if scorer_selection == "unsupervised" else -1]
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
                pvalue = df_wilco.query(
                    f"estimator == '{idx[1]}' & dataset == '{col}'"
                )[
                    "pvalue"
                ].values[0]
                color = shade_of_color_pvalue(
                    value,
                    pvalue,
                    min_value=min_value,
                    mean_value=mean_value,
                    max_value=max_value,
                )
                df_tab.loc[idx, col] = color
            df_tab.loc[df_tab.index[1], col] = "\\cellcolor{good_color!%d}{%s}" % (
                60,
                df_tab.loc[df_tab.index[1], col],
            )

    if scorer_selection == "supervised":
        df_tab = df_tab.reindex(
            columns=[
                "covariate_shift",
                "target_shift",
                "concept_drift",
                "subspace",
                "Office31Decaf",
                "OfficeHomeResnet",
                "mnist_usps_pca",
                "20NewsGroups",
                "AmazonReview",
                "Mushrooms",
                "Phishing",
                "bci_projected",
                "rank",
            ],
        )
    else:
        df_tab = df_tab.reindex(
            columns=[
                "covariate_shift",
                "target_shift",
                "concept_drift",
                "subspace",
                "Office31Decaf",
                "OfficeHomeResnet",
                "mnist_usps_pca",
                "20NewsGroups",
                "AmazonReview",
                "Mushrooms",
                "Phishing",
                "bci_projected",
                "scorer_best",
                "rank",
            ],
        )
    
    if output_format == "latex":
        df_tab = df_tab.rename(
            columns={
                "covariate_shift": "\\mcrot{1}{l}{45}{\\underline{Cov. shift}}",
                "target_shift": "\\mcrot{1}{l}{45}{\\underline{Tar. shift}}",
                "concept_drift": "\\mcrot{1}{l}{45}{\\underline{Cond. shift}}",
                "subspace": "\\mcrot{1}{l}{45}{\\underline{Sub. shift}}",
                "Office31Decaf": "\\mcrot{1}{l}{45}{Office31}",
                "OfficeHomeResnet": "\\mcrot{1}{l}{45}{OfficeHome}",
                "mnist_usps_pca": "\\mcrot{1}{l}{45}{MNIST/USPS}",
                "20NewsGroups": "\\mcrot{1}{l}{45}{20NewsGroups}",
                "AmazonReview": "\\mcrot{1}{l}{45}{AmazonReview}",
                "Mushrooms": "\\mcrot{1}{l}{45}{Mushrooms}",
                "Phishing": "\\mcrot{1}{l}{45}{Phishing}",
                "bci_projected": "\\mcrot{1}{l}{45}{BCI}",
                "scorer_best": "\\mcrot{1}{l}{45}{Selected Scorer}",
                "rank": "\\mcrot{1}{l}{45}{Rank}",
            }
        )

        df_tab = df_tab.fillna("\\color{gray!90}NA")

        # convert to latex
        lat_tab = df_tab.to_latex(
            escape=False,
            multicolumn_format="c",
            multirow=True,
            column_format="|l|l||rrrr||rrr|rr|rr|r||rr|",
        )
        lat_tab = lat_tab.replace("\type & estimator &  &  &  &  \\", "")
        lat_tab = lat_tab.replace("toprule", "hline")
        lat_tab = lat_tab.replace("midrule", "hline")
        if scorer == "supervised":
            lat_tab = lat_tab.replace("cline{1-15}", "hline\\hline")
        else:
            lat_tab = lat_tab.replace("cline{1-16}", "hline\\hline")
        lat_tab = lat_tab.replace("\\multirow[t]", "\\multirow")
        lat_tab = lat_tab.replace("bottomrule", "hline")
        lat_tab = lat_tab.replace("mnist_usps", "MNIST/USPS")
        lat_tab = lat_tab.replace("OfficeHomeResnet", "OfficeHome")
        lat_tab = lat_tab.replace("circular_validation", "CircV")
        lat_tab = lat_tab.replace("prediction_entropy", "PE")
        lat_tab = lat_tab.replace("importance_weighted", "IW")
        lat_tab = lat_tab.replace("soft_neighborhood_density", "SND")
        lat_tab = lat_tab.replace("deep_embedded_validation", "DEV")
        lat_tab = lat_tab.replace("mix_val_intra", "MixVal")
        lat_tab = lat_tab.replace("mix_val_both", "MixVal")
        lat_tab = lat_tab.replace("mix_val_inter", "MixVal")

        # save to txt file
        with open(
            f"{output_folder}/table_results_all_dataset_{scorer_selection}_{score}.txt", "w"
        ) as f:
            f.write(lat_tab)
            
    elif output_format == "markdown":
        df_tab = df_tab.fillna("NA")
        
        # Remove type in the index of df
        new_index = df_tab.index.get_level_values('estimator')
        df_tab.index = new_index

        # Convert to markdown
        md_tab = df_tab.to_markdown(index=True)

        # save to md file
        with open(
            f"{output_folder}/table_results_shallow_all_dataset_{scorer_selection}_{score}.md", "w"
        ) as f:
            f.write(md_tab)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate main table for all datasets",
    )

    parser.add_argument(
        "--csv-folder",
        type=str,
        help="Path to the csv file containing results for real data",
        default="./readable_csv/results_all_datasets_experiments.csv",
    )

    parser.add_argument(
        "--csv-folder-simulated",
        type=str,
        help="Path to the csv file containing results for real data",
        default="./readable_csv/results_all_datasets_experiments.csv",
    )

    parser.add_argument(
        "--output-folder",
        type=str,
        help="Path to the output folder",
        default="./",
    )

    parser.add_argument("--scorer-selection", type=str, default="unsupervised")

    parser.add_argument("--score", type=str, default="accuracy")

    parser.add_argument(
        "--output-format",
        type=str,
        choices=["latex", "markdown"],
        default="latex",
        help="Output format for the table (latex or markdown)"
    )

    args = parser.parse_args()
    df = generate_table(
        csv_folder=args.csv_folder,
        csv_folder_simulated=args.csv_folder_simulated,
        output_folder=args.output_folder,
        scorer_selection=args.scorer_selection,
        score=args.score,
        output_format=args.output_format,
    )
