import pandas as pd
import yaml
import os

PATH = os.path.dirname(os.path.dirname(__file__))
CONFIG_FILE = os.path.join(PATH, "config", "best_base_estimators.yml")
RESULT_FILE = os.path.join(
    PATH, "results_base_estimators", "results_base_estim_experiments.csv"
)

SCORE_COL = "source_accuracy-test-mean"


if __name__ == "__main__":

    with open(CONFIG_FILE) as stream:
        best_base_estimators = yaml.safe_load(stream)

    df = pd.read_csv(RESULT_FILE)

    def rename_params(x):
        return x.split("['")[-1].split("']")[0]

    def extract_svc(x):
        if "SVC" in x:
            return True
        else:
            return False

    df["params"] = df["params"].apply(rename_params)
    df = df.loc[df["scorer"] == "supervised"]
    df = df.loc[df["estimator"] == "NO_DA_SOURCE_ONLY_BASE_ESTIM"]

    # Find best Estim
    for dataset in df.dataset.unique():

        df_best = df.loc[df.dataset == dataset]
        df_best = (
            df_best.groupby(["dataset", "params"])
            .mean(numeric_only=True).reset_index()
        )
        mask_svc = df_best.params.apply(extract_svc)
        df_best_svc = df_best.loc[mask_svc]

        best_estim = df_best.iloc[df_best[SCORE_COL].argmax()].params
        best_acc = df_best.iloc[df_best[SCORE_COL].argmax()][SCORE_COL]

        id_best = df_best_svc[SCORE_COL].argmax()
        best_estim_svc = df_best_svc.iloc[id_best].params
        best_acc_svc = df_best_svc.iloc[id_best][SCORE_COL]

        print(dataset, "Best:", best_estim, best_acc)
        print(dataset, "Best SVC:", best_estim_svc, best_acc_svc)

        best_base_estimators[dataset] = dict(
            Best=best_estim, BestSVC=best_estim_svc
        )

    with open(CONFIG_FILE, 'w+') as ff:
        yaml.dump(best_base_estimators, ff, default_flow_style=False)
