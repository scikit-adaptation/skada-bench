import pandas as pd
import yaml

with open("../config/best_base_estimators.yml") as stream:
    best_base_estimators = yaml.safe_load(stream)

df = pd.read_csv("../results/results_base_estim_experiments.csv")

def rename_estimator(x):
    return x.split("['")[-1].split("']")[0]

def extract_svc(x):
    if "SVC" in x:
        return True
    else:
        return False

df["estimator"] = df["estimator"].apply(rename_estimator)
df = df.loc[df["scorer"] == "supervised"]
df = df.drop(["type", "shift", "scorer"], axis=1)

# Find best Estim
for dataset in df.dataset.unique():
    
    df_best = df.loc[df.dataset == dataset]
    df_best = df_best.groupby(["dataset", "estimator"]).mean().reset_index()
    mask_svc = df_best.estimator.apply(extract_svc)
    df_best_svc = df_best.loc[mask_svc]
    
    best_estim = df_best.iloc[df_best["accuracy-mean"].argmax()].estimator
    best_acc = df_best.iloc[df_best["accuracy-mean"].argmax()]["accuracy-mean"]
    
    best_estim_svc = df_best_svc.iloc[df_best_svc["accuracy-mean"].argmax()].estimator
    best_acc_svc = df_best_svc.iloc[df_best_svc["accuracy-mean"].argmax()]["accuracy-mean"]

    print(dataset, "Best:", best_estim, best_acc)
    print(dataset, "Best SVC:", best_estim_svc, best_acc_svc)

    best_base_estimators[dataset] = dict(Best=best_estim, BestSVC=best_estim_svc)

with open("../config/best_base_estimators.yml", 'w+') as ff:
    yaml.dump(best_base_estimators, ff, default_flow_style=False)