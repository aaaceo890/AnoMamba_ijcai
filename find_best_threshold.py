import os
import os.path as path
import numpy as np
import pandas as pd
from tqdm import tqdm

from baseline_read_dataset import provide_subdatas
from baseline_evaluate import getAffiliationMetrics
from evaluate.spot import SPOT

def get_q(subdata, outlier_type):
    level = 0.98
    # if "machine-1" in subdata:
    #     q = 0.018
    # elif "machine-2" in subdata:
    #     # q = 0.005
    #     q = 0.018
    # elif "machine-3" in subdata:
    #     q = 0.018
    if "machine" in subdata:
        q = 0.03

    elif outlier_type == "Daphnet_HumanActivity":
        q = 0.01
    elif outlier_type == "LTDB_Medical":
        q = 0.015
    elif outlier_type == "MITDB_Medical":
        q = 0.01
    elif outlier_type == "SVDB_Medical":
        q = 0.01
    # UCR
    elif outlier_type == "UCR-ECG-v":
        q = 0.001 # confirmed
    elif outlier_type == "UCR-EPG":
        # confirmed
        if subdata in ["Lab2Cmac011215EPG2", "Lab2Cmac011215EPG4", "Lab2Cmac011215EPG6"]:
            q = 0.01
        elif subdata in ["insectEPG1", "insectEPG2", "insectEPG3"]:
            q = 0.1
        else:
            q = 0.001
        # q, level = 0.01, 0.98
    elif outlier_type == "UCR-Gait":
        # 0.69
        q = 0.005
    elif outlier_type == "UCR-GaitPhase":
        # 0.8 confirmed
        q = 0.01
    elif outlier_type == "UCR-NASA":
        # 0.76 confirmed
        if subdata in ["TkeepMARS1", "TkeepMARS4"]:
            q = 0.1
        else:
            q = 0.005
    else:
        raise ValueError

    return q, level


    # METRICS = ["P_af", "R_af", "F1_af", "th"]

METRICS = ["P_af", "R_af", "F1_af"]



source_dir = "/home/Chenjq/Project/AnoMamba_org/result/SMD_new_alpha"
dataset = "SMD"

subdatas = provide_subdatas(dataset)
df_list = []
for seed in [42, 123, 2025, 3407, 7777]:
# for seed in [42]:
    result_df = pd.DataFrame(columns=["Datasets", "Outlier_type"] + METRICS)

    for subdata in tqdm(subdatas):
        outlier_type = "all"
        # if "LTDB" not in subdata:
        #     continue
        q, level = get_q(subdata, outlier_type)

        labels = np.load(path.join(source_dir, f"{subdata}_{seed}", "labels.npy"))
        scores = np.load(path.join(source_dir, f"{subdata}_{seed}", "scores.npy"))
        init_scores = np.load(path.join(source_dir, f"{subdata}_{seed}", "init_scores.npy"))

        spot = SPOT(init_scores, q=q)
        try:
            spot.initialize(level=level)
            th_pot = spot.extreme_quantile
        except:
            th_pot = np.sort(init_scores)[-int(q * init_scores.shape[0])]

        # th_pot = np.sort(scores)[-int(ratio * scores.shape[0])]

        pred = (scores > th_pot).astype(int)

        res = {}
        precision, recall, f1_score = getAffiliationMetrics(labels.copy(), pred.copy())
        res['P_af'] = precision
        res['R_af'] = recall
        res['F1_af'] = f1_score
        # res['th'] = th_pot

        row = {"Datasets": subdata, "Outlier_type": outlier_type}
        for m in METRICS:
            if m in res:
                row.update({m: np.nan_to_num(res[m], 0)})

        row = pd.DataFrame(row, index=[0])
        result_df = pd.concat([result_df, pd.DataFrame(row, index=[0])], ignore_index=True)

    result_df = result_df.sort_values(by="Datasets")
    avg_df = result_df.groupby("Outlier_type").mean().reset_index().set_index("Outlier_type").reset_index()
    # avg_df = avg_df.append(pd.Series(avg_df.mean(numeric_only=True), name="total")).reset_index()
    avg_df["Datasets"] = "Avg_" + avg_df["Outlier_type"]
    result_df = pd.concat([result_df, avg_df], ignore_index=True)

    pd.set_option('display.max_columns',None)
    pd.set_option('display.max_rows',None)
    df_list.append(result_df)

df_all = pd.concat(df_list, ignore_index=True)
grouped = df_all.groupby(["Datasets", "Outlier_type"]).agg(['mean', 'std'])
grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
print(grouped)


