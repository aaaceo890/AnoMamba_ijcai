import os
import os.path as path

import pickle
import numpy as np
import pandas as pd

import matplotlib
from matplotlib import pyplot as plt

def provide_subdatas(dataset):
    if dataset == "ECG":
        subdatas = [
            "s20101m",
            "s20101mML2",
            "sddb49"
        ]
    elif dataset == "EPG":
        subdatas = [f"Lab2Cmac011215EPG{i}" for i in range(1, 7)]
        subdatas += [f"insectEPG{i}" for i in range(1, 6)]
    elif dataset == "Gait":
        subdatas = [f"GP711MarkerLFM5z{i}" for i in range(1, 6)]
    elif dataset == "NASA":
        subdatas = [f"TkeepMARS{i}" for i in range(1, 6)]
    elif dataset == "LTDB":
        subdatas = [f"LTDB_{i}_Medical" for i in range(1, 6)]
    elif dataset == "MITDB":
        subdatas = [f"MITDB_{i}_Medical" for i in range(1, 14)]
    elif dataset == "SVDB":
        subdatas = [f"SVDB_{i}_Medical" for i in range(1, 32)]
    elif dataset == "SMD":
        subdatas_1 = [f"machine-1-{i+1}" for i in range(8)]
        subdatas_2 = [f"machine-2-{i+1}" for i in range(9)]
        subdatas_3 = [f"machine-3-{i+1}" for i in range(11)]
        subdatas = subdatas_1 + subdatas_2 + subdatas_3
    else:
        raise ValueError

    return subdatas

def read_dataset(dataset, subname=None):
    if dataset in ["ECG", "EPG", "Gait", "NASA"]:
        data_root = f"./data_process/univariate/{dataset}"
        train_pack = pd.read_csv(path.join(data_root, f"{subname}.train.csv"))
        train_data = train_pack.iloc[:, 1:-1].values
        test_pack = pd.read_csv(path.join(data_root, f"{subname}.test.csv"))
        test_data = test_pack.iloc[:, 1:-1].values
        test_label = test_pack.iloc[:, -1].values

    elif dataset == "SMD":
        data_root = "./data_process/multivariate/SMD"
        train_data = pd.read_csv(path.join(data_root, f"{subname}.train.csv")).values[:, 1:-1]
        test_data = pd.read_csv(path.join(data_root, f"{subname}.test.csv")).values[:, 1:-1]
        single_label = pd.read_csv(path.join(data_root, f"{subname}.test.csv")).values[:, -1]
        test_label = single_label[:, np.newaxis]

    elif dataset in ["LTDB", "MITDB", "SVDB"]:
        data_root = f"./data_process/multivariate/{dataset}"
        train_data = pd.read_csv(path.join(data_root, f"{subname}.train.csv")).values[:, 1:-1]
        test_data = pd.read_csv(path.join(data_root, f"{subname}.test.csv")).values[:, 1:-1]
        test_label = pd.read_csv(path.join(data_root, f"{subname}.test.csv")).values[:, -1]
        test_label = test_label[:, np.newaxis]

    else:
        raise NotImplementedError

    train_data = train_data.astype(np.float64)
    test_data = test_data.astype(np.float64)
    train_data = np.nan_to_num(train_data)
    test_data = np.nan_to_num(test_data)
    test_label = test_label.astype(np.int64)

    return train_data, test_data, test_label
