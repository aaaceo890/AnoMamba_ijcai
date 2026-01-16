import os
import os.path as path
import argparse
import logging
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from dataset import RecDataset

from model.AnoMamba import AnoMamba
from trainer import Trainer
from detect import detect
from baseline_read_dataset import read_dataset, provide_subdatas

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
# global parameters
METRICS = ["F1_af", "VUS_ROC"]
def init_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_q(dataset, subname=None):
    level = 0.98
    if dataset == "SMD":
        q = 0.018
        level = 0.98
    elif dataset == "ECG":
        q = 0.001
    elif dataset == "EPG":
        if subname in ["Lab2Cmac011215EPG2", "Lab2Cmac011215EPG4", "Lab2Cmac011215EPG6"]:
            q = 0.01
        elif subname in ["insectEPG1", "insectEPG2", "insectEPG3"]:
            q = 0.1
        else:
            q = 0.001
    elif dataset == "Gait":
        q = 0.01
    elif dataset == "NASA":
        if subname in ["TkeepMARS1", "TkeepMARS4"]:
            q = 0.1
        else:
            q = 0.005
    elif dataset == "LTDB":
        q = 0.02
    elif dataset == "MITDB":
        q = 0.01
    elif dataset == "SVDB":
        q = 0.01
    else:
        raise ValueError

    return q, level

def get_period(train_data, dataset, subname):
    if subname == "sddb49":
        period = 285
    elif dataset == "SMD":
        period = 1440
    else:
        try:
            T = train_data.shape[0]
            fft = np.fft.rfft(train_data, axis=0)
            est_freq = torch.topk(torch.tensor(fft).abs()[1:T // 2], 1, dim=0)[1] + 1
            est_period = [int(T / f) for f in est_freq]
            period = est_period[0]
        except:
            period = None

    return period

if __name__ == '__main__':
    # get parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help='dataset')
    parser.add_argument("--tag", type=str, default="debug", help='tag')
    parser.add_argument("--device", type=str, default="gpu", help='device')
    parser.add_argument("--save_value", action="store_true", help="save values")
    parser.add_argument("--test", action="store_true", help="test only")
    parser.add_argument("--seed",  nargs='+', type=int, default=[42], help="seed")

    parser.add_argument("--d_model", type=int, default=16, help="model size")
    parser.add_argument("--state_size", type=int, default=8, help="state size")
    parser.add_argument("--patch_size", type=int, default=8, help="patch_size")
    parser.add_argument("--patch_stride", type=int, default=4, help="patch_stride")
    parser.add_argument("--lambda_kl", type=float, default=0.1, help="lambda_kl")

    args = parser.parse_args()

    subdata = provide_subdatas(args.dataset)
    output_root = f"./result/{args.dataset}_{args.tag}"
    # run code with multi seeds
    for seed in args.seed:
        result_df = pd.DataFrame(columns=["Datasets"] + METRICS)

        for subname in subdata:
            logging.info(f"seed: {seed}")
            init_seed(seed=seed)

            output_dir = path.join(output_root, f"{subname}_{seed}")
            os.makedirs(path.join(output_root, f"fig_{seed}"), exist_ok=True)

            # get dataset
            train_data, test_data, test_label = read_dataset(args.dataset, subname)

            # get method info
            train_config = {"lr": 1e-3, "optim_conf": {"weight_decay": 0.00001},
                            "schedule_conf": {"step_num": 5, "decay": 0.9}, "batch_size": 16, "max_epochs": 8,
                            "log_period": 40, "num_recent_models": -1, "early_stop_count": -1, "train_loop": 1,
                            "down_stream": False, "test_bsz": 128, "norm_type": "norm",
                            }
            window_length = 100
            test_window_length = 100
            test_align = "causal_pad"

            # training
            L = int(0.9 * train_data.shape[0])
            period = get_period(train_data, args.dataset, subname)
            if period is None or (train_data.shape[0] - L) >= period * 5:
                train_split = train_data[:L]
                if train_data.shape[0] - L >= window_length:
                    val_split = train_data[L:]
                else:
                    val_split = train_data[-window_length:]
            else:
                train_split = train_data[:L]
                val_split = train_data[-period * 5:]

            train_dataset = RecDataset(train_split, label=np.zeros_like(train_split), dtype=np.float32,
                                       partition=True, shuffle=False, window_length=window_length,
                                       normalization_type="norm", align="none"
                                       )
            val_dataset = RecDataset(val_split, label=np.zeros_like(val_split), dtype=np.float32,
                                     partition=False, shuffle=False, window_length=window_length,
                                     normalization_type="norm", xscaler=train_dataset.xscaler, align="none"
                                     )

            clf = AnoMamba(
                input_size=train_dataset.input_dim,
                window_size=window_length,
                d_model=args.d_model,
                state_size=args.state_size,
                expand=2,
                block_num=3,
                patch_size=args.patch_size,
                patch_stride=args.patch_stride,
                lambda_kl=args.lambda_kl,
            )
            logging.info("Training...")

            trainer = Trainer(clf,
                              output_dir=path.join(output_dir),
                              init_model=None,
                              device=args.device,
                              **train_config
                              )
            if not args.test:
                trainer.fit(train_dataset, val_dataset=val_dataset)

            test_dataset = RecDataset(test_data, test_label, dtype=np.float32,
                                      partition=False, shuffle=False, window_length=test_window_length,
                                      xscaler=train_dataset.xscaler, align=test_align)
            test_dataloader = DataLoader(test_dataset, batch_size=train_config["test_bsz"], num_workers=0)
            init_dataset = RecDataset(train_data, np.zeros(train_data.shape[0]), dtype=np.float32,
                                      partition=False, shuffle=False, window_length=test_window_length,
                                      xscaler=train_dataset.xscaler, align=test_align)
            init_dataloader = DataLoader(init_dataset, batch_size=train_config["test_bsz"], num_workers=0)

            test_model = trainer.final_model
            results, labels = detect(clf, test_model, test_dataloader, device=args.device,
                                     init_dataloader=init_dataloader,
                                     pot_params=get_q(args.dataset, subname))

            if args.save_value:
                # save scores and threshold
                y_hats = results["y_hats"]
                scores = results["scores"]
                init_scores = results["init_scores"]
                th_pot = results["th_pot"]

                save_dir = output_dir

                np.save(path.join(save_dir, "y_hats.npy"), y_hats)
                np.save(path.join(save_dir, "scores.npy"), scores)
                np.save(path.join(save_dir, "init_scores.npy"), init_scores)
                np.save(path.join(save_dir, "labels.npy"), labels)
                np.save(path.join(save_dir, "th_pot.npy"), th_pot)

            dataname = subname if subname is not None else args.dataset

            row = {"Datasets": dataname}
            for m in METRICS:
                if m in results:
                    row.update({m: np.nan_to_num(results[m], 0)})

            row = pd.DataFrame(row, index=[0])
            result_df = pd.concat([result_df, pd.DataFrame(row, index=[0])], ignore_index=True)

        # save results
        result_df = result_df.sort_values(by="Datasets")
        avg_row = pd.DataFrame([{
            "Datasets": "Avg",
            "F1_af": result_df["F1_af"].mean(),
            "VUS_ROC": result_df["VUS_ROC"].mean()
        }])
        result_df = pd.concat([result_df, avg_row], ignore_index=True)
        result_df.to_csv(path.join(path.abspath(
            path.join(output_root, f"AnoMamba_{args.dataset}_{args.tag}_{seed}_result_valid_best.csv"))),
                         index=False)
    # summarize all seeds
    csv_files = [path.abspath(path.join(output_root, f"AnoMamba_{args.dataset}_{args.tag}_{seed}_result_valid_best.csv"))
                     for seed in args.seed]
    df_list = [pd.read_csv(f) for f in csv_files]
    df_all = pd.concat(df_list, ignore_index=True)
    grouped = df_all.groupby(["Datasets"]).agg(['mean', 'std'])
    grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
    grouped.to_csv(path.abspath(path.join(output_root,
                                            f"all_seeds_summary_AnoMamba_{args.dataset}_{args.tag}_result.csv")))

