import random
from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from typing import OrderedDict, Union

import numpy as np
import torch
from pathlib import Path
import pandas as pd
import os
from tqdm import tqdm

PROJECT_DIR = Path(__file__).parent.parent.parent.resolve()
LOG_DIR = PROJECT_DIR / "logs"
TEMP_DIR = PROJECT_DIR / "temp"
DATA_DIR = PROJECT_DIR / "data"


def fix_random_seed(seed: int) -> None:
    torch.cuda.empty_cache()
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def clone_parameters(
    src: Union[OrderedDict[str, torch.Tensor], torch.nn.Module]
) -> OrderedDict[str, torch.Tensor]:
    if isinstance(src, OrderedDict):
        return OrderedDict(
            {
                name: param.clone().detach().requires_grad_(param.requires_grad)
                for name, param in src.items()
            }
        )
    if isinstance(src, torch.nn.Module):
        return OrderedDict(
            {
                name: param.clone().detach().requires_grad_(param.requires_grad)
                for name, param in src.state_dict(keep_vars=True).items()
            }
        )


def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--global_epochs", type=int, default=1000)
    parser.add_argument("--local_epochs", type=int, default=10)
    parser.add_argument("--local_lr", type=float, default=1e-2)
    parser.add_argument("--verbose_gap", type=int, default=20)
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["mnist", "cifar10", "cifar100", "emnist", "fmnist"],
        default="mnist",
    )
    parser.add_argument("--batch_size", type=int, default=-1)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--log", type=int, default=0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--client_num_per_round", type=int, default=2)
    parser.add_argument("--save_period", type=int, default=20)
    parser.add_argument("--config_idx", type=int, default=-1, help="CSV config index")
    parser.add_argument("--model", type=str, default="cnn", choices=["cnn", "logistic"])
    parser.add_argument("--mu", type=float, default=1.0, help="FedProx regularization")
    parser.add_argument("--scaffold_variant", type=int, default=2, help="SCAFFOLD variant (1 or 2)")

    parser.add_argument("--dataset_type", type=str, default="user")
    parser.add_argument("--data_root",
                        type=str,
                        default=None,
                        help="root dir that contains <dataset>/pickles for this run")
    parser.add_argument("--reset", action="store_true", help="Reset model before training", default=1)

    args = parser.parse_args()

    # 자동 하이퍼파라미터 적용
    if args.config_idx >= 0:
        csv_path = os.path.join(PROJECT_DIR, "experiment_config_from_paper.csv")
        df = pd.read_csv(csv_path)
        row = df.iloc[args.config_idx]
        args.local_epochs = int(row["epochs"])
        batch_frac = float(row["batch_fraction"])
        args.batch_size = int(32 * batch_frac)
        args.dataset = "emnist"
        #args.client_num_per_round = 20
        args.local_lr = 0.1
        args.model = "logistic"
        args.mu = 1.0
        args.scaffold_variant = 2
        args.client_num_per_round = 1 if row["algorithm"] == "sgd" else 20
        sim = int(row["similarity"])
        args.data_root = str(PROJECT_DIR / f"data/datasets/emnist_sim{sim}")
        # 실험 정보 출력
        print("Running experiment:")
        print(f"  Algorithm: {row['algorithm']}")
        print(f"  Similarity: {row['similarity']}")
        print(f"  Epochs: {row['epochs']}")
        print(f"  Batch fraction: {row['batch_fraction']}")

    return args


def run_all_experiments():
    csv_path = os.path.join(PROJECT_DIR, "experiment_config_from_paper.csv")
    df = pd.read_csv(csv_path)
    result_dir = os.path.join(PROJECT_DIR, "experiment_results")
    os.makedirs(result_dir, exist_ok=True)

    for i in tqdm(range(len(df)), desc="Running Experiments"):
        row = df.iloc[i]
        algorithm = row["algorithm"]
        output_file = os.path.join(result_dir, f"result_{i}_{algorithm}.txt")

        cmd = (
            f"python ./src/server/{algorithm}.py "
            f"--config_idx {i} > {output_file}"
        )
        print(f"\n▶️ Running: {cmd}")
        os.system(cmd)


if __name__ == "__main__":
    run_all_experiments()
