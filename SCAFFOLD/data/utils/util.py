import json
import math
import os
import pickle
from typing import Dict, List, Tuple, Union

from pathlib import Path
from torch.utils.data import Subset, random_split

_CURRENT_DIR = Path(__file__).parent.resolve()
_ARGS_DICT = json.load(open(_CURRENT_DIR.parent / "args.json", "r"))

import sys
import data.utils.dataset as actual_dataset
sys.modules['dataset'] = actual_dataset

def get_dataset(
    dataset: str,
    client_id: int,
    data_root: str | Path | None = None,   # ⬅️  새 인자
) -> Dict[str, Subset]:

    """
    Parameters
    ----------
    dataset   : "emnist" · "mnist" · …
    client_id : 정수형 클라이언트 ID
    data_root :  data/datasets/emnist_sim{sim}  처럼
                 ‘실험 버전 폴더’의 절대·상대 경로.
                 None → 기존 기본 경로(data/<dataset>/pickles) 사용
    """

    client_num_in_each_pickles = _ARGS_DICT["client_num_in_each_pickles"]

    base = Path(data_root) if data_root else (
        Path(_ARGS_DICT.get("root", _CURRENT_DIR.parent))
    )

    if base.name != dataset:        # root 안에 이미 emnist 폴더가 없으면
        base = base / dataset       #   하나 더 붙인다

    pickles_dir = base / "pickles"

    if os.path.isdir(pickles_dir) is False:
        raise RuntimeError("Please preprocess and create pickles first.")

    pickle_path = (
        pickles_dir / f"{math.floor(client_id / client_num_in_each_pickles)}.pkl"
    )
    with open(pickle_path, "rb") as f:
        subset = pickle.load(f)
    client_dataset = subset[client_id % client_num_in_each_pickles]
    trainset = client_dataset["train"]
    valset = client_dataset["val"]
    testset = client_dataset["test"]
    return {"train": trainset, "val": valset, "test": testset}



def get_client_id_indices(dataset, data_root=None):
    """
    data_root :  data/datasets/emnist_sim{sim}  형태.
                 None 이면 기존 기본 경로(data/<dataset>/pickles) 사용
    """
    base = Path(data_root) if data_root else _CURRENT_DIR.parent / dataset
    pickles_dir = base / "emnist" / "pickles"

    with open(pickles_dir / "seperation.pkl", "rb") as f:
        sep = pickle.load(f)

    if _ARGS_DICT["type"] == "user":
        return sep["train"], sep["test"], sep["total"]
    else:
        return sep["id"], sep["total"]
