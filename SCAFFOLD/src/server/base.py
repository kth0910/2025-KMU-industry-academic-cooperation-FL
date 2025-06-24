import os
import pickle
import random
from argparse import Namespace
from collections import OrderedDict

import torch
from rich.console import Console
from rich.progress import track
from tqdm import tqdm
from pathlib import Path  # ← path 대신 pathlib로 임포트도 변경 필요

_CURRENT_DIR = Path(__file__).parent.resolve()

import sys

sys.path.append(_CURRENT_DIR.parent)

from config.models import LogisticRegression
from config.util import (
    DATA_DIR,
    LOG_DIR,
    PROJECT_DIR,
    TEMP_DIR,
    clone_parameters,
    fix_random_seed,
)

sys.path.append(PROJECT_DIR)
sys.path.append(DATA_DIR)
from client.base import ClientBase
from data.utils.util import get_client_id_indices


class ServerBase:
    def __init__(self, args: Namespace, algo: str):
        self.algo = algo
        self.args = args
        # default log file format
        self.log_name = "{}_{}_{}_{}.html".format(
            self.algo,
            self.args.dataset,
            self.args.global_epochs,
            self.args.local_epochs,
        )
        self.device = torch.device(
            "cuda" if self.args.gpu and torch.cuda.is_available() else "cpu"
        )
        fix_random_seed(self.args.seed)
        self.backbone = LogisticRegression
        self.logger = Console(
            record=True,
            log_path=False,
            log_time=False,
        )
        # base.py 내부
        result = get_client_id_indices(self.args.dataset, data_root=self.args.data_root)

        if self.args.dataset_type == "user":
            self.client_id_indices_train, self.client_id_indices_test, self.client_num_in_total = result
            self.client_id_indices = self.client_id_indices_train  # 기본값으로 train 사용
        else:
            self.client_id_indices, self.client_num_in_total = result

        self.temp_dir = TEMP_DIR / self.algo
        if not os.path.isdir(self.temp_dir):
            os.makedirs(self.temp_dir)

        _dummy_model = self.backbone(self.args.dataset).to(self.device)
        passed_epoch = 0
        self.global_params_dict: OrderedDict[str : torch.Tensor] = None
        if self.args.reset:
            if os.path.exists(self.temp_dir / "global_model.pt"):
                os.remove(self.temp_dir / "global_model.pt")
            if os.path.exists(self.temp_dir / "epoch.pkl"):
                os.remove(self.temp_dir / "epoch.pkl")

        if os.listdir(self.temp_dir) != [] and self.args.save_period > 0:
            if os.path.exists(self.temp_dir / "global_model.pt"):
                self.global_params_dict = torch.load(self.temp_dir / "global_model.pt")
                self.logger.log("Find existed global model...")

            if os.path.exists(self.temp_dir / "epoch.pkl"):
                with open(self.temp_dir / "epoch.pkl", "rb") as f:
                    passed_epoch = pickle.load(f)
                self.logger.log(
                    f"Have run {passed_epoch} epochs already.",
                )
        else:
            self.global_params_dict = OrderedDict(
                _dummy_model.state_dict(keep_vars=True)
            )

        self.global_epochs = self.args.global_epochs - passed_epoch
        self.logger.log("Backbone:", _dummy_model)

        self.trainer: ClientBase = None
        self.num_correct = [[] for _ in range(self.global_epochs)]
        self.num_samples = [[] for _ in range(self.global_epochs)]
        self.global_test_acc = []

    def train(self):
        self.logger.log("=" * 30, "TRAINING", "=" * 30, style="bold green")
        progress_bar = (
            track(
                range(self.global_epochs),
                "[bold green]Training...",
                console=self.logger,
                disable=True,
                transient=True,
            )
            if not self.args.log
            else tqdm(range(self.global_epochs), "Training...")
        )

        for E in progress_bar:

            if E % self.args.verbose_gap == 0:
                self.logger.log("=" * 30, f"ROUND: {E}", "=" * 30)

            selected_clients = random.sample(
                self.client_id_indices, self.args.client_num_per_round
            )
            res_cache = []
            for client_id in selected_clients:
                client_local_params = clone_parameters(self.global_params_dict)
                res, stats = self.trainer.train(
                    client_id=client_id,
                    model_params=client_local_params,
                    verbose=(E % self.args.verbose_gap) == 0,
                )

                res_cache.append(res)
                self.num_correct[E].append(stats["correct"])
                self.num_samples[E].append(stats["size"])
            self.aggregate(res_cache)

            if E % self.args.save_period == 0:
                torch.save(
                    self.global_params_dict,
                    self.temp_dir / "global_model.pt",
                )
                with open(self.temp_dir / "epoch.pkl", "wb") as f:
                    pickle.dump(E, f)
            local_acc = sum(self.num_correct[E]) / sum(self.num_samples[E]) * 100.0
            self.logger.log(f"[TRAIN - Round {E}] Local acc: {local_acc:.2f}%")
            # 글로벌 테스트 후 early stopping 조건 확인
            test_acc = self.test(E)  # E를 넘기려면 test()도 수정 필요
            if test_acc >= 50.0:
                self.logger.log(f"[Early Stop] Reached {test_acc:.2f}% at round {E}", style="bold red")
                break

    @torch.no_grad()
    def aggregate(self, res_cache):
        updated_params_cache = list(zip(*res_cache))[0]
        weights_cache = list(zip(*res_cache))[1]
        weight_sum = sum(weights_cache)
        weights = torch.tensor(weights_cache, device=self.device) / weight_sum

        aggregated_params = []

        for params in zip(*updated_params_cache):
            aggregated_params.append(
                torch.sum(weights * torch.stack(params, dim=-1), dim=-1)
            )

        self.global_params_dict = OrderedDict(
            zip(self.global_params_dict.keys(), aggregated_params)
        )

    def test(self, round_num=None) -> float:
        self.logger.log("=" * 30, "TESTING", "=" * 30, style="bold blue")
        all_loss = []
        all_correct = []
        all_samples = []
        for client_id in track(
            self.client_id_indices,
            "[bold blue]Testing...",
            console=self.logger,
            disable=True,
            transient=True,
        ):
            client_local_params = clone_parameters(self.global_params_dict)
            stats = self.trainer.test(
                client_id=client_id,
                model_params=client_local_params,
            )
            all_loss.append(stats["loss"])
            all_correct.append(stats["correct"])
            all_samples.append(stats["size"])
        
        test_acc = sum(all_correct) / sum(all_samples) * 100.0
        test_loss = sum(all_loss) / sum(all_samples)

        self.logger.log("=" * 20, "RESULTS", "=" * 20, style="bold green")
        self.logger.log(
            f"loss: {test_loss:.4f}    accuracy: {test_acc:.2f}%"
        )



        acc_range = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0]
        logged = set()

        for E, (corr, n) in enumerate(zip(self.num_correct, self.num_samples)):
            total_samples = sum(n)
            if total_samples == 0:
                self.logger.log("⚠️  No test samples available; skipping accuracy computation.")
                return 0.0
            avg_acc = sum(corr) / total_samples * 100.0
            
            # 매 라운드 출력
            self.logger.log(f"[Round {E}] {self.algo} accuracy: {avg_acc:.2f}%")
            
            # milestone 달성 시 출력
            for acc in acc_range:
                if acc not in logged and avg_acc >= acc:
                    self.logger.log(
                        f"{self.algo} achieved {acc}% accuracy ({avg_acc:.2f}%) at round: {E}"
                    )
                    logged.add(acc)
        if round_num is not None:
            self.global_test_acc.append(test_acc)
            self.logger.log(f"[TEST  - Round {round_num}] Global acc: {test_acc:.2f}%")
            
        return test_acc
            
    def run(self):
        self.logger.log("Arguments:", dict(self.args._get_kwargs()))
        self.train()
        self.test()
        if self.args.log:
            if not os.path.isdir(LOG_DIR):
                os.mkdir(LOG_DIR)
            self.logger.save_html(LOG_DIR / self.log_name)

        # delete all temporary files
        if os.listdir(self.temp_dir) != []:
            os.system(f"rm -rf {self.temp_dir}")
