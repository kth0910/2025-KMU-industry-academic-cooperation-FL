import torch
from rich.console import Console

from .base import ClientBase


class FedAvgClient(ClientBase):
    def __init__(
        self,
        backbone: torch.nn.Module,
        dataset: str,
        batch_size: int,
        local_epochs: int,
        local_lr: float,
        logger: Console,
        gpu: int,
        data_root=None,            # ★ 추가

    ):
        super(FedAvgClient, self).__init__(
            backbone,
            dataset,
            batch_size,
            local_epochs,
            local_lr,
            logger,
            gpu,
            data_root=data_root,            # ★ 추가
        )
