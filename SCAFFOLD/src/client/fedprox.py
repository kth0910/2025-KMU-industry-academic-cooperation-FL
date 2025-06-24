from collections import OrderedDict
from typing import OrderedDict, List

import torch
from rich.console import Console

from .base import ClientBase


class FedProxClient(ClientBase):
    def __init__(
        self,
        backbone: torch.nn.Module,
        dataset: str,
        batch_size: int,
        local_epochs: int,
        local_lr: float,
        logger: Console,
        gpu: int,
        mu=1.0,
        data_root=None,            # ★ 추가
    ):
        super(FedProxClient, self).__init__(
            backbone,
            dataset,
            batch_size,
            local_epochs,
            local_lr,
            logger,
            gpu,
            data_root=data_root,            # ★ 추가
        )
        self.trainable_global_params: List[torch.Tensor] = None
        self.mu = mu

    def _train(self):
        self.model.train()
        for _ in range(self.local_epochs):
            x, y = self.get_data_batch()
            logits = self.model(x)
            loss = self.criterion(logits, y)
            self.optimizer.zero_grad()
            loss.backward()
            for w, w_g in zip(self.model.parameters(), self.trainable_global_params):
                w.grad.data += self.mu * (w_g.data - w.data)
            self.optimizer.step()
        return (
            list(self.model.state_dict(keep_vars=True).values()),
            len(self.trainset.dataset),
        )

    def set_parameters(
        self,
        model_params: OrderedDict[str, torch.Tensor],
    ):
        super().set_parameters(model_params)
        self.trainable_global_params = list(
            filter(lambda p: p.requires_grad, model_params.values())
        )
