from .base import ServerBase
from client.fedprox import FedProxClient
from config.util import get_args
import sys
import os
import json
import argparse
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

class FedProxServer(ServerBase):
    def __init__(self):
        super(FedProxServer, self).__init__(get_args(), "FedProx")

        self.trainer = FedProxClient(
            backbone=self.backbone(self.args.dataset),
            dataset=self.args.dataset,
            batch_size=self.args.batch_size,
            local_epochs=self.args.local_epochs,
            local_lr=self.args.local_lr,
            logger=self.logger,
            gpu=self.args.gpu,
            mu=self.args.mu,  # 👈 추가,
            data_root= self.args.data_root,   # ★ 추가

        )


if __name__ == "__main__":
    server = FedProxServer()
    server.run()
