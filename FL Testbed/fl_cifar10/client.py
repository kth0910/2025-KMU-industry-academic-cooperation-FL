# client.py

from parse_config import load_config
import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from model import CNN
from utils import load_datasets
import os
import csv

# 1) 설정 로드
cfg = load_config()
strategy_name = cfg["strategy"].lower()
lr_init = cfg["lr_init"]
lr_decay = cfg["lr_decay"]
epochs = cfg["epochs"]
batch_size = cfg["batch_size"]
num_clients = cfg["clients"]
cid = cfg["cid"]

# 2) 로깅 파일 설정 (전략+lr+client id 별)
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
local_log_path = f"{log_dir}/client_{strategy_name}_lr{lr_init}_cid{cid}.csv"
if not os.path.exists(local_log_path):
    with open(local_log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "loss", "accuracy"])

# 3) 글로벌 라운드 카운터
global_round = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 4) Flower Client 구현
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cfg):
        self.cfg = cfg
        # 모델 초기화
        self.model = CNN().to(device)
        # 데이터 로드 및 분할
        trainset, testset = load_datasets()
        n = len(trainset) // self.cfg["clients"]
        start, end = self.cfg["cid"] * n, (self.cfg["cid"] + 1) * n
        local_subset = Subset(trainset, list(range(start, end)))
        self.trainloader = DataLoader(local_subset, batch_size=self.cfg["batch_size"], shuffle=True)
        self.testloader = DataLoader(testset, batch_size=self.cfg["batch_size"], shuffle=False)
        # 손실함수
        self.criterion = nn.CrossEntropyLoss()

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        global global_round
        global_round += 1

        # learning rate 계산
        lr = self.cfg["lr_init"] * (self.cfg["lr_decay"] ** global_round)
        # 모델 파라미터 업데이트
        self.set_parameters(parameters)
        self.model.train()
        optimizer = optim.SGD(self.model.parameters(), lr=lr)

        # local training
        for _ in range(self.cfg["epochs"]):
            for x, y in self.trainloader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                preds = self.model(x)
                loss = self.criterion(preds, y)
                loss.backward()
                optimizer.step()

        return self.get_parameters(config={}), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        # 파라미터 로드
        self.set_parameters(parameters)
        self.model.eval()

        correct, total, loss_sum = 0, 0, 0.0
        with torch.no_grad():
            for x, y in self.testloader:
                x, y = x.to(device), y.to(device)
                preds = self.model(x)
                loss = self.criterion(preds, y)
                loss_sum += loss.item() * x.size(0)
                correct += (preds.argmax(1) == y).sum().item()
                total += y.size(0)

        avg_loss = loss_sum / total
        acc = correct / total

        # ✅ 로컬 로그에 기록
        with open(local_log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([global_round, avg_loss, acc])

        return float(avg_loss), total, {"accuracy": acc}

if __name__ == "__main__":
    # Flower 클라이언트 시작

    print("Using device:", device, "for client", cid)
    client = FlowerClient(cfg)
    fl.client.start_numpy_client(server_address="localhost:8080", client=client)
