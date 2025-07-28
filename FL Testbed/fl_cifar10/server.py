from parse_config import load_config
import flwr as fl
import torch
import torch.nn as nn
from model import CNN
from utils import load_datasets
import os
import csv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_evaluate_fn(cfg):
    """서버에서 global evaluation을 수행하고 전략+lr 별 로그를 저장하는 함수."""
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    _, testset = load_datasets()
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)

    strategy_name = cfg["strategy"]
    lr_init = cfg.get("lr_init", "na")
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = f"{log_dir}/global_{strategy_name}_lr{lr_init}.csv"

    # 헤더 쓰기 (파일이 없을 때만)
    if not os.path.exists(log_path):
        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["round", "loss", "accuracy"])

    def evaluate(server_round, parameters, config):
        # 모델에 파라미터 적용
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        torch.save(model.state_dict(), f"checkpoints/global_round{server_round}.pt")
        # 테스트셋 평가
        correct, total, loss_sum = 0, 0, 0.0
        with torch.no_grad():
            for x, y in testloader:
                x, y = x.to(device), y.to(device)
                output = model(x)
                loss = criterion(output, y)
                loss_sum += loss.item()
                preds = output.argmax(1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        acc = correct / total
        avg_loss = loss_sum / len(testloader)

        # 로그 파일에 기록
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([server_round, avg_loss, acc])

        return avg_loss, {"accuracy": acc}

    return evaluate

def main():
    # 1) 설정 로드
    cfg = load_config()
    strategy_name = cfg["strategy"].lower()
    num_clients = cfg["clients"]

    # 2) 전략 선택 & evaluate_fn 연결
    if strategy_name == "fedavg":
        strategy = fl.server.strategy.FedAvg(
            evaluate_fn=get_evaluate_fn(cfg),
            fraction_fit=1.0,
            min_fit_clients=num_clients,
            min_available_clients=num_clients,
        )
    elif strategy_name == "fedsgd":
        # Flower에 기본 FedSGD 전략이 없으므로, E=1인 FedAvg로 대체
        strategy = fl.server.strategy.FedAvg(
            evaluate_fn=get_evaluate_fn(cfg),
            fraction_fit=1.0,
            min_fit_clients=num_clients,
            min_available_clients=num_clients,
        )
    else:
        raise ValueError(f"지원하지 않는 strategy: {strategy_name}")

    # 3) 서버 시작
    fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=cfg["rounds"]),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
