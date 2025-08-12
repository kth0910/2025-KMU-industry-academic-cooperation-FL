# server.py
import flwr as fl
from client import client_fn
from flwr.server.strategy import FedAvg
from logger import log_metrics


def run_fl_strategy(strategy_name: str, params: dict, sim_cfg: dict) -> list:
    num_clients = sim_cfg["num_clients"]
    num_rounds = sim_cfg.get("num_rounds", 1000)
    C, E, B = params["clients_per_round"], params["epochs"], params["batch_size"]
    frac = C / num_clients
    lr = params["learning_rates"][0]

    if strategy_name == "FedSGD":
        strategy = FedAvg(
            fraction_fit=frac,
            min_fit_clients=C,
            min_available_clients=C,
            on_fit_config_fn=lambda rnd: {"lr": lr, "epochs": E, "batch_size": B},
        )
    else:
        strategy = FedAvg(
            fraction_fit=frac,
            min_fit_clients=C,
            min_available_clients=C,
            on_fit_config_fn=lambda rnd: {"lr": lr, "epochs": E, "batch_size": B},
        )

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        client_resources={"num_gpus": 1, "num_cpus": 0.1},

        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )

    # metrics 추출 및 CSV 로깅
    dist_acc = history.metrics_distributed.get("accuracy", [])
    dist_loss = history.metrics_distributed.get("loss", [])
    cent_acc = history.metrics_centralized.get("accuracy", [])
    cent_loss = history.metrics_centralized.get("loss", [])

    client_log = f"logs/{strategy_name}_lr{lr}_C{C}_clients.csv"
    global_log = f"logs/{strategy_name}_lr{lr}_global.csv"
    log_metrics(
        distributed={"accuracy": dist_acc, "loss": dist_loss},
        centralized={"accuracy": cent_acc, "loss": cent_loss},
        client_log_path=client_log,
        global_log_path=global_log,
    )

    return cent_acc