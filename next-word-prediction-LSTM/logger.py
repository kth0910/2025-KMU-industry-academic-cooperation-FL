# logger.py
import os
import csv


def validate_log_paths(*paths):
    """
    로그 파일을 쓰기 전에 디렉토리가 존재하는지 확인하고, 없으면 생성합니다.
    """
    for path in paths:
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)


def log_metrics(distributed: dict, centralized: dict,
                client_log_path: str, global_log_path: str):
    # 로그 디렉토리 생성
    validate_log_paths(client_log_path, global_log_path)

    # Client-level logs
    with open(client_log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "client_index", "accuracy", "loss"])
        for r, (accs, losss) in enumerate(zip(
                distributed.get("accuracy", []), distributed.get("loss", []))):
            for idx, (a, l) in enumerate(zip(accs, losss)):
                writer.writerow([r, idx, a, l])

    # Global logs
    with open(global_log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "accuracy", "loss"])
        for r, (a, l) in enumerate(zip(
                centralized.get("accuracy", []), centralized.get("loss", []))):
            writer.writerow([r, a, l])
