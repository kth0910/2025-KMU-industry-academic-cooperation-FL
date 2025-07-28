import subprocess
import time
import os
import sys

PYTHON = sys.executable

CONFIG_LIST = [
    "config/fedavg_lr0.05.json",
    "config/fedavg_lr0.15.json",
    "config/fedavg_lr0.25.json",
    "config/fedsgd_lr0.45.json",
    "config/fedsgd_lr0.6.json",
    "config/fedsgd_lr0.7.json",
]

CLIENTS_PER_EXPERIMENT = 5

def run_experiment(config_path):
    print(f"\n🚀 Running experiment: {config_path}")

    # 3) 서버 실행
    server_proc = subprocess.Popen(
        [PYTHON, "server.py", "--config", config_path],
        stdout=None,  # 터미널에 바로 출력
        stderr=None,
    )
    time.sleep(1)  # 서버가 시작될 때까지 잠시 대기

    # 4) 클라이언트 여러 개 실행
    client_procs = []
    for cid in range(CLIENTS_PER_EXPERIMENT):
        proc = subprocess.Popen(
            [PYTHON, "client.py",
             "--config", config_path,
             "--cid", str(cid)],
            stdout=None,
            stderr=None,
        )
        client_procs.append(proc)

    # 5) 모든 클라이언트 종료 대기
    for proc in client_procs:
        proc.wait()

    # 6) 서버 종료
    server_proc.terminate()
    try:
        server_proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        server_proc.kill()

    # 그 다음에 포트가 완전히 해제되도록 잠시 추가 대기
    time.sleep(2)

    print(f"✅ Finished: {config_path}")

if __name__ == "__main__":
    # logs 디렉토리 미리 만들어 두기 (선택)
    os.makedirs("logs", exist_ok=True)

    for cfg in CONFIG_LIST:
        # 파일 존재 여부 체크
        if not os.path.isfile(cfg):
            print(f"⚠️  Config not found: {cfg}")
            continue
        run_experiment(cfg)
        # 실험 간 짧은 딜레이
        time.sleep(1)