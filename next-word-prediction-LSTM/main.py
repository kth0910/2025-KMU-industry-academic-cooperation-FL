# main.py
import json
from config import load_hyperparams
from server import run_fl_strategy


def main():
    # 하이퍼파라미터 및 시뮬레이션 설정 로드
    cfg = load_hyperparams()
    sim_cfg = cfg["simulation"]
    results = {}

    # 디렉토리 검증: 데이터셋과 로그를 위한 기본 폴더 생성
    import dataset
    dataset.validate_dataset()

    # FedSGD, FedAvg_E1 각각 실험 실행
    for strategy_name in ["FedSGD", "FedAvg_E1"]:
        params = cfg[strategy_name]
        history = {}
        for lr in params["learning_rates"]:
            params_lr = params.copy()
            params_lr["learning_rates"] = [lr]
            acc_list = run_fl_strategy(strategy_name, params_lr, sim_cfg)
            history[lr] = acc_list
        results[strategy_name] = history

    # 전체 결과 저장
    with open("all_results.json", "w") as fp:
        json.dump(results, fp)
    print("Experiments complete. Results saved to all_results.json.")


if __name__ == "__main__":
    main()