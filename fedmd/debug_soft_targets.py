#!/usr/bin/env python3
"""
소프트 타겟 생성 디버그 스크립트
"""
import sys
import time
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.config import load_config_from_cli, create_arg_parser
from core.server import FedMDServer
from core.client import ClientNode
from core.utils import setup_logger, set_seed

def debug_soft_targets():
    """소프트 타겟 생성 디버그"""
    print("🔍 소프트 타겟 생성 디버그 시작")
    
    # 시드 설정
    set_seed(42)
    
    # 로거 설정
    logger = setup_logger("DebugSoftTargets", log_file="debug_soft_targets.log")
    
    # 테스트 설정
    config = {
        "dataset": {
            "public": {
                "name": "test_public",
                "path": "data/public",
                "index_file": "public_indices.json"
            },
            "private": {
                "root": "data/private",
                "partition": "iid"
            }
        },
        "model": {
            "backbone": "cnn_small",
            "num_classes": 10
        },
        "rounds": {
            "total": 1,
            "public_subset_size": 100
        },
        "distill": {
            "temperature": 3.0,
            "alpha": 0.7
        },
        "train": {
            "local_epochs": 1,
            "distill_epochs": 1,
            "batch_size": 32,
            "lr": 0.001
        },
        "clients": [
            {"id": "clientA", "model": "cnn_small", "weight": 1.0},
            {"id": "clientB", "model": "cnn_small", "weight": 1.0}
        ],
        "comms": {
            "grpc": {
                "address": "localhost:50051",
                "timeout_sec": 30
            }
        }
    }
    
    # 서버 초기화
    logger.info("서버 초기화 중...")
    server = FedMDServer(config, logger)
    
    # 클라이언트 초기화
    logger.info("클라이언트 초기화 중...")
    client_a = ClientNode("clientA", config, logger=logger)
    client_b = ClientNode("clientB", config, logger=logger)
    
    # 클라이언트들이 실제 서버 인스턴스를 사용하도록 설정
    client_a.mock_server = server
    client_b.mock_server = server
    
    # 클라이언트 등록
    logger.info("클라이언트 등록 중...")
    client_a.register()
    client_b.register()
    
    # 데이터셋 로드
    logger.info("데이터셋 로드 중...")
    client_a.load_datasets()
    client_b.load_datasets()
    
    # 모델 빌드
    logger.info("모델 빌드 중...")
    client_a.build_model()
    client_b.build_model()
    
    # 라운드 1 시작
    logger.info("라운드 1 시작...")
    server.start_round(1)
    
    # 클라이언트 라운드 번호 동기화
    client_a.current_round = 1
    client_b.current_round = 1
    # mock_server는 이미 server 인스턴스이므로 별도 설정 불필요
    
    # 클라이언트 A 로컬 학습 및 로짓 업로드
    logger.info("클라이언트 A 로컬 학습 및 로짓 업로드...")
    client_a.local_pretrain()
    logits_a = client_a.infer_public_logits()
    logger.info(f"클라이언트 A 로짓 수: {len(logits_a)}")
    client_a.upload_logits(logits_a)
    
    # 클라이언트 B 로컬 학습 및 로짓 업로드
    logger.info("클라이언트 B 로컬 학습 및 로짓 업로드...")
    client_b.local_pretrain()
    logits_b = client_b.infer_public_logits()
    logger.info(f"클라이언트 B 로짓 수: {len(logits_b)}")
    client_b.upload_logits(logits_b)
    
    # 서버 상태 확인
    logger.info("서버 상태 확인...")
    logger.info(f"서버 현재 라운드: {server.current_round}")
    logger.info(f"서버 라운드 상태: {server.round_status}")
    logger.info(f"수집된 로짓 수: {len(server.collected_logits.get(1, {}))}")
    logger.info(f"예상 클라이언트 수: {server.expected_clients}")
    
    # 소프트 타겟 생성 시도
    logger.info("소프트 타겟 생성 시도...")
    try:
        result = server.make_soft_targets()
        logger.info(f"소프트 타겟 생성 결과: {result}")
    except Exception as e:
        logger.error(f"소프트 타겟 생성 중 오류: {e}")
        import traceback
        traceback.print_exc()
    
    logger.info("디버그 완료")

if __name__ == "__main__":
    debug_soft_targets()
