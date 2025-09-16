"""
FedMD 공통 유틸리티 함수들
"""
import os
import random
import logging
import time
from typing import Any, Dict, List, Optional, Union
import numpy as np
import torch
import json
from pathlib import Path


def set_seed(seed: int = 42) -> None:
    """시드 설정으로 재현 가능한 실험 환경 구성"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """사용 가능한 디바이스 자동 선택"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def setup_logger(name: str, level: int = logging.INFO, log_file: Optional[str] = "output.log") -> logging.Logger:
    """로거 설정"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 기존 핸들러 제거
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 포맷터 설정
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 파일 핸들러 (기본값: output.log)
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class Timer:
    """간단한 타이머 컨텍스트 매니저"""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        print(f"{self.name} 완료: {duration:.2f}초")


def save_metrics(metrics: Dict[str, Any], filepath: Union[str, Path]) -> None:
    """메트릭을 JSON 파일로 저장"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


def load_metrics(filepath: Union[str, Path]) -> Dict[str, Any]:
    """JSON 파일에서 메트릭 로드"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def calculate_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """정확도 계산"""
    if predictions.dim() > 1:
        predictions = predictions.argmax(dim=1)
    return (predictions == targets).float().mean().item()


def calculate_loss_metrics(predictions: torch.Tensor, targets: torch.Tensor, 
                          criterion: torch.nn.Module) -> Dict[str, float]:
    """손실 및 메트릭 계산"""
    with torch.no_grad():
        loss = criterion(predictions, targets).item()
        accuracy = calculate_accuracy(predictions, targets)
        
        return {
            "loss": loss,
            "accuracy": accuracy
        }


def ensure_dir(path: Union[str, Path]) -> Path:
    """디렉터리가 존재하지 않으면 생성"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_global_indices_file(config: Dict[str, Any]) -> Path:
    """글로벌 인덱스 파일 경로 반환"""
    data_dir = Path(config["dataset"]["public"]["path"])
    index_file = config["dataset"]["public"]["index_file"]
    return data_dir / index_file


def log_round_summary(logger: logging.Logger, round_id: int, 
                     client_metrics: Dict[str, Dict[str, float]]) -> None:
    """라운드 요약 로그 출력"""
    logger.info(f"=== 라운드 {round_id} 완료 ===")
    
    for client_id, metrics in client_metrics.items():
        logger.info(f"클라이언트 {client_id}: "
                   f"손실={metrics.get('loss', 0):.4f}, "
                   f"정확도={metrics.get('accuracy', 0):.4f}")
    
    # 평균 메트릭 계산
    if client_metrics:
        avg_loss = np.mean([m.get('loss', 0) for m in client_metrics.values()])
        avg_acc = np.mean([m.get('accuracy', 0) for m in client_metrics.values()])
        logger.info(f"평균: 손실={avg_loss:.4f}, 정확도={avg_acc:.4f}")


def validate_config(config: Dict[str, Any]) -> List[str]:
    """설정 검증 및 오류 메시지 반환"""
    errors = []
    
    # 필수 키 검증
    required_keys = [
        "dataset.public.name",
        "dataset.public.path", 
        "dataset.public.index_file",
        "model.backbone",
        "model.num_classes",
        "rounds.total",
        "distill.temperature",
        "distill.alpha"
    ]
    
    for key_path in required_keys:
        keys = key_path.split('.')
        current = config
        try:
            for key in keys:
                current = current[key]
        except KeyError:
            errors.append(f"필수 설정 누락: {key_path}")
    
    # 값 범위 검증
    if "distill" in config:
        temp = config["distill"].get("temperature", 1.0)
        if temp <= 0:
            errors.append("distill.temperature는 0보다 커야 합니다")
        
        alpha = config["distill"].get("alpha", 0.5)
        if not 0 <= alpha <= 1:
            errors.append("distill.alpha는 0과 1 사이여야 합니다")
    
    return errors
