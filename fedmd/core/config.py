"""
FedMD 구성 관리 시스템
JSON 파일들을 병합하고 검증하는 기능 제공
"""
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field, validator
import logging

from core.utils import validate_config, setup_logger


class DatasetConfig(BaseModel):
    """데이터셋 설정"""
    name: str = Field(..., description="데이터셋 이름")
    path: str = Field(..., description="데이터셋 경로")
    index_file: str = Field(..., description="인덱스 파일명")
    subset_size: Optional[int] = Field(None, description="서브셋 크기")


class PublicDatasetConfig(DatasetConfig):
    """공용 데이터셋 설정"""
    pass


class PrivateDatasetConfig(BaseModel):
    """개인 데이터셋 설정"""
    root: str = Field(..., description="개인 데이터 루트 경로")
    partition: str = Field(..., description="파티션 방식")


class DatasetSection(BaseModel):
    """데이터셋 섹션"""
    public: PublicDatasetConfig
    private: PrivateDatasetConfig


class ModelConfig(BaseModel):
    """모델 설정"""
    backbone: str = Field(..., description="백본 모델명")
    num_classes: int = Field(..., gt=0, description="클래스 수")


class RoundsConfig(BaseModel):
    """라운드 설정"""
    total: int = Field(..., gt=0, description="총 라운드 수")


class DistillConfig(BaseModel):
    """증류 설정"""
    temperature: float = Field(..., gt=0, description="온도 파라미터")
    alpha: float = Field(..., ge=0, le=1, description="소프트/하드 타겟 비율")
    
    @validator('temperature')
    def validate_temperature(cls, v):
        if v <= 0:
            raise ValueError('temperature는 0보다 커야 합니다')
        return v


class MetricsConfig(BaseModel):
    """메트릭 설정"""
    eval_subset: int = Field(256, gt=0, description="평가 서브셋 크기")


class TrainConfig(BaseModel):
    """학습 설정"""
    local: Dict[str, Any] = Field(..., description="로컬 학습 설정")
    distill: Dict[str, Any] = Field(..., description="증류 학습 설정")


class ClientConfig(BaseModel):
    """클라이언트 설정"""
    id: str = Field(..., description="클라이언트 ID")
    model: str = Field(..., description="모델명")
    weight: float = Field(1.0, gt=0, description="클라이언트 가중치")


class ClientsConfig(BaseModel):
    """클라이언트들 설정"""
    clients: List[ClientConfig] = Field(..., description="클라이언트 목록")


class GrpcConfig(BaseModel):
    """gRPC 설정"""
    address: str = Field(..., description="gRPC 주소")
    use_tls: bool = Field(False, description="TLS 사용 여부")


class CommsConfig(BaseModel):
    """통신 설정"""
    grpc: GrpcConfig
    timeout_sec: int = Field(30, gt=0, description="타임아웃(초)")


class FedMDConfig(BaseModel):
    """FedMD 전체 설정"""
    dataset: DatasetSection
    model: ModelConfig
    rounds: RoundsConfig
    distill: DistillConfig
    metrics: MetricsConfig
    train: Optional[TrainConfig] = None
    clients: Optional[ClientsConfig] = None
    
    @validator('clients', pre=True)
    def validate_clients(cls, v):
        if isinstance(v, list):
            return ClientsConfig(clients=v)
        return v
    comms: Optional[CommsConfig] = None


class ConfigLoader:
    """구성 로더 클래스"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or setup_logger("ConfigLoader")
    
    def load_json(self, filepath: Union[str, Path]) -> Dict[str, Any]:
        """JSON 파일 로드"""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def merge_configs(self, configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """여러 설정을 병합 (나중에 로드된 것이 우선)"""
        merged = {}
        
        for config in configs:
            merged = self._deep_merge(merged, config)
        
        return merged
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """딥 머지 (재귀적)"""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def load_from_files(self, config_files: List[Union[str, Path]]) -> FedMDConfig:
        """여러 설정 파일에서 로드하고 병합"""
        configs = []
        
        for config_file in config_files:
            self.logger.info(f"설정 파일 로드 중: {config_file}")
            config = self.load_json(config_file)
            configs.append(config)
        
        # 병합
        merged_config = self.merge_configs(configs)
        
        # 검증
        errors = validate_config(merged_config)
        if errors:
            error_msg = "설정 검증 실패:\n" + "\n".join(f"- {error}" for error in errors)
            raise ValueError(error_msg)
        
        # Pydantic 모델로 변환
        try:
            return FedMDConfig(**merged_config)
        except Exception as e:
            self.logger.error(f"설정 파싱 실패: {e}")
            raise
    
    def load_from_cli_args(self, args: argparse.Namespace) -> FedMDConfig:
        """CLI 인자에서 설정 로드"""
        config_files = []
        
        # 기본 설정 파일들
        default_files = [
            "configs/default.json",
            "configs/training.json", 
            "configs/clients.json",
            "configs/comms.json"
        ]
        
        # CLI에서 지정된 파일들
        if hasattr(args, 'configs') and args.configs:
            config_files = args.configs.split(',')
        else:
            config_files = default_files
        
        # 파일 존재 확인
        existing_files = []
        for file_path in config_files:
            if Path(file_path).exists():
                existing_files.append(file_path)
            else:
                self.logger.warning(f"설정 파일을 찾을 수 없습니다: {file_path}")
        
        if not existing_files:
            raise FileNotFoundError("유효한 설정 파일이 없습니다")
        
        return self.load_from_files(existing_files)
    
    def save_config(self, config: FedMDConfig, filepath: Union[str, Path]) -> None:
        """설정을 JSON 파일로 저장"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config.dict(), f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"설정 저장됨: {filepath}")


def create_arg_parser() -> argparse.ArgumentParser:
    """CLI 인자 파서 생성"""
    parser = argparse.ArgumentParser(description="FedMD 실험 실행")
    
    parser.add_argument(
        "--configs",
        type=str,
        help="설정 파일들 (쉼표로 구분, 예: config1.json,config2.json)"
    )
    
    parser.add_argument(
        "--client_id",
        type=str,
        help="클라이언트 ID (클라이언트 실행 시 필수)"
    )
    
    parser.add_argument(
        "--rounds",
        type=int,
        help="실행할 라운드 수 (기본값: 설정 파일에서 로드)"
    )
    
    parser.add_argument(
        "--log_level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="로그 레벨"
    )
    
    return parser


def load_config_from_cli() -> FedMDConfig:
    """CLI에서 설정 로드 (편의 함수)"""
    parser = create_arg_parser()
    args = parser.parse_args()
    
    loader = ConfigLoader()
    config = loader.load_from_cli_args(args)
    
    # CLI 인자로 오버라이드
    if args.rounds:
        config.rounds.total = args.rounds
    
    return config, args


if __name__ == "__main__":
    # 테스트 실행
    loader = ConfigLoader()
    config = loader.load_from_files([
        "configs/default.json",
        "configs/training.json",
        "configs/clients.json", 
        "configs/comms.json"
    ])
    
    print("설정 로드 성공!")
    print(f"데이터셋: {config.dataset.public.name}")
    print(f"모델: {config.model.backbone}")
    print(f"라운드: {config.rounds.total}")
    print(f"클라이언트 수: {len(config.clients.clients) if config.clients else 0}")
