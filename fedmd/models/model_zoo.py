"""
모델 팩토리 및 등록 시스템
"""
from typing import Dict, Callable, Any
import torch.nn as nn
import logging

from .backbones.cnn_small import CNNSmall
from core.utils import setup_logger


class ModelRegistry:
    """모델 등록 및 관리 클래스"""
    
    def __init__(self):
        self._models: Dict[str, Callable] = {}
        self.logger = setup_logger("ModelRegistry")
        self._register_default_models()
    
    def register(self, name: str, model_fn: Callable) -> None:
        """모델 등록"""
        if name in self._models:
            self.logger.warning(f"모델 '{name}'이 이미 등록되어 있습니다. 덮어씁니다.")
        
        self._models[name] = model_fn
        self.logger.info(f"모델 '{name}' 등록됨")
    
    def get(self, name: str) -> Callable:
        """모델 빌더 함수 반환"""
        if name not in self._models:
            raise ValueError(f"등록되지 않은 모델: {name}. 사용 가능한 모델: {list(self._models.keys())}")
        
        return self._models[name]
    
    def list_models(self) -> list:
        """등록된 모델 목록 반환"""
        return list(self._models.keys())
    
    def _register_default_models(self) -> None:
        """기본 모델들 등록"""
        # CNN Small
        self.register("cnn_small", CNNSmall)
        
        # 추가 모델들은 여기에 등록
        # self.register("resnet18", ResNet18)
        # self.register("mobilenet", MobileNet)
    
    def build_model(self, name: str, **kwargs) -> nn.Module:
        """모델 빌드"""
        model_fn = self.get(name)
        return model_fn(**kwargs)


# 전역 모델 레지스트리
_model_registry = ModelRegistry()


def register_models() -> Dict[str, Callable]:
    """모델 이름 → 빌더 함수 매핑 등록 (하위 호환성)"""
    return {name: _model_registry.get(name) for name in _model_registry.list_models()}


def build_model(name: str, num_classes: int, **kwargs) -> nn.Module:
    """백본 이름으로 분기하여 모델 인스턴스 생성"""
    return _model_registry.build_model(name, num_classes=num_classes, **kwargs)


def register_model(name: str, model_fn: Callable) -> None:
    """새 모델 등록"""
    _model_registry.register(name, model_fn)


def list_available_models() -> list:
    """사용 가능한 모델 목록 반환"""
    return _model_registry.list_models()


def get_model_info(name: str) -> Dict[str, Any]:
    """모델 정보 반환"""
    if name not in _model_registry.list_models():
        raise ValueError(f"등록되지 않은 모델: {name}")
    
    # 기본 정보
    info = {
        "name": name,
        "type": "neural_network",
        "framework": "pytorch"
    }
    
    # 모델별 특수 정보
    if name == "cnn_small":
        info.update({
            "description": "작은 CNN 모델 (CIFAR-10용)",
            "input_size": (3, 32, 32),
            "parameters": "약 1.2M",
            "suitable_for": ["CIFAR-10", "CIFAR-100", "작은 이미지 분류"]
        })
    
    return info


class ModelBuilder:
    """모델 빌더 헬퍼 클래스"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = setup_logger("ModelBuilder")
    
    def build(self, model_name: str = None) -> nn.Module:
        """설정에서 모델 빌드"""
        if model_name is None:
            model_name = self.config.get("backbone", "cnn_small")
        
        num_classes = self.config.get("num_classes", 10)
        
        self.logger.info(f"모델 빌드 중: {model_name} (클래스 수: {num_classes})")
        
        model = build_model(model_name, num_classes=num_classes)
        
        # 모델 정보 로깅
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.logger.info(f"모델 생성 완료: {total_params:,}개 파라미터 (학습 가능: {trainable_params:,}개)")
        
        return model
    
    def get_model_summary(self, model: nn.Module) -> Dict[str, Any]:
        """모델 요약 정보 반환"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "non_trainable_parameters": total_params - trainable_params,
            "model_size_mb": total_params * 4 / (1024 * 1024),  # float32 기준
            "layers": len(list(model.modules()))
        }


def create_model_from_config(config: Dict[str, Any]) -> nn.Module:
    """설정에서 모델 생성 (편의 함수)"""
    builder = ModelBuilder(config)
    return builder.build()


if __name__ == "__main__":
    # 테스트 실행
    print("사용 가능한 모델들:")
    for model_name in list_available_models():
        info = get_model_info(model_name)
        print(f"- {model_name}: {info.get('description', '설명 없음')}")
    
    # 모델 빌드 테스트
    config = {
        "backbone": "cnn_small",
        "num_classes": 10
    }
    
    model = create_model_from_config(config)
    print(f"\n모델 생성 완료: {type(model).__name__}")
    
    # 모델 요약
    builder = ModelBuilder(config)
    summary = builder.get_model_summary(model)
    print(f"파라미터 수: {summary['total_parameters']:,}")
    print(f"모델 크기: {summary['model_size_mb']:.2f} MB")
