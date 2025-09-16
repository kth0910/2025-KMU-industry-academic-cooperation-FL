"""
로짓 앙상블 및 정규화
"""
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Union, Any
import numpy as np
import logging
from tqdm import tqdm

from core.utils import setup_logger


class LogitsAggregator:
    """
    로짓 앙상블러
    
    여러 클라이언트의 로짓을 수집하여 소프트 타겟을 생성
    """
    
    def __init__(self, 
                 method: str = "mean",
                 temperature: float = 1.0,
                 eps: float = 1e-12,
                 use_softmax: bool = True):
        """
        Args:
            method: 앙상블 방법 ("mean", "weighted_mean", "median", "max")
            temperature: 온도 파라미터 (소프트맥스 전)
            eps: 수치 안정성을 위한 작은 값
            use_softmax: 소프트맥스 적용 여부
        """
        self.method = method
        self.temperature = temperature
        self.eps = eps
        self.use_softmax = use_softmax
        
        self.logger = setup_logger("LogitsAggregator")
        
        # 앙상블 방법 매핑
        self.aggregation_methods = {
            "mean": self._mean_aggregation,
            "weighted_mean": self._weighted_mean_aggregation,
            "median": self._median_aggregation,
            "max": self._max_aggregation,
            "log_mean": self._log_mean_aggregation
        }
        
        if method not in self.aggregation_methods:
            raise ValueError(f"지원하지 않는 앙상블 방법: {method}. "
                           f"사용 가능한 방법: {list(self.aggregation_methods.keys())}")
    
    def aggregate(self, 
                 client_logits: Dict[str, torch.Tensor],
                 client_weights: Optional[Dict[str, float]] = None) -> torch.Tensor:
        """
        클라이언트 로짓들을 앙상블하여 소프트 타겟 생성
        
        Args:
            client_logits: {client_id: logits_tensor} 형태의 딕셔너리
            client_weights: {client_id: weight} 형태의 가중치 딕셔너리 (선택사항)
            
        Returns:
            앙상블된 소프트 타겟 확률 분포 (N, num_classes)
        """
        if not client_logits:
            raise ValueError("클라이언트 로짓이 비어있습니다")
        
        # 로짓 텐서들을 리스트로 변환
        logits_list = list(client_logits.values())
        client_ids = list(client_logits.keys())
        
        # 모든 로짓이 같은 형태인지 확인
        shapes = [logits.shape for logits in logits_list]
        if not all(shape == shapes[0] for shape in shapes):
            raise ValueError(f"로짓 텐서들의 형태가 일치하지 않습니다: {shapes}")
        
        self.logger.debug(f"앙상블 시작: {len(logits_list)}개 클라이언트, "
                         f"로짓 형태: {shapes[0]}")
        
        # 앙상블 수행 과정 시각화
        pbar_aggregate = tqdm(total=1, desc="로짓 앙상블 수행", leave=False)
        pbar_aggregate.set_postfix({"방법": self.method, "클라이언트 수": len(logits_list)})
        
        aggregated_logits = self.aggregation_methods[self.method](
            logits_list, client_weights, client_ids
        )
        
        pbar_aggregate.update(1)
        pbar_aggregate.close()
        
        # 소프트맥스 적용 (옵션)
        if self.use_softmax:
            if self.temperature != 1.0:
                aggregated_logits = aggregated_logits / self.temperature
            soft_targets = F.softmax(aggregated_logits, dim=1)
        else:
            soft_targets = aggregated_logits
        
        # 정규화 및 안정화
        soft_targets = self.sanitize(soft_targets)
        
        self.logger.debug(f"앙상블 완료: 소프트 타겟 형태 {soft_targets.shape}")
        
        return soft_targets
    
    def _mean_aggregation(self, 
                         logits_list: List[torch.Tensor],
                         client_weights: Optional[Dict[str, float]],
                         client_ids: List[str]) -> torch.Tensor:
        """평균 앙상블"""
        if client_weights is not None:
            # 가중 평균
            weights = torch.tensor([client_weights.get(cid, 1.0) for cid in client_ids])
            weights = weights / weights.sum()  # 정규화
            
            # 가중합
            aggregated = torch.zeros_like(logits_list[0])
            for i, logits in enumerate(logits_list):
                aggregated += weights[i] * logits
        else:
            # 단순 평균
            aggregated = torch.stack(logits_list).mean(dim=0)
        
        return aggregated
    
    def _weighted_mean_aggregation(self, 
                                  logits_list: List[torch.Tensor],
                                  client_weights: Optional[Dict[str, float]],
                                  client_ids: List[str]) -> torch.Tensor:
        """가중 평균 앙상블 (명시적)"""
        if client_weights is None:
            # 가중치가 없으면 균등 가중치 사용
            client_weights = {cid: 1.0 for cid in client_ids}
        
        weights = torch.tensor([client_weights.get(cid, 1.0) for cid in client_ids])
        weights = weights / weights.sum()  # 정규화
        
        aggregated = torch.zeros_like(logits_list[0])
        for i, logits in enumerate(logits_list):
            aggregated += weights[i] * logits
        
        return aggregated
    
    def _median_aggregation(self, 
                           logits_list: List[torch.Tensor],
                           client_weights: Optional[Dict[str, float]],
                           client_ids: List[str]) -> torch.Tensor:
        """중앙값 앙상블"""
        stacked_logits = torch.stack(logits_list)
        return torch.median(stacked_logits, dim=0)[0]
    
    def _max_aggregation(self, 
                        logits_list: List[torch.Tensor],
                        client_weights: Optional[Dict[str, float]],
                        client_ids: List[str]) -> torch.Tensor:
        """최대값 앙상블"""
        stacked_logits = torch.stack(logits_list)
        return torch.max(stacked_logits, dim=0)[0]
    
    def _log_mean_aggregation(self, 
                             logits_list: List[torch.Tensor],
                             client_weights: Optional[Dict[str, float]],
                             client_ids: List[str]) -> torch.Tensor:
        """로그 공간에서의 평균 앙상블"""
        # 로그 확률로 변환
        log_probs_list = [F.log_softmax(logits, dim=1) for logits in logits_list]
        
        if client_weights is not None:
            # 가중 평균
            weights = torch.tensor([client_weights.get(cid, 1.0) for cid in client_ids])
            weights = weights / weights.sum()
            
            aggregated = torch.zeros_like(log_probs_list[0])
            for i, log_probs in enumerate(log_probs_list):
                aggregated += weights[i] * log_probs
        else:
            # 단순 평균
            aggregated = torch.stack(log_probs_list).mean(dim=0)
        
        return aggregated
    
    def sanitize(self, probs: torch.Tensor) -> torch.Tensor:
        """
        확률 분포 정규화 및 안정화
        
        Args:
            probs: 확률 분포 텐서
            
        Returns:
            정규화된 확률 분포
        """
        # NaN/Inf 체크
        if torch.isnan(probs).any():
            self.logger.warning("NaN 값이 발견되어 0으로 대체합니다")
            probs = torch.where(torch.isnan(probs), torch.zeros_like(probs), probs)
        
        if torch.isinf(probs).any():
            self.logger.warning("Inf 값이 발견되어 0으로 대체합니다")
            probs = torch.where(torch.isinf(probs), torch.zeros_like(probs), probs)
        
        # 음수 값 클리핑
        probs = torch.clamp(probs, min=self.eps)
        
        # 정규화 (합이 1이 되도록)
        probs = probs / probs.sum(dim=1, keepdim=True)
        
        # 라벨 스무딩 (옵션)
        if hasattr(self, 'label_smoothing') and self.label_smoothing > 0:
            num_classes = probs.size(1)
            uniform_dist = torch.ones_like(probs) / num_classes
            probs = (1 - self.label_smoothing) * probs + self.label_smoothing * uniform_dist
        
        return probs
    
    def set_label_smoothing(self, smoothing: float) -> None:
        """라벨 스무딩 설정"""
        if not 0 <= smoothing <= 1:
            raise ValueError("라벨 스무딩은 0과 1 사이여야 합니다")
        
        self.label_smoothing = smoothing
        self.logger.info(f"라벨 스무딩 설정: {smoothing}")
    
    def get_aggregation_info(self) -> Dict[str, Any]:
        """앙상블 정보 반환"""
        return {
            "method": self.method,
            "temperature": self.temperature,
            "eps": self.eps,
            "use_softmax": self.use_softmax,
            "label_smoothing": getattr(self, 'label_smoothing', 0.0)
        }


def create_aggregator(config: Dict[str, Any]) -> LogitsAggregator:
    """설정에서 앙상블러 생성"""
    distill_config = config.get("distill", {})
    
    return LogitsAggregator(
        method=distill_config.get("aggregation_method", "mean"),
        temperature=distill_config.get("temperature", 1.0),
        eps=distill_config.get("eps", 1e-12),
        use_softmax=distill_config.get("use_softmax", True)
    )


class EnsembleValidator:
    """앙상블 결과 검증 클래스"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or setup_logger("EnsembleValidator")
    
    def validate_logits(self, 
                       client_logits: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """로짓 검증"""
        validation_results = {
            "valid": True,
            "warnings": [],
            "errors": []
        }
        
        if not client_logits:
            validation_results["valid"] = False
            validation_results["errors"].append("클라이언트 로짓이 비어있습니다")
            return validation_results
        
        # 형태 일치 확인
        shapes = [logits.shape for logits in client_logits.values()]
        if not all(shape == shapes[0] for shape in shapes):
            validation_results["valid"] = False
            validation_results["errors"].append(f"로짓 형태 불일치: {shapes}")
        
        # NaN/Inf 확인
        for client_id, logits in client_logits.items():
            if torch.isnan(logits).any():
                validation_results["warnings"].append(f"클라이언트 {client_id}: NaN 값 발견")
            
            if torch.isinf(logits).any():
                validation_results["warnings"].append(f"클라이언트 {client_id}: Inf 값 발견")
        
        # 분산 확인
        if len(client_logits) > 1:
            logits_tensor = torch.stack(list(client_logits.values()))
            variance = torch.var(logits_tensor, dim=0).mean()
            
            if variance < 1e-6:
                validation_results["warnings"].append("클라이언트 로짓의 분산이 매우 낮습니다")
        
        return validation_results
    
    def validate_soft_targets(self, soft_targets: torch.Tensor) -> Dict[str, Any]:
        """소프트 타겟 검증"""
        validation_results = {
            "valid": True,
            "warnings": [],
            "errors": []
        }
        
        # 확률 분포 확인
        if not torch.allclose(soft_targets.sum(dim=1), torch.ones(soft_targets.size(0))):
            validation_results["warnings"].append("확률 분포의 합이 1이 아닙니다")
        
        # 음수 값 확인
        if (soft_targets < 0).any():
            validation_results["valid"] = False
            validation_results["errors"].append("음수 확률 값이 발견되었습니다")
        
        # NaN/Inf 확인
        if torch.isnan(soft_targets).any():
            validation_results["valid"] = False
            validation_results["errors"].append("NaN 값이 발견되었습니다")
        
        if torch.isinf(soft_targets).any():
            validation_results["valid"] = False
            validation_results["errors"].append("Inf 값이 발견되었습니다")
        
        return validation_results


if __name__ == "__main__":
    # 테스트 실행
    print("=== 로짓 앙상블러 테스트 ===")
    
    # 더미 데이터 생성
    batch_size = 4
    num_classes = 10
    num_clients = 3
    
    client_logits = {}
    for i in range(num_clients):
        client_id = f"client_{i}"
        logits = torch.randn(batch_size, num_classes)
        client_logits[client_id] = logits
    
    # 클라이언트 가중치
    client_weights = {
        "client_0": 1.0,
        "client_1": 0.8,
        "client_2": 1.2
    }
    
    # 다양한 앙상블 방법 테스트
    methods = ["mean", "weighted_mean", "median", "max", "log_mean"]
    
    for method in methods:
        aggregator = LogitsAggregator(method=method, temperature=2.0)
        soft_targets = aggregator.aggregate(client_logits, client_weights)
        
        print(f"{method}: 형태 {soft_targets.shape}, "
              f"합 {soft_targets.sum(dim=1).mean().item():.4f}")
    
    # 검증 테스트
    validator = EnsembleValidator()
    logits_validation = validator.validate_logits(client_logits)
    targets_validation = validator.validate_soft_targets(soft_targets)
    
    print(f"\n로짓 검증: {logits_validation}")
    print(f"소프트 타겟 검증: {targets_validation}")
    
    # 앙상블러 정보
    print(f"\n앙상블러 정보: {aggregator.get_aggregation_info()}")
