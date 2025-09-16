"""
지식 증류 손실 함수들
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
import logging

from core.utils import setup_logger


def kd_kl_div(student_logits: torch.Tensor, 
              teacher_probs: torch.Tensor, 
              T: float = 3.0) -> torch.Tensor:
    """
    KL Divergence 기반 지식 증류 손실
    
    Args:
        student_logits: 학생 모델의 로짓 (B, num_classes)
        teacher_probs: 교사 모델의 확률 분포 (B, num_classes)
        T: 온도 파라미터
        
    Returns:
        KL Divergence 손실
    """
    # 학생 모델의 소프트맥스 확률 계산
    student_probs = F.softmax(student_logits / T, dim=1)
    
    # 교사 모델의 확률을 로그 공간으로 변환
    teacher_log_probs = F.log_softmax(teacher_probs / T, dim=1)
    
    # KL Divergence 계산
    kl_loss = F.kl_div(teacher_log_probs, student_probs, reduction='batchmean')
    
    # 온도 제곱으로 스케일링 (원래 스케일로 복원)
    return kl_loss * (T ** 2)


def kd_ce_with_soft_targets(student_logits: torch.Tensor,
                           soft_targets: torch.Tensor,
                           T: float = 3.0,
                           alpha: float = 0.7,
                           hard_targets: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    하드 타겟과 소프트 타겟의 가중합 손실
    
    Args:
        student_logits: 학생 모델의 로짓 (B, num_classes)
        soft_targets: 소프트 타겟 확률 분포 (B, num_classes)
        T: 온도 파라미터
        alpha: 소프트 타겟 가중치 (0~1)
        hard_targets: 하드 타겟 (B,) - 선택사항
        
    Returns:
        가중합 손실
    """
    # 소프트 타겟 손실 (KL Divergence)
    soft_loss = kd_kl_div(student_logits, soft_targets, T)
    
    # 하드 타겟 손실 (Cross Entropy)
    if hard_targets is not None:
        hard_loss = F.cross_entropy(student_logits, hard_targets)
        return alpha * soft_loss + (1 - alpha) * hard_loss
    else:
        return soft_loss


def kd_mse_loss(student_logits: torch.Tensor,
                teacher_logits: torch.Tensor,
                T: float = 3.0) -> torch.Tensor:
    """
    MSE 기반 지식 증류 손실
    
    Args:
        student_logits: 학생 모델의 로짓 (B, num_classes)
        teacher_logits: 교사 모델의 로짓 (B, num_classes)
        T: 온도 파라미터
        
    Returns:
        MSE 손실
    """
    # 온도 스케일링
    student_scaled = student_logits / T
    teacher_scaled = teacher_logits / T
    
    # MSE 손실
    mse_loss = F.mse_loss(student_scaled, teacher_scaled)
    
    return mse_loss * (T ** 2)


def kd_js_div_loss(student_logits: torch.Tensor,
                   teacher_probs: torch.Tensor,
                   T: float = 3.0) -> torch.Tensor:
    """
    Jensen-Shannon Divergence 기반 지식 증류 손실
    
    Args:
        student_logits: 학생 모델의 로짓 (B, num_classes)
        teacher_probs: 교사 모델의 확률 분포 (B, num_classes)
        T: 온도 파라미터
        
    Returns:
        JS Divergence 손실
    """
    # 온도 스케일링된 확률 분포
    student_probs = F.softmax(student_logits / T, dim=1)
    teacher_probs_scaled = F.softmax(teacher_probs / T, dim=1)
    
    # 평균 분포
    mean_probs = 0.5 * (student_probs + teacher_probs_scaled)
    
    # KL Divergence 계산
    kl_student = F.kl_div(F.log_softmax(student_logits / T, dim=1), 
                         mean_probs, reduction='batchmean')
    kl_teacher = F.kl_div(F.log_softmax(teacher_probs / T, dim=1), 
                         mean_probs, reduction='batchmean')
    
    # JS Divergence
    js_loss = 0.5 * (kl_student + kl_teacher)
    
    return js_loss * (T ** 2)


class KnowledgeDistillationLoss(nn.Module):
    """
    지식 증류 손실 클래스
    """
    
    def __init__(self, 
                 method: str = "kl_div",
                 temperature: float = 3.0,
                 alpha: float = 0.7,
                 use_hard_targets: bool = True):
        """
        Args:
            method: 손실 함수 방법 ("kl_div", "mse", "js_div")
            temperature: 온도 파라미터
            alpha: 소프트/하드 타겟 가중치
            use_hard_targets: 하드 타겟 사용 여부
        """
        super(KnowledgeDistillationLoss, self).__init__()
        
        self.method = method
        self.temperature = temperature
        self.alpha = alpha
        self.use_hard_targets = use_hard_targets
        
        self.logger = setup_logger("KnowledgeDistillationLoss")
        
        # 손실 함수 매핑
        self.loss_functions = {
            "kl_div": self._kl_div_loss,
            "mse": self._mse_loss,
            "js_div": self._js_div_loss
        }
        
        if method not in self.loss_functions:
            raise ValueError(f"지원하지 않는 방법: {method}. "
                           f"사용 가능한 방법: {list(self.loss_functions.keys())}")
    
    def forward(self, 
                student_logits: torch.Tensor,
                teacher_probs: torch.Tensor,
                hard_targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            student_logits: 학생 모델의 로짓
            teacher_probs: 교사 모델의 확률 분포
            hard_targets: 하드 타겟 (선택사항)
            
        Returns:
            지식 증류 손실
        """
        return self.loss_functions[self.method](
            student_logits, teacher_probs, hard_targets
        )
    
    def _kl_div_loss(self, 
                    student_logits: torch.Tensor,
                    teacher_probs: torch.Tensor,
                    hard_targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        """KL Divergence 손실"""
        if self.use_hard_targets and hard_targets is not None:
            return kd_ce_with_soft_targets(
                student_logits, teacher_probs, 
                self.temperature, self.alpha, hard_targets
            )
        else:
            return kd_kl_div(student_logits, teacher_probs, self.temperature)
    
    def _mse_loss(self, 
                 student_logits: torch.Tensor,
                 teacher_probs: torch.Tensor,
                 hard_targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        """MSE 손실"""
        if self.use_hard_targets and hard_targets is not None:
            # 하드 타겟을 원-핫 인코딩으로 변환
            num_classes = student_logits.size(1)
            hard_targets_one_hot = F.one_hot(hard_targets, num_classes).float()
            
            # 소프트 타겟과 하드 타겟의 가중합
            mixed_targets = self.alpha * teacher_probs + (1 - self.alpha) * hard_targets_one_hot
            
            return kd_mse_loss(student_logits, mixed_targets, self.temperature)
        else:
            return kd_mse_loss(student_logits, teacher_probs, self.temperature)
    
    def _js_div_loss(self, 
                    student_logits: torch.Tensor,
                    teacher_probs: torch.Tensor,
                    hard_targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        """JS Divergence 손실"""
        if self.use_hard_targets and hard_targets is not None:
            # 하드 타겟을 원-핫 인코딩으로 변환
            num_classes = student_logits.size(1)
            hard_targets_one_hot = F.one_hot(hard_targets, num_classes).float()
            
            # 소프트 타겟과 하드 타겟의 가중합
            mixed_targets = self.alpha * teacher_probs + (1 - self.alpha) * hard_targets_one_hot
            
            return kd_js_div_loss(student_logits, mixed_targets, self.temperature)
        else:
            return kd_js_div_loss(student_logits, teacher_probs, self.temperature)
    
    def get_loss_info(self) -> Dict[str, Any]:
        """손실 함수 정보 반환"""
        return {
            "method": self.method,
            "temperature": self.temperature,
            "alpha": self.alpha,
            "use_hard_targets": self.use_hard_targets
        }


def create_kd_loss(config: Dict[str, Any]) -> KnowledgeDistillationLoss:
    """설정에서 지식 증류 손실 함수 생성"""
    distill_config = config.get("distill", {})
    
    return KnowledgeDistillationLoss(
        method=distill_config.get("method", "kl_div"),
        temperature=distill_config.get("temperature", 3.0),
        alpha=distill_config.get("alpha", 0.7),
        use_hard_targets=distill_config.get("use_hard_targets", True)
    )


if __name__ == "__main__":
    # 테스트 실행
    batch_size = 4
    num_classes = 10
    
    # 더미 데이터 생성
    student_logits = torch.randn(batch_size, num_classes)
    teacher_probs = F.softmax(torch.randn(batch_size, num_classes), dim=1)
    hard_targets = torch.randint(0, num_classes, (batch_size,))
    
    print("=== 지식 증류 손실 함수 테스트 ===")
    
    # KL Divergence 손실
    kl_loss = kd_kl_div(student_logits, teacher_probs, T=3.0)
    print(f"KL Divergence 손실: {kl_loss.item():.4f}")
    
    # 가중합 손실
    weighted_loss = kd_ce_with_soft_targets(
        student_logits, teacher_probs, T=3.0, alpha=0.7, hard_targets=hard_targets
    )
    print(f"가중합 손실: {weighted_loss.item():.4f}")
    
    # MSE 손실
    mse_loss = kd_mse_loss(student_logits, teacher_logits, T=3.0)
    print(f"MSE 손실: {mse_loss.item():.4f}")
    
    # JS Divergence 손실
    js_loss = kd_js_div_loss(student_logits, teacher_probs, T=3.0)
    print(f"JS Divergence 손실: {js_loss.item():.4f}")
    
    # KnowledgeDistillationLoss 클래스 테스트
    kd_loss = KnowledgeDistillationLoss(method="kl_div", temperature=3.0, alpha=0.7)
    loss = kd_loss(student_logits, teacher_probs, hard_targets)
    print(f"KnowledgeDistillationLoss: {loss.item():.4f}")
    
    print(f"손실 함수 정보: {kd_loss.get_loss_info()}")
