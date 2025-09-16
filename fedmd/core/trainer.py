"""
로컬 학습 유틸리티
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Tuple, List
import logging
import time
from tqdm import tqdm

from core.utils import setup_logger, calculate_accuracy, calculate_loss_metrics, get_device
from core.losses import KnowledgeDistillationLoss, create_kd_loss


class LocalTrainer:
    """로컬 학습기"""
    
    def __init__(self, 
                 model: nn.Module,
                 device: Optional[torch.device] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Args:
            model: 학습할 모델
            device: 사용할 디바이스
            logger: 로거
        """
        self.model = model
        self.device = device or get_device()
        self.logger = logger or setup_logger("LocalTrainer")
        
        # 모델을 디바이스로 이동
        self.model.to(self.device)
        
        # 학습 상태
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.kd_criterion = None
        
        # 메트릭 저장
        self.training_history = []
    
    def setup_optimizer(self, 
                       lr: float = 0.001,
                       weight_decay: float = 1e-4,
                       optimizer_type: str = "adam") -> None:
        """옵티마이저 설정"""
        if optimizer_type.lower() == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_type.lower() == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"지원하지 않는 옵티마이저: {optimizer_type}")
        
        self.logger.info(f"옵티마이저 설정: {optimizer_type}, lr={lr}")
    
    def setup_scheduler(self, 
                       scheduler_type: str = "step",
                       step_size: int = 10,
                       gamma: float = 0.1) -> None:
        """스케줄러 설정"""
        if self.optimizer is None:
            raise ValueError("옵티마이저가 설정되지 않았습니다")
        
        if scheduler_type.lower() == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=step_size,
                gamma=gamma
            )
        elif scheduler_type.lower() == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=step_size
            )
        else:
            raise ValueError(f"지원하지 않는 스케줄러: {scheduler_type}")
        
        self.logger.info(f"스케줄러 설정: {scheduler_type}")
    
    def setup_criterion(self, 
                       criterion_type: str = "cross_entropy",
                       **kwargs) -> None:
        """손실 함수 설정"""
        if criterion_type.lower() == "cross_entropy":
            self.criterion = nn.CrossEntropyLoss(**kwargs)
        elif criterion_type.lower() == "nll":
            self.criterion = nn.NLLLoss(**kwargs)
        else:
            raise ValueError(f"지원하지 않는 손실 함수: {criterion_type}")
        
        self.logger.info(f"손실 함수 설정: {criterion_type}")
    
    def setup_kd_criterion(self, config: Dict[str, Any]) -> None:
        """지식 증류 손실 함수 설정"""
        self.kd_criterion = create_kd_loss(config)
        self.logger.info(f"지식 증류 손실 함수 설정: {self.kd_criterion.get_loss_info()}")
    
    def pretrain(self, 
                dataset: DataLoader,
                epochs: int,
                batch_size: int = 64,
                lr: float = 0.001,
                optimizer_type: str = "adam",
                weight_decay: float = 1e-4,
                criterion_type: str = "cross_entropy") -> Dict[str, List[float]]:
        """
        로컬 사전학습
        
        Args:
            dataset: 학습 데이터셋
            epochs: 에포크 수
            batch_size: 배치 크기
            lr: 학습률
            optimizer_type: 옵티마이저 타입
            weight_decay: 가중치 감쇠
            criterion_type: 손실 함수 타입
        
        Returns:
            학습 메트릭 히스토리
        """
        self.logger.info(f"로컬 사전학습 시작: {epochs} 에포크")
        
        # 설정 초기화
        self.setup_optimizer(lr=lr, weight_decay=weight_decay, optimizer_type=optimizer_type)
        self.setup_criterion(criterion_type=criterion_type)
        
        # 학습 히스토리 초기화
        train_losses = []
        train_accuracies = []
        
        # 학습 모드 설정
        self.model.train()
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_accuracy = 0.0
            num_batches = 0
            
            # 진행률 표시
            pbar = tqdm(dataset, desc=f"Epoch {epoch+1}/{epochs}")
            
            for batch_idx, (data, target) in enumerate(pbar):
                # 데이터를 디바이스로 이동
                data, target = data.to(self.device), target.to(self.device)
                
                # 그래디언트 초기화
                self.optimizer.zero_grad()
                
                # Forward pass
                output = self.model(data)
                loss = self.criterion(output, target)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # 메트릭 계산
                with torch.no_grad():
                    accuracy = calculate_accuracy(output, target)
                
                # 누적
                epoch_loss += loss.item()
                epoch_accuracy += accuracy
                num_batches += 1
                
                # 진행률 업데이트
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{accuracy:.4f}'
                })
            
            # 에포크 평균 메트릭
            avg_loss = epoch_loss / num_batches
            avg_accuracy = epoch_accuracy / num_batches
            
            train_losses.append(avg_loss)
            train_accuracies.append(avg_accuracy)
            
            self.logger.info(f"Epoch {epoch+1}/{epochs}: "
                           f"Loss={avg_loss:.4f}, Accuracy={avg_accuracy:.4f}")
            
            # 스케줄러 업데이트
            if self.scheduler:
                self.scheduler.step()
        
        # 히스토리 저장
        history = {
            "train_loss": train_losses,
            "train_accuracy": train_accuracies
        }
        self.training_history.append(history)
        
        self.logger.info(f"로컬 사전학습 완료: 최종 정확도 {avg_accuracy:.4f}")
        
        return history
    
    def distill(self, 
               dataset: DataLoader,
               soft_targets: Dict[int, List[float]],
               T: float = 3.0,
               alpha: float = 0.7,
               epochs: int = 1,
               batch_size: int = 64,
               lr: float = 0.001) -> Dict[str, List[float]]:
        """
        지식 증류 학습
        
        Args:
            dataset: 학습 데이터셋
            soft_targets: 소프트 타겟 {global_idx: probs}
            T: 온도 파라미터
            alpha: 소프트/하드 타겟 가중치
            epochs: 에포크 수
            batch_size: 배치 크기
            lr: 학습률
        
        Returns:
            학습 메트릭 히스토리
        """
        self.logger.info(f"지식 증류 학습 시작: {epochs} 에포크, T={T}, alpha={alpha}")
        
        # 옵티마이저 재설정 (더 낮은 학습률)
        self.setup_optimizer(lr=lr, weight_decay=1e-4, optimizer_type="adam")
        
        # 지식 증류 손실 함수 설정
        if self.kd_criterion is None:
            config = {
                "distill": {
                    "temperature": T,
                    "alpha": alpha,
                    "method": "kl_div"
                }
            }
            self.setup_kd_criterion(config)
        
        # 학습 히스토리 초기화
        distill_losses = []
        distill_accuracies = []
        
        # 학습 모드 설정
        self.model.train()
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_accuracy = 0.0
            num_batches = 0
            
            # 진행률 표시
            pbar = tqdm(dataset, desc=f"Distill Epoch {epoch+1}/{epochs}")
            
            for batch_idx, (data, meta) in enumerate(pbar):
                # 데이터를 디바이스로 이동
                data = data.to(self.device)
                
                # 글로벌 인덱스 추출
                global_indices = meta["global_idx"]
                targets = meta["target"]
                
                # 소프트 타겟 준비
                soft_targets_batch = []
                hard_targets_batch = []
                
                for i, global_idx in enumerate(global_indices):
                    global_idx = global_idx.item()
                    if global_idx in soft_targets:
                        soft_targets_batch.append(soft_targets[global_idx])
                        hard_targets_batch.append(targets[i].item())
                    else:
                        # 소프트 타겟이 없으면 하드 타겟만 사용
                        soft_targets_batch.append([0.0] * 10)  # 더미
                        hard_targets_batch.append(targets[i].item())
                
                # 텐서로 변환
                soft_targets_tensor = torch.tensor(soft_targets_batch, device=self.device)
                hard_targets_tensor = torch.tensor(hard_targets_batch, device=self.device)
                
                # 그래디언트 초기화
                self.optimizer.zero_grad()
                
                # Forward pass
                output = self.model(data)
                
                # 지식 증류 손실 계산
                loss = self.kd_criterion(
                    output, 
                    soft_targets_tensor, 
                    hard_targets_tensor
                )
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # 메트릭 계산
                with torch.no_grad():
                    accuracy = calculate_accuracy(output, hard_targets_tensor)
                
                # 누적
                epoch_loss += loss.item()
                epoch_accuracy += accuracy
                num_batches += 1
                
                # 진행률 업데이트
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{accuracy:.4f}'
                })
            
            # 에포크 평균 메트릭
            avg_loss = epoch_loss / num_batches
            avg_accuracy = epoch_accuracy / num_batches
            
            distill_losses.append(avg_loss)
            distill_accuracies.append(avg_accuracy)
            
            self.logger.info(f"Distill Epoch {epoch+1}/{epochs}: "
                           f"Loss={avg_loss:.4f}, Accuracy={avg_accuracy:.4f}")
        
        # 히스토리 저장
        history = {
            "distill_loss": distill_losses,
            "distill_accuracy": distill_accuracies
        }
        self.training_history.append(history)
        
        self.logger.info(f"지식 증류 학습 완료: 최종 정확도 {avg_accuracy:.4f}")
        
        return history
    
    def evaluate(self, dataset: DataLoader) -> Dict[str, float]:
        """
        모델 평가
        
        Args:
            dataset: 평가 데이터셋
        
        Returns:
            평가 메트릭
        """
        self.logger.info("모델 평가 시작")
        
        # 평가 모드 설정
        self.model.eval()
        
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for data, target in tqdm(dataset, desc="Evaluating"):
                # 데이터를 디바이스로 이동
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                output = self.model(data)
                
                # 메트릭 계산
                if self.criterion:
                    loss = self.criterion(output, target).item()
                    total_loss += loss
                
                accuracy = calculate_accuracy(output, target)
                total_accuracy += accuracy
                num_batches += 1
        
        # 평균 메트릭
        avg_loss = total_loss / num_batches if self.criterion else 0.0
        avg_accuracy = total_accuracy / num_batches
        
        metrics = {
            "loss": avg_loss,
            "accuracy": avg_accuracy
        }
        
        self.logger.info(f"평가 완료: Loss={avg_loss:.4f}, Accuracy={avg_accuracy:.4f}")
        
        return metrics
    
    def get_training_history(self) -> List[Dict[str, List[float]]]:
        """학습 히스토리 반환"""
        return self.training_history
    
    def save_checkpoint(self, filepath: str) -> None:
        """체크포인트 저장"""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer else None,
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "training_history": self.training_history
        }
        
        torch.save(checkpoint, filepath)
        self.logger.info(f"체크포인트 저장됨: {filepath}")
    
    def load_checkpoint(self, filepath: str) -> None:
        """체크포인트 로드"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        
        if checkpoint["optimizer_state_dict"] and self.optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if checkpoint["scheduler_state_dict"] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        self.training_history = checkpoint.get("training_history", [])
        
        self.logger.info(f"체크포인트 로드됨: {filepath}")


if __name__ == "__main__":
    # 테스트 실행
    from models import build_model
    from core.private_dataset import PrivateDataset
    from torch.utils.data import DataLoader
    
    # 모델 생성
    model = build_model("cnn_small", num_classes=10)
    
    # 트레이너 생성
    trainer = LocalTrainer(model)
    
    # 더미 데이터셋
    config = {"root": "data/private", "num_classes": 10, "num_samples": 100}
    dataset = PrivateDataset("test_client", config)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 사전학습 테스트
    print("=== 사전학습 테스트 ===")
    history = trainer.pretrain(dataloader, epochs=2, lr=0.001)
    print(f"학습 히스토리: {history}")
    
    # 평가 테스트
    print("\n=== 평가 테스트 ===")
    metrics = trainer.evaluate(dataloader)
    print(f"평가 메트릭: {metrics}")
    
    # 지식 증류 테스트
    print("\n=== 지식 증류 테스트 ===")
    soft_targets = {0: [0.1, 0.9], 1: [0.2, 0.8]}  # 더미 소프트 타겟
    distill_history = trainer.distill(dataloader, soft_targets, epochs=1)
    print(f"증류 히스토리: {distill_history}")
