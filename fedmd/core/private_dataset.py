"""
개인 데이터셋 관리
각 클라이언트의 프라이빗 데이터를 관리
"""
import json
import random
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import logging

from core.utils import setup_logger, ensure_dir


class PrivateDataset(Dataset):
    """개인 데이터셋"""
    
    def __init__(self, client_id: str, config: Dict[str, Any], 
                 partition_type: str = "dir-per-client"):
        """
        Args:
            client_id: 클라이언트 ID
            config: 데이터셋 설정
            partition_type: 파티션 방식 ("dir-per-client", "iid", "non-iid")
        """
        self.client_id = client_id
        self.config = config
        self.partition_type = partition_type
        self.logger = setup_logger(f"PrivateDataset-{client_id}")
        
        # 데이터 로드
        self.data, self.targets = self._load_private_data()
        
        self.logger.info(f"클라이언트 {client_id} 개인 데이터 로드 완료: {len(self.data)}개 샘플")
    
    def _load_private_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """개인 데이터 로드"""
        private_root = Path(self.config["root"])
        client_dir = private_root / self.client_id
        
        # 클라이언트별 데이터 파일 경로
        data_file = client_dir / "data.pt"
        targets_file = client_dir / "targets.pt"
        
        if data_file.exists() and targets_file.exists():
            # 기존 데이터 로드
            data = torch.load(data_file)
            targets = torch.load(targets_file)
        else:
            # 새 데이터 생성 (더미 데이터)
            self.logger.warning(f"클라이언트 {self.client_id} 데이터가 없습니다. 더미 데이터를 생성합니다.")
            data, targets = self._generate_dummy_data()
            
            # 저장
            ensure_dir(client_dir)
            torch.save(data, data_file)
            torch.save(targets, targets_file)
        
        return data, targets
    
    def _generate_dummy_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """더미 개인 데이터 생성"""
        # 설정에서 클래스 수 가져오기 (기본값: 10)
        num_classes = self.config.get("num_classes", 10)
        num_samples = self.config.get("num_samples", 100)
        
        # 랜덤 시드 설정 (클라이언트별로 다른 시드)
        random.seed(hash(self.client_id) % 2**32)
        np.random.seed(hash(self.client_id) % 2**32)
        torch.manual_seed(hash(self.client_id) % 2**32)
        
        # 더미 데이터 생성 (CIFAR-10 스타일)
        data = torch.randn(num_samples, 3, 32, 32)
        targets = torch.randint(0, num_classes, (num_samples,))
        
        # 클래스 불균형 시뮬레이션 (non-IID)
        if self.partition_type == "non-iid":
            # 클라이언트별로 특정 클래스에 집중
            client_class = hash(self.client_id) % num_classes
            # 70%는 특정 클래스, 30%는 랜덤
            for i in range(num_samples):
                if random.random() < 0.7:
                    targets[i] = client_class
                else:
                    targets[i] = random.randint(0, num_classes - 1)
        
        return data, targets
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            idx: 데이터셋 내 인덱스
            
        Returns:
            (data, target): 데이터와 타겟
        """
        return self.data[idx], self.targets[idx]
    
    def get_class_distribution(self) -> Dict[int, int]:
        """클래스별 분포 반환"""
        unique, counts = torch.unique(self.targets, return_counts=True)
        return {int(cls): int(count) for cls, count in zip(unique, counts)}


def create_private_dataloader(client_id: str, 
                            config: Dict[str, Any],
                            batch_size: int = 64,
                            shuffle: bool = True) -> DataLoader:
    """개인 데이터셋 DataLoader 생성"""
    dataset = PrivateDataset(client_id, config)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # Windows 호환성
        pin_memory=torch.cuda.is_available()
    )


def create_client_partitions(config: Dict[str, Any], 
                           client_ids: List[str],
                           source_dataset: str = "CIFAR10",
                           partition_type: str = "dir-per-client") -> None:
    """
    클라이언트별 데이터 파티션 생성
    
    Args:
        config: 데이터셋 설정
        client_ids: 클라이언트 ID 목록
        source_dataset: 소스 데이터셋명
        partition_type: 파티션 방식
    """
    logger = setup_logger("create_client_partitions")
    private_root = Path(config["root"])
    ensure_dir(private_root)
    
    # 소스 데이터셋 로드
    if source_dataset.lower() == "cifar10":
        data, targets = _load_cifar10_for_partition()
    elif source_dataset.lower() == "mnist":
        data, targets = _load_mnist_for_partition()
    else:
        raise ValueError(f"지원하지 않는 소스 데이터셋: {source_dataset}")
    
    logger.info(f"소스 데이터셋 로드 완료: {len(data)}개 샘플")
    
    # 파티션 생성
    if partition_type == "dir-per-client":
        _create_dir_per_client_partition(data, targets, client_ids, private_root, logger)
    elif partition_type == "iid":
        _create_iid_partition(data, targets, client_ids, private_root, logger)
    elif partition_type == "non-iid":
        _create_non_iid_partition(data, targets, client_ids, private_root, logger)
    else:
        raise ValueError(f"지원하지 않는 파티션 방식: {partition_type}")


def _load_cifar10_for_partition() -> Tuple[torch.Tensor, torch.Tensor]:
    """CIFAR-10 파티션용 로드"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    dataset = torchvision.datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=transform
    )
    
    data = torch.stack([dataset[i][0] for i in range(len(dataset))])
    targets = torch.tensor([dataset[i][1] for i in range(len(dataset))])
    
    return data, targets


def _load_mnist_for_partition() -> Tuple[torch.Tensor, torch.Tensor]:
    """MNIST 파티션용 로드"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    dataset = torchvision.datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=transform
    )
    
    data = torch.stack([dataset[i][0] for i in range(len(dataset))])
    targets = torch.tensor([dataset[i][1] for i in range(len(dataset))])
    
    return data, targets


def _create_dir_per_client_partition(data: torch.Tensor, 
                                   targets: torch.Tensor,
                                   client_ids: List[str],
                                   private_root: Path,
                                   logger: logging.Logger) -> None:
    """디렉터리별 클라이언트 파티션 생성"""
    samples_per_client = len(data) // len(client_ids)
    
    for i, client_id in enumerate(client_ids):
        start_idx = i * samples_per_client
        end_idx = start_idx + samples_per_client if i < len(client_ids) - 1 else len(data)
        
        client_data = data[start_idx:end_idx]
        client_targets = targets[start_idx:end_idx]
        
        client_dir = private_root / client_id
        ensure_dir(client_dir)
        
        torch.save(client_data, client_dir / "data.pt")
        torch.save(client_targets, client_dir / "targets.pt")
        
        logger.info(f"클라이언트 {client_id}: {len(client_data)}개 샘플")


def _create_iid_partition(data: torch.Tensor,
                         targets: torch.Tensor,
                         client_ids: List[str],
                         private_root: Path,
                         logger: logging.Logger) -> None:
    """IID 파티션 생성"""
    # 랜덤 셔플
    indices = torch.randperm(len(data))
    data = data[indices]
    targets = targets[indices]
    
    samples_per_client = len(data) // len(client_ids)
    
    for i, client_id in enumerate(client_ids):
        start_idx = i * samples_per_client
        end_idx = start_idx + samples_per_client if i < len(client_ids) - 1 else len(data)
        
        client_data = data[start_idx:end_idx]
        client_targets = targets[start_idx:end_idx]
        
        client_dir = private_root / client_id
        ensure_dir(client_dir)
        
        torch.save(client_data, client_dir / "data.pt")
        torch.save(client_targets, client_dir / "targets.pt")
        
        logger.info(f"클라이언트 {client_id}: {len(client_data)}개 샘플")


def _create_non_iid_partition(data: torch.Tensor,
                             targets: torch.Tensor,
                             client_ids: List[str],
                             private_root: Path,
                             logger: logging.Logger) -> None:
    """Non-IID 파티션 생성 (클라이언트별로 특정 클래스 집중)"""
    num_classes = len(torch.unique(targets))
    classes_per_client = max(1, num_classes // len(client_ids))
    
    for i, client_id in enumerate(client_ids):
        # 클라이언트별 클래스 할당
        start_class = i * classes_per_client
        end_class = min(start_class + classes_per_client, num_classes)
        
        if i == len(client_ids) - 1:  # 마지막 클라이언트는 남은 모든 클래스
            end_class = num_classes
        
        # 해당 클래스들의 데이터만 선택
        class_mask = torch.zeros(len(targets), dtype=torch.bool)
        for cls in range(start_class, end_class):
            class_mask |= (targets == cls)
        
        client_data = data[class_mask]
        client_targets = targets[class_mask]
        
        # 랜덤 셔플
        indices = torch.randperm(len(client_data))
        client_data = client_data[indices]
        client_targets = client_targets[indices]
        
        client_dir = private_root / client_id
        ensure_dir(client_dir)
        
        torch.save(client_data, client_dir / "data.pt")
        torch.save(client_targets, client_dir / "targets.pt")
        
        logger.info(f"클라이언트 {client_id}: {len(client_data)}개 샘플, "
                   f"클래스 {start_class}-{end_class-1}")


if __name__ == "__main__":
    # 테스트 실행
    config = {
        "root": "data/private",
        "num_classes": 10,
        "num_samples": 100
    }
    
    # 더미 데이터 생성 테스트
    dataset = PrivateDataset("test_client", config)
    print(f"데이터셋 크기: {len(dataset)}")
    print(f"클래스 분포: {dataset.get_class_distribution()}")
    
    # 샘플 로드
    data, target = dataset[0]
    print(f"데이터 형태: {data.shape}, 타겟: {target}")
