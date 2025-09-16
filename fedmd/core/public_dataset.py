"""
공용 데이터셋 관리
FedMD에서 모든 클라이언트가 공유하는 레퍼런스 데이터셋
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


class PublicRefDataset(Dataset):
    """공용 레퍼런스 데이터셋"""
    
    def __init__(self, config: Dict[str, Any], indices: Optional[List[int]] = None):
        """
        Args:
            config: 데이터셋 설정
            indices: 사용할 인덱스 목록 (None이면 전체 사용)
        """
        self.config = config
        self.logger = setup_logger(f"PublicRefDataset")
        
        # 데이터셋 로드
        self.data, self.targets = self._load_dataset()
        
        # 인덱스 설정
        if indices is not None:
            self.indices = indices
        else:
            self.indices = list(range(len(self.data)))
        
        self.logger.info(f"공용 데이터셋 로드 완료: {len(self.indices)}개 샘플")
    
    def _load_dataset(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """데이터셋 로드"""
        dataset_name = self.config["name"].lower()
        
        if dataset_name == "cifar10":
            return self._load_cifar10()
        elif dataset_name == "mnist":
            return self._load_mnist()
        elif dataset_name == "test_public":
            return self._load_test_public()
        else:
            raise ValueError(f"지원하지 않는 데이터셋: {dataset_name}")
    
    def _load_cifar10(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """CIFAR-10 데이터셋 로드"""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        # 테스트 데이터셋 사용 (공용 데이터로)
        dataset = torchvision.datasets.CIFAR10(
            root=self.config["path"],
            train=False,
            download=True,
            transform=transform
        )
        
        data = torch.stack([dataset[i][0] for i in range(len(dataset))])
        targets = torch.tensor([dataset[i][1] for i in range(len(dataset))])
        
        return data, targets

    def _load_test_public(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """테스트용 공용 데이터셋 로드 (더미 데이터)"""
        # 더미 데이터 생성
        num_samples = 1000
        num_classes = 10
        
        # 랜덤 이미지 데이터 (3, 32, 32)
        data = torch.randn(num_samples, 3, 32, 32)
        
        # 랜덤 레이블
        targets = torch.randint(0, num_classes, (num_samples,))
        
        self.logger.info(f"테스트용 공용 데이터셋 생성: {num_samples}개 샘플")
        
        return data, targets

    def _load_mnist(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """MNIST 데이터셋 로드"""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        dataset = torchvision.datasets.MNIST(
            root=self.config["path"],
            train=False,
            download=True,
            transform=transform
        )
        
        data = torch.stack([dataset[i][0] for i in range(len(dataset))])
        targets = torch.tensor([dataset[i][1] for i in range(len(dataset))])
        
        return data, targets
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Args:
            idx: 데이터셋 내 인덱스
            
        Returns:
            (data, meta): 데이터와 메타데이터
        """
        global_idx = self.indices[idx]
        data = self.data[global_idx]
        target = self.targets[global_idx]
        
        meta = {
            "global_idx": global_idx,
            "target": target.item(),
            "dataset_name": self.config["name"]
        }
        
        return data, meta
    
    def get_subset(self, size: int, random_seed: int = 42) -> 'PublicRefDataset':
        """서브셋 생성"""
        random.seed(random_seed)
        subset_indices = random.sample(self.indices, min(size, len(self.indices)))
        return PublicRefDataset(self.config, subset_indices)


def get_public_indices(config: Dict[str, Any], 
                      subset_size: Optional[int] = None,
                      random_seed: int = 42) -> List[int]:
    """
    공용 데이터셋의 글로벌 인덱스 생성/로드
    
    Args:
        config: 데이터셋 설정
        subset_size: 서브셋 크기 (None이면 전체)
        random_seed: 랜덤 시드
        
    Returns:
        글로벌 인덱스 목록
    """
    logger = setup_logger("get_public_indices")
    
    # 인덱스 파일 경로
    data_dir = Path(config["path"])
    index_file = data_dir / config["index_file"]
    
    # 기존 인덱스 파일이 있으면 로드
    if index_file.exists():
        logger.info(f"기존 인덱스 파일 로드: {index_file}")
        with open(index_file, 'r') as f:
            indices = json.load(f)
        return indices
    
    # 새로 생성
    logger.info("새 인덱스 파일 생성 중...")
    
    # 전체 데이터셋 크기 확인
    dataset_name = config["name"].lower()
    if dataset_name == "cifar10":
        full_size = 10000  # CIFAR-10 테스트셋
    elif dataset_name == "mnist":
        full_size = 10000  # MNIST 테스트셋
    else:
        raise ValueError(f"지원하지 않는 데이터셋: {dataset_name}")
    
    # 서브셋 크기 결정
    if subset_size is None:
        subset_size = config.get("subset_size", full_size)
    
    subset_size = min(subset_size, full_size)
    
    # 랜덤 인덱스 생성
    random.seed(random_seed)
    indices = random.sample(range(full_size), subset_size)
    indices.sort()  # 정렬하여 일관성 보장
    
    # 저장
    ensure_dir(data_dir)
    with open(index_file, 'w') as f:
        json.dump(indices, f, indent=2)
    
    logger.info(f"인덱스 파일 저장됨: {index_file} ({len(indices)}개 인덱스)")
    return indices


def create_public_dataloader(config: Dict[str, Any], 
                           indices: List[int],
                           batch_size: int = 64,
                           shuffle: bool = False) -> DataLoader:
    """공용 데이터셋 DataLoader 생성"""
    dataset = PublicRefDataset(config, indices)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # Windows 호환성
        pin_memory=torch.cuda.is_available()
    )


def load_public_indices_from_file(config: Dict[str, Any]) -> List[int]:
    """파일에서 공용 인덱스 로드"""
    data_dir = Path(config["path"])
    index_file = data_dir / config["index_file"]
    
    if not index_file.exists():
        raise FileNotFoundError(f"인덱스 파일을 찾을 수 없습니다: {index_file}")
    
    with open(index_file, 'r') as f:
        return json.load(f)


def save_public_indices(indices: List[int], config: Dict[str, Any]) -> None:
    """공용 인덱스를 파일로 저장"""
    data_dir = Path(config["path"])
    index_file = data_dir / config["index_file"]
    
    ensure_dir(data_dir)
    with open(index_file, 'w') as f:
        json.dump(indices, f, indent=2)


if __name__ == "__main__":
    # 테스트 실행
    config = {
        "name": "CIFAR10",
        "path": "data/public",
        "index_file": "public_indices.json",
        "subset_size": 100
    }
    
    # 인덱스 생성
    indices = get_public_indices(config, subset_size=100)
    print(f"생성된 인덱스: {len(indices)}개")
    
    # 데이터셋 테스트
    dataset = PublicRefDataset(config, indices)
    print(f"데이터셋 크기: {len(dataset)}")
    
    # 샘플 로드
    data, meta = dataset[0]
    print(f"데이터 형태: {data.shape}")
    print(f"메타데이터: {meta}")
