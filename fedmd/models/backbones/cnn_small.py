"""
작은 CNN 모델 (CIFAR-10용)
FedMD 실험에 적합한 경량 모델
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class CNNSmall(nn.Module):
    """
    작은 CNN 모델
    
    CIFAR-10 분류를 위한 경량 CNN 아키텍처
    - 3x3 컨볼루션 레이어들
    - 배치 정규화
    - 드롭아웃
    - 글로벌 평균 풀링
    """
    
    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.5):
        """
        Args:
            num_classes: 분류할 클래스 수
            dropout_rate: 드롭아웃 비율
        """
        super(CNNSmall, self).__init__()
        
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # 컨볼루션 레이어들
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        # 풀링 레이어
        self.pool = nn.MaxPool2d(2, 2)
        
        # 글로벌 평균 풀링
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # 분류기
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(256, num_classes)
        
        # 가중치 초기화
        self._initialize_weights()
    
    def _initialize_weights(self):
        """가중치 초기화"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: 입력 텐서 (B, 3, 32, 32)
            
        Returns:
            로짓 텐서 (B, num_classes)
        """
        # 첫 번째 컨볼루션 블록
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # (B, 32, 16, 16)
        
        # 두 번째 컨볼루션 블록
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # (B, 64, 8, 8)
        
        # 세 번째 컨볼루션 블록
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # (B, 128, 4, 4)
        
        # 네 번째 컨볼루션 블록
        x = F.relu(self.bn4(self.conv4(x)))  # (B, 256, 4, 4)
        
        # 글로벌 평균 풀링
        x = self.global_avg_pool(x)  # (B, 256, 1, 1)
        x = x.view(x.size(0), -1)  # (B, 256)
        
        # 분류기
        x = self.dropout(x)
        x = self.fc(x)  # (B, num_classes)
        
        return x
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        특징 추출 (분류기 제외)
        
        Args:
            x: 입력 텐서 (B, 3, 32, 32)
            
        Returns:
            특징 텐서 (B, 256)
        """
        # 컨볼루션 레이어들
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = F.relu(self.bn4(self.conv4(x)))
        
        # 글로벌 평균 풀링
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        
        return x
    
    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """임베딩 추출 (드롭아웃 제외)"""
        features = self.get_features(x)
        return features
    
    def count_parameters(self) -> dict:
        """파라미터 수 계산"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "total": total_params,
            "trainable": trainable_params,
            "non_trainable": total_params - trainable_params
        }
    
    def get_model_size(self) -> float:
        """모델 크기 (MB) 계산"""
        param_size = 0
        for param in self.parameters():
            param_size += param.nelement() * param.element_size()
        
        buffer_size = 0
        for buffer in self.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_all_mb = (param_size + buffer_size) / 1024**2
        return size_all_mb


class CNNSmallV2(nn.Module):
    """
    CNNSmall의 개선 버전
    
    - 더 깊은 네트워크
    - 잔차 연결 (Residual-like)
    - 더 나은 정규화
    """
    
    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.3):
        super(CNNSmallV2, self).__init__()
        
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # 컨볼루션 블록들
        self.conv_block1 = self._make_conv_block(3, 64, 2)
        self.conv_block2 = self._make_conv_block(64, 128, 2)
        self.conv_block3 = self._make_conv_block(128, 256, 2)
        self.conv_block4 = self._make_conv_block(256, 512, 1)
        
        # 글로벌 평균 풀링
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # 분류기
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(512, num_classes)
        
        self._initialize_weights()
    
    def _make_conv_block(self, in_channels: int, out_channels: int, 
                        num_convs: int) -> nn.Module:
        """컨볼루션 블록 생성"""
        layers = []
        
        for i in range(num_convs):
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ])
            in_channels = out_channels
        
        layers.append(nn.MaxPool2d(2, 2))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """가중치 초기화"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        x = self.conv_block1(x)  # (B, 64, 16, 16)
        x = self.conv_block2(x)  # (B, 128, 8, 8)
        x = self.conv_block3(x)  # (B, 256, 4, 4)
        x = self.conv_block4(x)  # (B, 512, 4, 4)
        
        x = self.global_avg_pool(x)  # (B, 512, 1, 1)
        x = x.view(x.size(0), -1)  # (B, 512)
        
        x = self.dropout(x)
        x = self.fc(x)  # (B, num_classes)
        
        return x


if __name__ == "__main__":
    # 테스트 실행
    model = CNNSmall(num_classes=10)
    
    # 입력 테스트
    x = torch.randn(2, 3, 32, 32)
    output = model(x)
    
    print(f"입력 형태: {x.shape}")
    print(f"출력 형태: {output.shape}")
    print(f"파라미터 수: {model.count_parameters()}")
    print(f"모델 크기: {model.get_model_size():.2f} MB")
    
    # 특징 추출 테스트
    features = model.get_features(x)
    print(f"특징 형태: {features.shape}")
    
    # V2 모델 테스트
    model_v2 = CNNSmallV2(num_classes=10)
    output_v2 = model_v2(x)
    print(f"V2 출력 형태: {output_v2.shape}")
    print(f"V2 파라미터 수: {model_v2.count_parameters()}")
