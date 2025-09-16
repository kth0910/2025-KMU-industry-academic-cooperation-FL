"""
FedMD 클라이언트 구현
"""
import time
import torch
from typing import Dict, Any, List, Optional, Tuple
import logging
from pathlib import Path
from tqdm import tqdm

from core.utils import setup_logger, get_device, save_metrics, ensure_dir
from core.public_dataset import PublicRefDataset, create_public_dataloader
from core.private_dataset import PrivateDataset, create_private_dataloader
from core.trainer import LocalTrainer
from core.server import FedMDServer  # 모의 서버용
from models import build_model
from comms import FedMDGrpcClient


class ClientNode:
    """FedMD 클라이언트 노드"""
    
    def __init__(self, 
                 client_id: str,
                 config: Dict[str, Any],
                 comm_client: Optional[FedMDGrpcClient] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Args:
            client_id: 클라이언트 ID
            config: 클라이언트 설정
            comm_client: 통신 클라이언트 (None이면 모의 서버 사용)
            logger: 로거
        """
        self.client_id = client_id
        self.config = config
        self.comm_client = comm_client
        self.logger = logger or setup_logger(f"ClientNode-{client_id}")
        
        # 디바이스 설정
        self.device = get_device()
        self.logger.info(f"클라이언트 {client_id} 초기화: 디바이스 {self.device}")
        
        # 데이터셋
        self.private_dataset = None
        self.public_dataset = None
        self.private_dataloader = None
        self.public_dataloader = None
        
        # 모델
        self.model = None
        self.trainer = None
        
        # 서버 정보
        self.server_info = {}
        self.public_indices = []
        self.current_round = 0
        
        # 메트릭 저장
        self.metrics_history = []
        self.artifacts_dir = Path("artifacts") / client_id
        ensure_dir(self.artifacts_dir)
        
        # 모의 서버 (통신 클라이언트가 없을 때)
        self.mock_server = None
        if self.comm_client is None:
            self.mock_server = FedMDServer(config, self.logger)
    
    def register(self) -> bool:
        """서버에 등록"""
        self.logger.info(f"서버 등록 시작: {self.client_id}")
        
        try:
            if self.comm_client:
                # gRPC 클라이언트 사용
                result = self.comm_client.register(
                    model_name=self.config["model"]["backbone"],
                    num_classes=self.config["model"]["num_classes"]
                )
            else:
                # 모의 서버 사용
                client_info = {
                    "client_id": self.client_id,
                    "model_name": self.config["model"]["backbone"],
                    "weight": 1.0
                }
                result = self.mock_server.register_client(client_info)
            
            if result["success"]:
                self.server_info = result
                self.public_indices = result.get("public_indices", [])
                self.current_round = result.get("round_id", 0)
                
                self.logger.info(f"서버 등록 성공: 라운드 {self.current_round}")
                return True
            else:
                self.logger.error(f"서버 등록 실패: {result.get('message', '알 수 없는 오류')}")
                return False
                
        except Exception as e:
            self.logger.error(f"서버 등록 중 오류: {e}")
            return False
    
    def load_datasets(self) -> bool:
        """데이터셋 로드"""
        self.logger.info("데이터셋 로드 시작")
        
        try:
            # 개인 데이터셋 로드
            self.private_dataset = PrivateDataset(
                self.client_id,
                self.config["dataset"]["private"]
            )
            
            # 공용 데이터셋 로드
            if self.public_indices:
                self.public_dataset = PublicRefDataset(
                    self.config["dataset"]["public"],
                    self.public_indices
                )
            else:
                self.logger.warning("공용 인덱스가 없습니다. 공용 데이터셋을 로드할 수 없습니다.")
                return False
            
            # DataLoader 생성
            train_config = self.config.get("train", {})
            local_config = train_config.get("local", {})
            distill_config = train_config.get("distill", {})
            
            self.private_dataloader = create_private_dataloader(
                self.client_id,
                self.config["dataset"]["private"],
                batch_size=local_config.get("batch_size", 64),
                shuffle=True
            )
            
            self.public_dataloader = create_public_dataloader(
                self.config["dataset"]["public"],
                self.public_indices,
                batch_size=distill_config.get("batch_size", 64),
                shuffle=False
            )
            
            self.logger.info(f"데이터셋 로드 완료: 개인 {len(self.private_dataset)}개, "
                           f"공용 {len(self.public_dataset)}개")
            
            return True
            
        except Exception as e:
            self.logger.error(f"데이터셋 로드 실패: {e}")
            return False
    
    def build_model(self) -> bool:
        """모델 빌드"""
        self.logger.info("모델 빌드 시작")
        
        try:
            # 모델 생성
            self.model = build_model(
                self.config["model"]["backbone"],
                self.config["model"]["num_classes"]
            )
            
            # 트레이너 생성
            self.trainer = LocalTrainer(self.model, self.device, self.logger)
            
            self.logger.info(f"모델 빌드 완료: {self.config['model']['backbone']}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"모델 빌드 실패: {e}")
            return False
    
    def local_pretrain(self) -> Dict[str, Any]:
        """로컬 사전학습"""
        self.logger.info("로컬 사전학습 시작")
        
        if not self.private_dataloader:
            raise ValueError("개인 데이터셋이 로드되지 않았습니다")
        
        if not self.trainer:
            raise ValueError("트레이너가 초기화되지 않았습니다")
        
        # 학습 설정
        train_config = self.config.get("train", {})
        local_config = train_config.get("local", {})
        
        epochs = local_config.get("epochs", 2)
        batch_size = local_config.get("batch_size", 64)
        lr = local_config.get("lr", 0.001)
        
        # 사전학습 실행
        history = self.trainer.pretrain(
            self.private_dataloader,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr
        )
        
        self.logger.info("로컬 사전학습 완료")
        
        return history
    
    def infer_public_logits(self) -> Dict[int, Dict[str, Any]]:
        """공용 데이터셋에 대한 로짓 추론"""
        self.logger.info("공용 데이터셋 로짓 추론 시작")
        
        if not self.public_dataloader:
            raise ValueError("공용 데이터셋이 로드되지 않았습니다")
        
        if not self.model:
            raise ValueError("모델이 빌드되지 않았습니다")
        
        # 평가 모드
        self.model.eval()
        
        logits_data = {}
        
        with torch.no_grad():
            for data, meta in self.public_dataloader:
                # 데이터를 디바이스로 이동
                data = data.to(self.device)
                
                # Forward pass
                output = self.model(data)
                
                # 로짓 추출
                for i, global_idx in enumerate(meta["global_idx"]):
                    global_idx = global_idx.item()
                    logits = output[i].cpu().tolist()
                    
                    # 신뢰도 계산 (최대 확률)
                    confidence = torch.softmax(output[i], dim=0).max().item()
                    
                    logits_data[global_idx] = {
                        "logits": logits,
                        "confidence": confidence,
                        "timestamp": int(time.time() * 1000)
                    }
        
        self.logger.info(f"로짓 추론 완료: {len(logits_data)}개 샘플")
        
        return logits_data
    
    def upload_logits(self, logits_data: Dict[int, Dict[str, Any]]) -> bool:
        """로짓 업로드"""
        self.logger.info("로짓 업로드 시작")
        
        try:
            if self.comm_client:
                # gRPC 클라이언트 사용
                result = self.comm_client.upload_logits(
                    self.current_round,
                    logits_data
                )
            else:
                # 모의 서버 사용 - 서버의 현재 라운드와 동기화
                server_round = self.mock_server.current_round
                if server_round != self.current_round:
                    self.logger.warning(f"클라이언트 라운드({self.current_round})와 서버 라운드({server_round}) 불일치, 서버 라운드로 동기화")
                    self.current_round = server_round
                
                # 서버의 라운드 상태를 COLLECTING으로 설정
                if self.mock_server.round_status != "COLLECTING":
                    self.mock_server.round_status = "COLLECTING"
                    self.logger.info(f"서버 라운드 상태를 COLLECTING으로 설정")
                
                result = self.mock_server.collect_logits(
                    self.client_id,
                    self.current_round,
                    logits_data
                )
            
            if result["success"]:
                self.logger.info("로짓 업로드 성공")
                return True
            else:
                self.logger.error(f"로짓 업로드 실패: {result.get('message', '알 수 없는 오류')}")
                return False
                
        except Exception as e:
            self.logger.error(f"로짓 업로드 중 오류: {e}")
            return False
    
    def receive_soft_targets(self) -> Optional[Dict[int, List[float]]]:
        """소프트 타겟 수신"""
        self.logger.info("소프트 타겟 수신 시작")
        
        try:
            if self.comm_client:
                # gRPC 클라이언트 사용
                result = self.comm_client.get_soft_targets(self.current_round)
            else:
                # 모의 서버 사용 - 서버의 현재 라운드와 동기화
                server_round = self.mock_server.current_round
                if server_round != self.current_round:
                    self.logger.warning(f"클라이언트 라운드({self.current_round})와 서버 라운드({server_round}) 불일치, 서버 라운드로 동기화")
                    self.current_round = server_round
                
                result = self.mock_server.get_soft_targets(self.current_round)
            
            if result["success"]:
                soft_targets = result["soft_targets"]
                self.logger.info(f"소프트 타겟 수신 완료: {len(soft_targets)}개")
                return soft_targets
            else:
                self.logger.error(f"소프트 타겟 수신 실패: {result.get('message', '알 수 없는 오류')}")
                return None
                
        except Exception as e:
            self.logger.error(f"소프트 타겟 수신 중 오류: {e}")
            return None
    
    def distill(self, soft_targets: Dict[int, List[float]]) -> Dict[str, Any]:
        """지식 증류 학습"""
        self.logger.info("지식 증류 학습 시작")
        
        if not self.public_dataloader:
            raise ValueError("공용 데이터셋이 로드되지 않았습니다")
        
        if not self.trainer:
            raise ValueError("트레이너가 초기화되지 않았습니다")
        
        # 학습 설정
        train_config = self.config.get("train", {})
        distill_config = train_config.get("distill", {})
        
        epochs = distill_config.get("epochs", 1)
        batch_size = distill_config.get("batch_size", 64)
        lr = distill_config.get("lr", 0.001)
        
        # 증류 설정
        distill_params = self.config.get("distill", {})
        temperature = distill_params.get("temperature", 3.0)
        alpha = distill_params.get("alpha", 0.7)
        
        # 지식 증류 학습 실행
        history = self.trainer.distill(
            self.public_dataloader,
            soft_targets,
            T=temperature,
            alpha=alpha,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr
        )
        
        self.logger.info("지식 증류 학습 완료")
        
        return history
    
    def report_metrics(self, metrics: Dict[str, float]) -> bool:
        """메트릭 보고"""
        self.logger.info("메트릭 보고 시작")
        
        try:
            # 메트릭 데이터 변환
            metrics_data = {}
            for name, value in metrics.items():
                metrics_data[name] = {
                    "value": value,
                    "unit": "",
                    "timestamp": int(time.time() * 1000)
                }
            
            if self.comm_client:
                # gRPC 클라이언트 사용
                result = self.comm_client.report_metrics(
                    self.current_round,
                    metrics_data
                )
            else:
                # 모의 서버 사용
                result = self.mock_server.report_metrics(
                    self.client_id,
                    self.current_round,
                    metrics_data
                )
            
            if result["success"]:
                self.logger.info("메트릭 보고 성공")
                return True
            else:
                self.logger.error(f"메트릭 보고 실패: {result.get('message', '알 수 없는 오류')}")
                return False
                
        except Exception as e:
            self.logger.error(f"메트릭 보고 중 오류: {e}")
            return False
    
    def evaluate(self) -> Dict[str, float]:
        """모델 평가"""
        self.logger.info("모델 평가 시작")
        
        if not self.private_dataloader:
            raise ValueError("개인 데이터셋이 로드되지 않았습니다")
        
        if not self.trainer:
            raise ValueError("트레이너가 초기화되지 않았습니다")
        
        # 평가 실행
        metrics = self.trainer.evaluate(self.private_dataloader)
        
        self.logger.info(f"모델 평가 완료: {metrics}")
        
        return metrics
    
    def run_round(self) -> Dict[str, Any]:
        """라운드 실행"""
        self.logger.info(f"라운드 {self.current_round} 실행 시작")
        
        round_results = {
            "round_id": self.current_round,
            "client_id": self.client_id,
            "success": False,
            "metrics": {}
        }
        
        # 라운드 단계별 진행률 표시
        steps = [
            "1️⃣ 로컬 사전학습",
            "2️⃣ 공용 데이터셋 로짓 추론", 
            "3️⃣ 로짓 업로드",
            "4️⃣ 소프트 타겟 수신 대기",
            "5️⃣ 지식 증류 학습",
            "6️⃣ 모델 평가",
            "7️⃣ 메트릭 보고"
        ]
        
        with tqdm(total=len(steps), desc=f"👤 {self.client_id} 라운드 {self.current_round}", 
                 unit="단계", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} 단계 [{elapsed}<{remaining}]") as pbar:
            
            try:
                # 1. 로컬 사전학습
                pbar.set_description(f"👤 {self.client_id} - {steps[0]}")
                pbar.set_postfix_str("로컬 학습 중...")
                self.logger.info("1단계: 로컬 사전학습")
                pretrain_history = self.local_pretrain()
                pbar.update(1)
                
                # 2. 공용 데이터셋 로짓 추론
                pbar.set_description(f"👤 {self.client_id} - {steps[1]}")
                pbar.set_postfix_str("로짓 추론 중...")
                self.logger.info("2단계: 공용 데이터셋 로짓 추론")
                logits_data = self.infer_public_logits()
                pbar.update(1)
                
                # 3. 로짓 업로드
                pbar.set_description(f"👤 {self.client_id} - {steps[2]}")
                pbar.set_postfix_str("업로드 중...")
                self.logger.info("3단계: 로짓 업로드")
                if not self.upload_logits(logits_data):
                    raise Exception("로짓 업로드 실패")
                pbar.update(1)
                
                # 4. 소프트 타겟 수신 대기
                pbar.set_description(f"👤 {self.client_id} - {steps[3]}")
                pbar.set_postfix_str("대기 중...")
                self.logger.info("4단계: 소프트 타겟 수신 대기")
                
                wait_pbar = tqdm(desc="⏳ 서버 처리 대기", unit="초", 
                               bar_format="{l_bar}{bar}| {elapsed}초 대기 중", 
                               position=1, leave=False)
                
                for _ in range(2):  # 2초 대기
                    time.sleep(1)
                    wait_pbar.update(1)
                
                wait_pbar.close()
                
                soft_targets = self.receive_soft_targets()
                if soft_targets is None:
                    raise Exception("소프트 타겟 수신 실패")
                pbar.update(1)
                
                # 5. 지식 증류 학습
                pbar.set_description(f"👤 {self.client_id} - {steps[4]}")
                pbar.set_postfix_str("증류 학습 중...")
                self.logger.info("5단계: 지식 증류 학습")
                distill_history = self.distill(soft_targets)
                pbar.update(1)
                
                # 6. 모델 평가
                pbar.set_description(f"👤 {self.client_id} - {steps[5]}")
                pbar.set_postfix_str("평가 중...")
                self.logger.info("6단계: 모델 평가")
                metrics = self.evaluate()
                pbar.update(1)
                
                # 7. 메트릭 보고
                pbar.set_description(f"👤 {self.client_id} - {steps[6]}")
                pbar.set_postfix_str("보고 중...")
                self.logger.info("7단계: 메트릭 보고")
                self.report_metrics(metrics)
                pbar.update(1)
                
                # 결과 저장
                round_results.update({
                    "success": True,
                    "metrics": metrics,
                    "pretrain_history": pretrain_history,
                    "distill_history": distill_history
                })
                
                # 메트릭 히스토리 저장
                self.metrics_history.append({
                    "round_id": self.current_round,
                    "metrics": metrics,
                    "timestamp": int(time.time() * 1000)
                })
                
                # 메트릭 저장
                self._save_round_metrics(round_results)
                
                pbar.set_description(f"✅ {self.client_id} 라운드 {self.current_round} 완료")
                pbar.set_postfix_str(f"정확도: {metrics.get('accuracy', 0):.3f}")
                self.logger.info(f"라운드 {self.current_round} 실행 완료")
                
            except Exception as e:
                pbar.set_description(f"❌ {self.client_id} 라운드 {self.current_round} 실패")
                pbar.set_postfix_str(f"오류: {str(e)[:20]}...")
                self.logger.error(f"라운드 {self.current_round} 실행 실패: {e}")
                round_results["error"] = str(e)
        
        return round_results
    
    def _save_round_metrics(self, round_results: Dict[str, Any]) -> None:
        """라운드 메트릭 저장"""
        metrics_file = self.artifacts_dir / f"round_{self.current_round:04d}_metrics.json"
        save_metrics(round_results, metrics_file)
        self.logger.info(f"라운드 메트릭 저장됨: {metrics_file}")
    
    def run(self, total_rounds: int = 1) -> List[Dict[str, Any]]:
        """전체 라운드 실행"""
        self.logger.info(f"클라이언트 {self.client_id} 시작: {total_rounds} 라운드")
        
        # 초기화 단계 진행률 표시
        init_steps = ["서버 등록", "데이터셋 로드", "모델 빌드"]
        
        with tqdm(total=len(init_steps), desc=f"🔧 {self.client_id} 초기화", 
                 unit="단계", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} 단계 [{elapsed}]") as init_pbar:
            
            # 서버 등록
            init_pbar.set_description(f"🔧 {self.client_id} - 서버 등록")
            init_pbar.set_postfix_str("등록 중...")
            if not self.register():
                raise Exception("서버 등록 실패")
            init_pbar.update(1)
            
            # 데이터셋 로드
            init_pbar.set_description(f"🔧 {self.client_id} - 데이터셋 로드")
            init_pbar.set_postfix_str("로딩 중...")
            if not self.load_datasets():
                raise Exception("데이터셋 로드 실패")
            init_pbar.update(1)
            
            # 모델 빌드
            init_pbar.set_description(f"🔧 {self.client_id} - 모델 빌드")
            init_pbar.set_postfix_str("빌드 중...")
            if not self.build_model():
                raise Exception("모델 빌드 실패")
            init_pbar.update(1)
        
        # 라운드 실행
        all_results = []
        
        with tqdm(total=total_rounds, desc=f"🔄 {self.client_id} 라운드 실행", 
                 unit="라운드", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} 라운드 [{elapsed}<{remaining}]") as round_pbar:
            
            for round_num in range(1, total_rounds + 1):
                self.current_round = round_num
                round_pbar.set_description(f"🔄 {self.client_id} 라운드 {round_num}")
                round_pbar.set_postfix_str("실행 중...")
                
                round_results = self.run_round()
                all_results.append(round_results)
                
                if not round_results["success"]:
                    round_pbar.set_description(f"❌ {self.client_id} 라운드 {round_num} 실패")
                    round_pbar.set_postfix_str("중단됨")
                    self.logger.error(f"라운드 {round_num} 실패, 중단")
                    break
                else:
                    metrics = round_results.get("metrics", {})
                    accuracy = metrics.get("accuracy", 0)
                    round_pbar.set_postfix_str(f"정확도: {accuracy:.3f}")
                
                round_pbar.update(1)
        
        self.logger.info(f"클라이언트 {self.client_id} 완료")
        
        return all_results


if __name__ == "__main__":
    # 테스트 실행
    config = {
        "dataset": {
            "public": {
                "name": "CIFAR10",
                "path": "data/public",
                "index_file": "public_indices.json",
                "subset_size": 100
            },
            "private": {
                "root": "data/private",
                "num_classes": 10,
                "num_samples": 100
            }
        },
        "model": {
            "backbone": "cnn_small",
            "num_classes": 10
        },
        "train": {
            "local": {"epochs": 1, "batch_size": 32, "lr": 0.001},
            "distill": {"epochs": 1, "batch_size": 32, "lr": 0.001}
        },
        "distill": {
            "temperature": 3.0,
            "alpha": 0.7
        }
    }
    
    # 클라이언트 생성 및 실행
    client = ClientNode("test_client", config)
    
    try:
        results = client.run(total_rounds=1)
        print(f"클라이언트 실행 완료: {len(results)} 라운드")
        for i, result in enumerate(results):
            print(f"라운드 {i+1}: {'성공' if result['success'] else '실패'}")
            if result['success']:
                print(f"  메트릭: {result['metrics']}")
    except Exception as e:
        print(f"클라이언트 실행 실패: {e}")
