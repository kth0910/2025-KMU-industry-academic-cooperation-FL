"""
FedMD 서버 구현
"""
import time
import threading
from typing import Dict, Any, List, Optional, Tuple
import logging
from collections import defaultdict
import json
from pathlib import Path
from tqdm import tqdm

from core.utils import setup_logger, save_metrics, ensure_dir
from core.aggregator import LogitsAggregator, create_aggregator
from core.public_dataset import get_public_indices


class FedMDServer:
    """FedMD 서버"""
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Args:
            config: 서버 설정
            logger: 로거
        """
        self.config = config
        self.logger = logger or setup_logger("FedMDServer")
        
        # 서버 상태
        self.current_round = 0
        self.round_status = "IDLE"  # IDLE, COLLECTING, AGGREGATING, DISPATCHING, COMPLETED
        self.start_time = None
        self.last_update_time = None
        
        # 클라이언트 관리
        self.registered_clients = {}  # {client_id: client_info}
        self.client_weights = {}  # {client_id: weight}
        
        # 로짓 수집
        self.collected_logits = defaultdict(dict)  # {round_id: {client_id: logits_data}}
        self.expected_clients = 0
        
        # 소프트 타겟
        self.soft_targets = {}  # {round_id: {global_idx: probs}}
        
        # 메트릭 수집
        self.client_metrics = defaultdict(dict)  # {round_id: {client_id: metrics}}
        
        # 앙상블러
        self.aggregator = create_aggregator(config)
        
        # 공용 인덱스
        self.public_indices = get_public_indices(config["dataset"]["public"])
        
        # 스레드 락
        self._lock = threading.Lock()
        
        # 아티팩트 디렉터리
        self.artifacts_dir = Path("artifacts")
        ensure_dir(self.artifacts_dir)
        
        self.logger.info("FedMD 서버 초기화 완료")
    
    def start_round(self, round_id: int) -> Dict[str, Any]:
        """라운드 시작"""
        with self._lock:
            self.logger.info(f"라운드 {round_id} 시작")
            
            # 라운드 상태 초기화
            self.current_round = round_id
            self.logger.info(f"서버 current_round 설정: {self.current_round}")
            self.round_status = "COLLECTING"
            self.start_time = int(time.time() * 1000)
            self.last_update_time = self.start_time
            
            # 로짓 수집 초기화
            self.collected_logits[round_id] = {}
            self.expected_clients = len(self.registered_clients)
            
            # 공용 인덱스 생성/로드
            if not self.public_indices:
                self.public_indices = get_public_indices(
                    self.config["dataset"]["public"],
                    subset_size=self.config["dataset"]["public"].get("subset_size", 512)
                )
            
            # 라운드 정보 반환
            round_info = {
                "round_id": round_id,
                "public_indices": self.public_indices,
                "temperature": self.config["distill"]["temperature"],
                "alpha": self.config["distill"]["alpha"],
                "timeout_sec": self.config.get("comms", {}).get("timeout_sec", 30),
                "expected_clients": self.expected_clients
            }
            
            self.logger.info(f"라운드 {round_id} 시작 완료: {len(self.public_indices)}개 인덱스")
            
            return round_info
    
    def register_client(self, client_info: Dict[str, Any]) -> Dict[str, Any]:
        """클라이언트 등록"""
        with self._lock:
            client_id = client_info["client_id"]
            
            if client_id in self.registered_clients:
                self.logger.warning(f"클라이언트 {client_id}가 이미 등록되어 있습니다")
                return {
                    "success": False,
                    "message": "이미 등록된 클라이언트입니다"
                }
            
            # 클라이언트 등록
            self.registered_clients[client_id] = client_info
            self.client_weights[client_id] = client_info.get("weight", 1.0)
            
            self.logger.info(f"클라이언트 {client_id} 등록 완료")
            
            # 현재 라운드 정보 반환
            if self.round_status != "IDLE":
                return {
                    "success": True,
                    "message": "등록 성공",
                    "round_id": self.current_round,
                    "public_indices": self.public_indices,
                    "temperature": self.config["distill"]["temperature"],
                    "alpha": self.config["distill"]["alpha"],
                    "timeout_sec": self.config.get("comms", {}).get("timeout_sec", 30),
                    "required_capabilities": []
                }
            else:
                # IDLE 상태에서도 기본 정보 반환
                return {
                    "success": True,
                    "message": "등록 성공, 라운드 대기 중",
                    "round_id": self.current_round,
                    "public_indices": self.public_indices if hasattr(self, 'public_indices') else [],
                    "temperature": self.config["distill"]["temperature"],
                    "alpha": self.config["distill"]["alpha"],
                    "timeout_sec": self.config.get("comms", {}).get("timeout_sec", 30),
                    "required_capabilities": []
                }
    
    def collect_logits(self, 
                      client_id: str, 
                      round_id: int, 
                      logits_data: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        """로짓 수집"""
        with self._lock:
            if round_id != self.current_round:
                return {
                    "success": False,
                    "message": f"잘못된 라운드 ID: {round_id} (현재: {self.current_round})"
                }
            
            if client_id not in self.registered_clients:
                return {
                    "success": False,
                    "message": f"등록되지 않은 클라이언트: {client_id}"
                }
            
            # 로짓 저장
            self.collected_logits[round_id][client_id] = logits_data
            self.last_update_time = int(time.time() * 1000)
            
            # 라운드 상태를 COLLECTING으로 설정 (이미 COLLECTING이 아닌 경우에만)
            if self.round_status != "COLLECTING":
                self.round_status = "COLLECTING"
            
            self.logger.debug(f"로짓 수집: {client_id}, {len(logits_data)}개 샘플")
            
            return {
                "success": True,
                "message": "로짓 수집 완료"
            }
    
    def can_aggregate(self) -> bool:
        """앙상블 가능 여부 확인"""
        with self._lock:
            self.logger.debug(f"앙상블 조건 확인 - 라운드 상태: {self.round_status}")
            if self.round_status != "COLLECTING":
                self.logger.debug("앙상블 불가: 라운드 상태가 COLLECTING이 아님")
                return False
            
            collected_count = len(self.collected_logits[self.current_round])
            self.logger.debug(f"앙상블 조건 확인 - 수집된 로짓: {collected_count}, 예상: {self.expected_clients}")
            can_agg = collected_count >= self.expected_clients
            self.logger.debug(f"앙상블 가능 여부: {can_agg}")
            return can_agg
    
    def make_soft_targets(self) -> Dict[str, Any]:
        """소프트 타겟 생성"""
        self.logger.info(f"소프트 타겟 생성 시도 - 현재 라운드: {self.current_round}")
        self.logger.info(f"수집된 로짓 수: {len(self.collected_logits.get(self.current_round, {}))}")
        self.logger.info(f"예상 클라이언트 수: {self.expected_clients}")
        
        if not self.can_aggregate():
            self.logger.warning("앙상블 조건 미충족")
            return {
                "success": False,
                "message": "앙상블 조건 미충족"
            }
        
        self.logger.info("소프트 타겟 생성 시작")
        
        # 락 없이 상태 변경
        self.round_status = "AGGREGATING"
        self.last_update_time = int(time.time() * 1000)
        
        try:
            # 로짓 앙상블
            round_logits = self.collected_logits[self.current_round]
            self.logger.info(f"라운드 {self.current_round} 로짓 수집 완료: {len(round_logits)}개 클라이언트")
            
            # 공용 인덱스 확인
            self.logger.info(f"공용 인덱스 수: {len(self.public_indices)}")
            self.logger.info(f"공용 인덱스: {self.public_indices[:5]}...")  # 처음 5개만 출력
            
            # 텐서로 변환
            client_logits = {}
            client_ids = list(round_logits.keys())
            self.logger.info(f"클라이언트 ID 목록: {client_ids}")
            
            # 클라이언트 로짓 텐서 변환
            self.logger.info(f"클라이언트 로짓 텐서 변환 시작: {len(client_ids)}개 클라이언트")
            for i, client_id in enumerate(client_ids):
                self.logger.info(f"클라이언트 {i+1}/{len(client_ids)}: {client_id}")
                
                # 로짓 텐서 생성
                logits_list = []
                client_data = round_logits[client_id]
                self.logger.info(f"클라이언트 {client_id} 데이터 키 수: {len(client_data)}")
                
                # 클래스 수 확인 (안전하게)
                if client_data:
                    first_key = next(iter(client_data.keys()))
                    num_classes = len(client_data[first_key]["logits"])
                    self.logger.info(f"클라이언트 {client_id} 클래스 수: {num_classes}")
                else:
                    num_classes = 10  # 기본값
                    self.logger.warning(f"클라이언트 {client_id} 데이터가 비어있음, 기본 클래스 수 사용: {num_classes}")
                
                self.logger.info(f"공용 인덱스 순회 시작: {len(self.public_indices)}개")
                for j, global_idx in enumerate(self.public_indices):
                    if global_idx in client_data:
                        logits_list.append(client_data[global_idx]["logits"])
                    else:
                        # 누락된 로짓은 0으로 채움
                        logits_list.append([0.0] * num_classes)
                    
                    if j % 5 == 0:  # 5개마다 진행 상황 출력
                        self.logger.info(f"인덱스 처리 진행: {j+1}/{len(self.public_indices)}")
                
                # 텐서로 변환
                self.logger.info(f"텐서 변환 시작: {len(logits_list)}개 로짓")
                import torch
                client_logits[client_id] = torch.tensor(logits_list)
                self.logger.info(f"클라이언트 {client_id} 텐서 변환 완료: {client_logits[client_id].shape}")
            
            self.logger.info("모든 클라이언트 로짓 텐서 변환 완료")
            
            # 앙상블 수행
            self.logger.info(f"앙상블 수행 시작 - 클라이언트 수: {len(client_logits)}")
            soft_targets_tensor = self.aggregator.aggregate(
                client_logits, 
                self.client_weights
            )
            self.logger.info(f"앙상블 완료 - 결과 텐서 형태: {soft_targets_tensor.shape}")
            
            # 딕셔너리로 변환
            soft_targets = {}
            self.logger.info(f"소프트 타겟 딕셔너리 변환 시작: {len(self.public_indices)}개 인덱스")
            for i, global_idx in enumerate(self.public_indices):
                soft_targets[global_idx] = soft_targets_tensor[i].tolist()
                if i % 20 == 0:  # 20개마다 로그 출력
                    self.logger.info(f"변환 진행: {i+1}/{len(self.public_indices)}")
            
            self.logger.info("소프트 타겟 딕셔너리 변환 완료")
            
            # 저장
            self.soft_targets[self.current_round] = soft_targets
            
            # 라운드 상태 변경
            self.round_status = "DISPATCHING"
            self.last_update_time = int(time.time() * 1000)
            
            # 소프트 타겟이 생성되면 자동으로 배포된 것으로 간주
            self.logger.info("소프트 타겟 배포 완료")
            
            self.logger.info(f"소프트 타겟 생성 완료: {len(soft_targets)}개")
            
            return {
                "success": True,
                "soft_targets": soft_targets,
                "round_id": self.current_round,
                "num_clients": len(round_logits),
                "checksum": ""  # 체크섬 계산 생략
            }
            
        except Exception as e:
            self.logger.error(f"소프트 타겟 생성 실패: {e}")
            self.round_status = "COLLECTING"  # 상태 복원
            return {
                "success": False,
                "message": f"소프트 타겟 생성 실패: {str(e)}"
            }
    
    def get_soft_targets(self, round_id: int) -> Dict[str, Any]:
        """소프트 타겟 조회"""
        with self._lock:
            if round_id not in self.soft_targets:
                return {
                    "success": False,
                    "message": f"라운드 {round_id}의 소프트 타겟이 없습니다"
                }
            
            return {
                "success": True,
                "soft_targets": self.soft_targets[round_id],
                "round_id": round_id,
                "num_clients": len(self.collected_logits.get(round_id, {})),
                "checksum": ""
            }
    
    def report_metrics(self, 
                      client_id: str, 
                      round_id: int, 
                      metrics: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """메트릭 보고"""
        with self._lock:
            if client_id not in self.registered_clients:
                return {
                    "success": False,
                    "message": f"등록되지 않은 클라이언트: {client_id}"
                }
            
            # 메트릭 저장
            self.client_metrics[round_id][client_id] = metrics
            self.last_update_time = int(time.time() * 1000)
            
            self.logger.debug(f"메트릭 보고: {client_id}, 라운드 {round_id}")
            
            return {
                "success": True,
                "message": "메트릭 보고 완료"
            }
    
    def get_round_status(self, round_id: int) -> Dict[str, Any]:
        """라운드 상태 조회"""
        with self._lock:
            if round_id != self.current_round:
                return {
                    "round_id": round_id,
                    "status": "NOT_FOUND",
                    "num_clients_registered": 0,
                    "num_logits_received": 0,
                    "expected_clients": 0,
                    "start_time": 0,
                    "last_update_time": 0,
                    "registered_clients": []
                }
            
            return {
                "round_id": round_id,
                "status": self.round_status,
                "num_clients_registered": len(self.registered_clients),
                "num_logits_received": len(self.collected_logits.get(round_id, {})),
                "expected_clients": self.expected_clients,
                "start_time": self.start_time or 0,
                "last_update_time": self.last_update_time or 0,
                "registered_clients": list(self.registered_clients.keys())
            }
    
    def get_health_status(self) -> Dict[str, Any]:
        """헬스 상태 조회"""
        with self._lock:
            return {
                "healthy": True,
                "status": "OK",
                "version": "1.0.0",
                "active_rounds": 1 if self.round_status != "IDLE" else 0,
                "connected_clients": len(self.registered_clients)
            }
    
    def complete_round(self) -> None:
        """라운드 완료"""
        with self._lock:
            self.round_status = "COMPLETED"
            self.last_update_time = int(time.time() * 1000)
            
            # 라운드 메트릭 저장
            self._save_round_metrics()
            
            self.logger.info(f"라운드 {self.current_round} 완료")
    
    def _save_round_metrics(self) -> None:
        """라운드 메트릭 저장"""
        if self.current_round not in self.client_metrics:
            return
        
        round_metrics = {
            "round_id": self.current_round,
            "timestamp": self.last_update_time,
            "client_metrics": dict(self.client_metrics[self.current_round]),
            "server_status": {
                "status": self.round_status,
                "num_clients": len(self.registered_clients),
                "num_logits_collected": len(self.collected_logits.get(self.current_round, {})),
                "public_indices_count": len(self.public_indices)
            }
        }
        
        # 파일 저장
        metrics_file = self.artifacts_dir / f"round_{self.current_round:04d}_metrics.json"
        save_metrics(round_metrics, metrics_file)
        
        self.logger.info(f"라운드 메트릭 저장됨: {metrics_file}")
    
    def run(self, total_rounds: int) -> None:
        """전체 라운드 실행"""
        self.logger.info(f"FedMD 서버 시작: {total_rounds} 라운드")
        
        # 전체 라운드 진행률 표시
        with tqdm(total=total_rounds, desc="🚀 FedMD 서버", unit="라운드", 
                 bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} 라운드 [{elapsed}<{remaining}]") as pbar:
            
            for round_id in range(1, total_rounds + 1):
                pbar.set_description(f"🚀 라운드 {round_id} 진행 중")
                self.logger.info(f"=== 라운드 {round_id} 시작 ===")
                
                # 라운드 시작
                round_info = self.start_round(round_id)
                
                # 로짓 수집 대기 (진행률 표시)
                pbar.set_postfix_str("로짓 수집 대기 중...")
                self.logger.info("로짓 수집 대기 중...")
                
                collect_pbar = tqdm(desc="📥 로짓 수집", unit="초", 
                                  bar_format="{l_bar}{bar}| {elapsed}초 대기 중", 
                                  position=1, leave=False)
                
                while not self.can_aggregate():
                    time.sleep(1)
                    collect_pbar.update(1)
                
                collect_pbar.close()
                
                # 소프트 타겟 생성
                pbar.set_postfix_str("소프트 타겟 생성 중...")
                soft_targets_result = self.make_soft_targets()
                if not soft_targets_result["success"]:
                    self.logger.error(f"라운드 {round_id} 소프트 타겟 생성 실패")
                    pbar.update(1)
                    continue
                
                # 소프트 타겟 배포 대기 (클라이언트들이 가져갈 때까지)
                pbar.set_postfix_str("클라이언트 처리 대기 중...")
                self.logger.info("소프트 타겟 배포 완료, 클라이언트 처리 대기 중...")
                
                wait_pbar = tqdm(desc="⏳ 클라이언트 대기", unit="초", 
                               bar_format="{l_bar}{bar}| {elapsed}초 대기 중", 
                               position=1, leave=False)
                
                for _ in range(5):  # 5초 대기
                    time.sleep(1)
                    wait_pbar.update(1)
                
                wait_pbar.close()
                
                # 라운드 완료
                self.complete_round()
                
                pbar.set_postfix_str(f"라운드 {round_id} 완료")
                self.logger.info(f"=== 라운드 {round_id} 완료 ===")
                pbar.update(1)
        
        self.logger.info("모든 라운드 완료")


if __name__ == "__main__":
    # 테스트 실행
    config = {
        "dataset": {
            "public": {
                "name": "CIFAR10",
                "path": "data/public",
                "index_file": "public_indices.json",
                "subset_size": 100
            }
        },
        "distill": {
            "temperature": 3.0,
            "alpha": 0.7
        },
        "comms": {
            "timeout_sec": 30
        }
    }
    
    server = FedMDServer(config)
    
    # 클라이언트 등록 테스트
    client_info = {
        "client_id": "test_client",
        "model_name": "cnn_small",
        "weight": 1.0
    }
    
    result = server.register_client(client_info)
    print(f"클라이언트 등록: {result}")
    
    # 라운드 시작 테스트
    round_info = server.start_round(1)
    print(f"라운드 시작: {round_info}")
    
    # 로짓 수집 테스트
    logits_data = {
        0: {"logits": [0.1, 0.9], "confidence": 0.8},
        1: {"logits": [0.2, 0.8], "confidence": 0.7}
    }
    
    result = server.collect_logits("test_client", 1, logits_data)
    print(f"로짓 수집: {result}")
    
    # 소프트 타겟 생성 테스트
    soft_targets = server.make_soft_targets()
    print(f"소프트 타겟: {soft_targets}")
    
    # 상태 조회
    status = server.get_round_status(1)
    print(f"라운드 상태: {status}")
