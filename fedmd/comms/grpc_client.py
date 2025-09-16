"""
gRPC 클라이언트 어댑터
ClientNode와 gRPC 서비스를 연결
"""
import grpc
import time
from typing import Dict, Any, List, Optional, Tuple
import logging
import threading

import sys
import os
sys.path.append(os.path.dirname(__file__))
import fedmd_pb2
import fedmd_pb2_grpc
from core.utils import setup_logger


class FedMDGrpcClient:
    """FedMD gRPC 클라이언트"""
    
    def __init__(self, 
                 server_address: str,
                 client_id: str,
                 timeout_sec: int = 30,
                 logger: Optional[logging.Logger] = None):
        """
        Args:
            server_address: 서버 주소 (예: "127.0.0.1:50051")
            client_id: 클라이언트 ID
            timeout_sec: 타임아웃 (초)
            logger: 로거
        """
        self.server_address = server_address
        self.client_id = client_id
        self.timeout_sec = timeout_sec
        self.logger = logger or setup_logger(f"FedMDGrpcClient-{client_id}")
        
        # gRPC 채널 및 스텁
        self.channel = None
        self.stub = None
        self._lock = threading.Lock()
        
        # 연결 상태
        self._connected = False
        self._last_round_id = 0
    
    def connect(self) -> bool:
        """서버에 연결"""
        try:
            with self._lock:
                if self._connected:
                    return True
                
                # gRPC 채널 생성
                options = [
                    ('grpc.max_send_message_length', 100 * 1024 * 1024),  # 100MB
                    ('grpc.max_receive_message_length', 100 * 1024 * 1024),  # 100MB
                    ('grpc.keepalive_time_ms', 30000),
                    ('grpc.keepalive_timeout_ms', 5000),
                    ('grpc.keepalive_permit_without_calls', True),
                    ('grpc.http2.max_pings_without_data', 0),
                    ('grpc.http2.min_time_between_pings_ms', 10000),
                    ('grpc.http2.min_ping_interval_without_data_ms', 300000)
                ]
                
                self.channel = grpc.insecure_channel(self.server_address, options=options)
                self.stub = fedmd_pb2_grpc.FedMDStub(self.channel)
                
                # 연결 테스트
                self._test_connection()
                
                self._connected = True
                self.logger.info(f"서버 연결 성공: {self.server_address}")
                return True
                
        except Exception as e:
            self.logger.error(f"서버 연결 실패: {e}")
            self._connected = False
            return False
    
    def disconnect(self):
        """서버 연결 해제"""
        with self._lock:
            if self.channel:
                self.channel.close()
                self.channel = None
                self.stub = None
                self._connected = False
                self.logger.info("서버 연결 해제됨")
    
    def _test_connection(self):
        """연결 테스트"""
        try:
            # 헬스 체크로 연결 테스트
            request = fedmd_pb2.Empty()
            response = self.stub.HealthCheck(request, timeout=self.timeout_sec)
            
            if not response.healthy:
                raise Exception(f"서버가 비정상 상태: {response.status}")
                
        except Exception as e:
            raise Exception(f"연결 테스트 실패: {e}")
    
    def register(self, 
                model_name: str = "cnn_small",
                version: str = "1.0.0",
                num_classes: int = 10,
                capabilities: List[str] = None) -> Dict[str, Any]:
        """
        서버에 클라이언트 등록
        
        Returns:
            등록 결과 및 서버 정보
        """
        if not self._connected:
            if not self.connect():
                return {"success": False, "message": "서버 연결 실패"}
        
        try:
            # 등록 요청 생성
            request = fedmd_pb2.ClientHello(
                client_id=self.client_id,
                model_name=model_name,
                version=version,
                num_classes=num_classes,
                capabilities=capabilities or []
            )
            
            # 등록 요청 전송
            response = self.stub.Register(request, timeout=self.timeout_sec)
            
            if response.success:
                self._last_round_id = response.round_id
                self.logger.info(f"서버 등록 성공: 라운드 {response.round_id}")
                
                return {
                    "success": True,
                    "message": response.message,
                    "round_id": response.round_id,
                    "public_indices": list(response.public_indices),
                    "temperature": response.temperature,
                    "alpha": response.alpha,
                    "timeout_sec": response.timeout_sec,
                    "required_capabilities": list(response.required_capabilities)
                }
            else:
                self.logger.warning(f"서버 등록 실패: {response.message}")
                return {
                    "success": False,
                    "message": response.message
                }
                
        except grpc.RpcError as e:
            self.logger.error(f"등록 중 gRPC 오류: {e.code()} - {e.details()}")
            return {
                "success": False,
                "message": f"gRPC 오류: {e.details()}"
            }
        except Exception as e:
            self.logger.error(f"등록 중 오류: {e}")
            return {
                "success": False,
                "message": f"등록 오류: {str(e)}"
            }
    
    def upload_logits(self, 
                     round_id: int,
                     logits_data: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        """
        로짓 업로드
        
        Args:
            round_id: 라운드 ID
            logits_data: {global_idx: {"logits": [...], "confidence": float, "timestamp": int}}
        
        Returns:
            업로드 결과
        """
        if not self._connected:
            return {"success": False, "message": "서버에 연결되지 않음"}
        
        try:
            # 로짓 행 생성
            logit_rows = []
            for global_idx, data in logits_data.items():
                logit_row = fedmd_pb2.LogitRow(
                    global_idx=global_idx,
                    logits=data["logits"],
                    confidence=data.get("confidence", 1.0),
                    timestamp=data.get("timestamp", int(time.time() * 1000))
                )
                logit_rows.append(logit_row)
            
            # 업로드 요청 생성
            request = fedmd_pb2.LogitsBatch(
                client_id=self.client_id,
                round_id=round_id,
                rows=logit_rows,
                batch_timestamp=int(time.time() * 1000),
                checksum=""  # 체크섬 계산 생략
            )
            
            # 업로드 요청 전송
            response = self.stub.UploadLogits(request, timeout=self.timeout_sec)
            
            if response.success:
                self.logger.debug(f"로짓 업로드 성공: {len(logit_rows)}개 행")
                return {
                    "success": True,
                    "message": response.message,
                    "timestamp": response.timestamp
                }
            else:
                self.logger.warning(f"로짓 업로드 실패: {response.message}")
                return {
                    "success": False,
                    "message": response.message
                }
                
        except grpc.RpcError as e:
            self.logger.error(f"로짓 업로드 중 gRPC 오류: {e.code()} - {e.details()}")
            return {
                "success": False,
                "message": f"gRPC 오류: {e.details()}"
            }
        except Exception as e:
            self.logger.error(f"로짓 업로드 중 오류: {e}")
            return {
                "success": False,
                "message": f"업로드 오류: {str(e)}"
            }
    
    def get_soft_targets(self, round_id: int) -> Dict[str, Any]:
        """
        소프트 타겟 수신
        
        Args:
            round_id: 라운드 ID
        
        Returns:
            소프트 타겟 데이터
        """
        if not self._connected:
            return {"success": False, "message": "서버에 연결되지 않음"}
        
        try:
            # 소프트 타겟 요청 생성
            request = fedmd_pb2.RoundQuery(
                round_id=round_id,
                client_id=self.client_id
            )
            
            # 요청 전송
            response = self.stub.GetSoftTargets(request, timeout=self.timeout_sec)
            
            if response.round_id == round_id and response.rows:
                # 소프트 타겟 데이터 변환
                soft_targets = {}
                for row in response.rows:
                    soft_targets[row.global_idx] = list(row.probs)
                
                self.logger.debug(f"소프트 타겟 수신 완료: {len(soft_targets)}개")
                
                return {
                    "success": True,
                    "soft_targets": soft_targets,
                    "round_id": response.round_id,
                    "num_clients": response.num_clients,
                    "timestamp": response.batch_timestamp
                }
            else:
                self.logger.warning(f"소프트 타겟 수신 실패: 라운드 {round_id}")
                return {
                    "success": False,
                    "message": "소프트 타겟을 받을 수 없습니다"
                }
                
        except grpc.RpcError as e:
            self.logger.error(f"소프트 타겟 요청 중 gRPC 오류: {e.code()} - {e.details()}")
            return {
                "success": False,
                "message": f"gRPC 오류: {e.details()}"
            }
        except Exception as e:
            self.logger.error(f"소프트 타겟 요청 중 오류: {e}")
            return {
                "success": False,
                "message": f"요청 오류: {str(e)}"
            }
    
    def report_metrics(self, 
                      round_id: int,
                      metrics: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        메트릭 보고
        
        Args:
            round_id: 라운드 ID
            metrics: {metric_name: {"value": float, "unit": str, "timestamp": int}}
        
        Returns:
            보고 결과
        """
        if not self._connected:
            return {"success": False, "message": "서버에 연결되지 않음"}
        
        try:
            # 메트릭 항목 생성
            metric_items = []
            for name, data in metrics.items():
                metric = fedmd_pb2.Metric(
                    name=name,
                    value=data["value"],
                    unit=data.get("unit", ""),
                    timestamp=data.get("timestamp", int(time.time() * 1000))
                )
                metric_items.append(metric)
            
            # 보고 요청 생성
            request = fedmd_pb2.MetricsReport(
                client_id=self.client_id,
                round_id=round_id,
                metrics=metric_items,
                report_timestamp=int(time.time() * 1000)
            )
            
            # 보고 요청 전송
            response = self.stub.ReportMetrics(request, timeout=self.timeout_sec)
            
            if response.success:
                self.logger.debug(f"메트릭 보고 성공: {len(metric_items)}개 항목")
                return {
                    "success": True,
                    "message": response.message,
                    "timestamp": response.timestamp
                }
            else:
                self.logger.warning(f"메트릭 보고 실패: {response.message}")
                return {
                    "success": False,
                    "message": response.message
                }
                
        except grpc.RpcError as e:
            self.logger.error(f"메트릭 보고 중 gRPC 오류: {e.code()} - {e.details()}")
            return {
                "success": False,
                "message": f"gRPC 오류: {e.details()}"
            }
        except Exception as e:
            self.logger.error(f"메트릭 보고 중 오류: {e}")
            return {
                "success": False,
                "message": f"보고 오류: {str(e)}"
            }
    
    def get_round_status(self, round_id: int) -> Dict[str, Any]:
        """라운드 상태 조회"""
        if not self._connected:
            return {"success": False, "message": "서버에 연결되지 않음"}
        
        try:
            request = fedmd_pb2.RoundQuery(
                round_id=round_id,
                client_id=self.client_id
            )
            
            response = self.stub.GetRoundStatus(request, timeout=self.timeout_sec)
            
            return {
                "success": True,
                "round_id": response.round_id,
                "status": response.status,
                "num_clients_registered": response.num_clients_registered,
                "num_logits_received": response.num_logits_received,
                "expected_clients": response.expected_clients,
                "start_time": response.start_time,
                "last_update_time": response.last_update_time,
                "registered_clients": list(response.registered_clients)
            }
            
        except Exception as e:
            self.logger.error(f"라운드 상태 조회 중 오류: {e}")
            return {
                "success": False,
                "message": f"상태 조회 오류: {str(e)}"
            }
    
    def health_check(self) -> Dict[str, Any]:
        """헬스 체크"""
        if not self._connected:
            return {"success": False, "message": "서버에 연결되지 않음"}
        
        try:
            request = fedmd_pb2.Empty()
            response = self.stub.HealthCheck(request, timeout=self.timeout_sec)
            
            return {
                "success": True,
                "healthy": response.healthy,
                "status": response.status,
                "timestamp": response.timestamp,
                "version": response.version,
                "active_rounds": response.active_rounds,
                "connected_clients": response.connected_clients
            }
            
        except Exception as e:
            self.logger.error(f"헬스 체크 중 오류: {e}")
            return {
                "success": False,
                "message": f"헬스 체크 오류: {str(e)}"
            }
    
    def is_connected(self) -> bool:
        """연결 상태 확인"""
        return self._connected
    
    def get_last_round_id(self) -> int:
        """마지막 라운드 ID 반환"""
        return self._last_round_id


if __name__ == "__main__":
    # 테스트 실행
    client = FedMDGrpcClient("127.0.0.1:50051", "test_client")
    
    # 연결 테스트
    if client.connect():
        print("서버 연결 성공")
        
        # 헬스 체크
        health = client.health_check()
        print(f"헬스 체크: {health}")
        
        # 등록 테스트
        reg_result = client.register()
        print(f"등록 결과: {reg_result}")
        
        # 연결 해제
        client.disconnect()
    else:
        print("서버 연결 실패")
