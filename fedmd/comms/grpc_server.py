"""
gRPC 서버 어댑터
FedMDServer와 gRPC 서비스를 연결
"""
import grpc
from concurrent import futures
import threading
import time
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime

import sys
import os
sys.path.append(os.path.dirname(__file__))
import fedmd_pb2
import fedmd_pb2_grpc
from core.utils import setup_logger


class FedMDGrpcService(fedmd_pb2_grpc.FedMDServicer):
    """FedMD gRPC 서비스 구현"""
    
    def __init__(self, fedmd_server, logger: Optional[logging.Logger] = None):
        self.fedmd_server = fedmd_server
        self.logger = logger or setup_logger("FedMDGrpcService")
        self._lock = threading.Lock()
    
    def Register(self, request, context):
        """클라이언트 등록 처리"""
        try:
            self.logger.info(f"클라이언트 등록 요청: {request.client_id}")
            
            # FedMDServer에 등록
            client_info = {
                "client_id": request.client_id,
                "model_name": request.model_name,
                "version": request.version,
                "num_classes": request.num_classes,
                "capabilities": list(request.capabilities)
            }
            
            result = self.fedmd_server.register_client(client_info)
            
            if result["success"]:
                # 서버 응답 구성
                response = fedmd_pb2.ServerHello(
                    success=True,
                    message="등록 성공",
                    round_id=result.get("round_id", 0),
                    public_indices=result.get("public_indices", []),
                    temperature=result.get("temperature", 3.0),
                    alpha=result.get("alpha", 0.7),
                    timeout_sec=result.get("timeout_sec", 30),
                    required_capabilities=result.get("required_capabilities", [])
                )
                
                self.logger.info(f"클라이언트 {request.client_id} 등록 완료")
                return response
            else:
                self.logger.warning(f"클라이언트 {request.client_id} 등록 실패: {result.get('message', '알 수 없는 오류')}")
                return fedmd_pb2.ServerHello(
                    success=False,
                    message=result.get("message", "등록 실패")
                )
                
        except Exception as e:
            self.logger.error(f"클라이언트 등록 중 오류: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"서버 내부 오류: {str(e)}")
            return fedmd_pb2.ServerHello(success=False, message=f"서버 오류: {str(e)}")
    
    def UploadLogits(self, request, context):
        """로짓 업로드 처리"""
        try:
            self.logger.debug(f"로짓 업로드 요청: {request.client_id}, 라운드 {request.round_id}")
            
            # 로짓 데이터 변환
            logits_data = {}
            for row in request.rows:
                logits_data[row.global_idx] = {
                    "logits": list(row.logits),
                    "confidence": row.confidence,
                    "timestamp": row.timestamp
                }
            
            # FedMDServer에 로짓 전달
            result = self.fedmd_server.collect_logits(
                request.client_id,
                request.round_id,
                logits_data
            )
            
            if result["success"]:
                self.logger.debug(f"로짓 업로드 성공: {request.client_id}")
                return fedmd_pb2.Ack(
                    success=True,
                    message="로짓 업로드 성공",
                    timestamp=int(time.time() * 1000),
                    metadata={"round_id": str(request.round_id)}
                )
            else:
                self.logger.warning(f"로짓 업로드 실패: {request.client_id} - {result.get('message', '알 수 없는 오류')}")
                return fedmd_pb2.Ack(
                    success=False,
                    message=result.get("message", "로짓 업로드 실패"),
                    timestamp=int(time.time() * 1000)
                )
                
        except Exception as e:
            self.logger.error(f"로짓 업로드 중 오류: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"서버 내부 오류: {str(e)}")
            return fedmd_pb2.Ack(success=False, message=f"서버 오류: {str(e)}")
    
    def GetSoftTargets(self, request, context):
        """소프트 타겟 요청 처리"""
        try:
            self.logger.debug(f"소프트 타겟 요청: {request.client_id}, 라운드 {request.round_id}")
            
            # FedMDServer에서 소프트 타겟 가져오기
            result = self.fedmd_server.get_soft_targets(request.round_id)
            
            if result["success"]:
                # 소프트 타겟 데이터 변환
                prob_rows = []
                for global_idx, probs in result["soft_targets"].items():
                    prob_row = fedmd_pb2.ProbRow(
                        global_idx=global_idx,
                        probs=list(probs),
                        entropy=float(-sum(p * (p.log() if p > 0 else -float('inf')) for p in probs)),
                        timestamp=int(time.time() * 1000)
                    )
                    prob_rows.append(prob_row)
                
                response = fedmd_pb2.SoftTargetsBatch(
                    round_id=request.round_id,
                    rows=prob_rows,
                    batch_timestamp=int(time.time() * 1000),
                    checksum=result.get("checksum", ""),
                    num_clients=result.get("num_clients", 0)
                )
                
                self.logger.debug(f"소프트 타겟 전송 완료: {len(prob_rows)}개")
                return response
            else:
                self.logger.warning(f"소프트 타겟 요청 실패: {result.get('message', '알 수 없는 오류')}")
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details(result.get("message", "소프트 타겟을 찾을 수 없습니다"))
                return fedmd_pb2.SoftTargetsBatch(round_id=request.round_id)
                
        except Exception as e:
            self.logger.error(f"소프트 타겟 요청 중 오류: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"서버 내부 오류: {str(e)}")
            return fedmd_pb2.SoftTargetsBatch(round_id=request.round_id)
    
    def ReportMetrics(self, request, context):
        """메트릭 보고 처리"""
        try:
            self.logger.debug(f"메트릭 보고: {request.client_id}, 라운드 {request.round_id}")
            
            # 메트릭 데이터 변환
            metrics_data = {}
            for metric in request.metrics:
                metrics_data[metric.name] = {
                    "value": metric.value,
                    "unit": metric.unit,
                    "timestamp": metric.timestamp
                }
            
            # FedMDServer에 메트릭 전달
            result = self.fedmd_server.report_metrics(
                request.client_id,
                request.round_id,
                metrics_data
            )
            
            if result["success"]:
                self.logger.debug(f"메트릭 보고 성공: {request.client_id}")
                return fedmd_pb2.Ack(
                    success=True,
                    message="메트릭 보고 성공",
                    timestamp=int(time.time() * 1000)
                )
            else:
                self.logger.warning(f"메트릭 보고 실패: {request.client_id} - {result.get('message', '알 수 없는 오류')}")
                return fedmd_pb2.Ack(
                    success=False,
                    message=result.get("message", "메트릭 보고 실패"),
                    timestamp=int(time.time() * 1000)
                )
                
        except Exception as e:
            self.logger.error(f"메트릭 보고 중 오류: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"서버 내부 오류: {str(e)}")
            return fedmd_pb2.Ack(success=False, message=f"서버 오류: {str(e)}")
    
    def GetRoundStatus(self, request, context):
        """라운드 상태 조회"""
        try:
            status = self.fedmd_server.get_round_status(request.round_id)
            
            response = fedmd_pb2.RoundStatus(
                round_id=request.round_id,
                status=status.get("status", "UNKNOWN"),
                num_clients_registered=status.get("num_clients_registered", 0),
                num_logits_received=status.get("num_logits_received", 0),
                expected_clients=status.get("expected_clients", 0),
                start_time=status.get("start_time", 0),
                last_update_time=status.get("last_update_time", 0),
                registered_clients=status.get("registered_clients", [])
            )
            
            return response
            
        except Exception as e:
            self.logger.error(f"라운드 상태 조회 중 오류: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"서버 내부 오류: {str(e)}")
            return fedmd_pb2.RoundStatus(round_id=request.round_id)
    
    def HealthCheck(self, request, context):
        """헬스 체크"""
        try:
            health_status = self.fedmd_server.get_health_status()
            
            response = fedmd_pb2.HealthStatus(
                healthy=health_status.get("healthy", True),
                status=health_status.get("status", "OK"),
                timestamp=int(time.time() * 1000),
                version=health_status.get("version", "1.0.0"),
                active_rounds=health_status.get("active_rounds", 0),
                connected_clients=health_status.get("connected_clients", 0)
            )
            
            return response
            
        except Exception as e:
            self.logger.error(f"헬스 체크 중 오류: {e}")
            return fedmd_pb2.HealthStatus(
                healthy=False,
                status=f"ERROR: {str(e)}",
                timestamp=int(time.time() * 1000)
            )


class FedMDGrpcServer:
    """FedMD gRPC 서버"""
    
    def __init__(self, fedmd_server, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        self.fedmd_server = fedmd_server
        self.config = config
        self.logger = logger or setup_logger("FedMDGrpcServer")
        
        # gRPC 설정
        self.address = config.get("address", "127.0.0.1:50051")
        self.use_tls = config.get("use_tls", False)
        self.max_workers = config.get("max_workers", 10)
        
        # 서버 인스턴스
        self.server = None
        self.grpc_service = None
    
    def start(self):
        """gRPC 서버 시작"""
        try:
            # 서버 옵션 설정
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
            
            # 서버 생성
            self.server = grpc.server(
                futures.ThreadPoolExecutor(max_workers=self.max_workers),
                options=options
            )
            
            # 서비스 등록
            self.grpc_service = FedMDGrpcService(self.fedmd_server, self.logger)
            fedmd_pb2_grpc.add_FedMDServicer_to_server(self.grpc_service, self.server)
            
            # 포트 바인딩
            if self.use_tls:
                # TLS 설정 (구현 필요)
                self.logger.warning("TLS는 아직 구현되지 않았습니다. 일반 연결을 사용합니다.")
            
            self.server.add_insecure_port(self.address)
            
            # 서버 시작
            self.server.start()
            self.logger.info(f"gRPC 서버 시작됨: {self.address}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"gRPC 서버 시작 실패: {e}")
            return False
    
    def stop(self, grace_period: float = 5.0):
        """gRPC 서버 중지"""
        if self.server:
            self.logger.info("gRPC 서버 중지 중...")
            self.server.stop(grace_period)
            self.logger.info("gRPC 서버 중지 완료")
    
    def wait_for_termination(self):
        """서버 종료 대기"""
        if self.server:
            self.server.wait_for_termination()
    
    def is_running(self) -> bool:
        """서버 실행 상태 확인"""
        return self.server is not None


if __name__ == "__main__":
    # 테스트 실행
    from ..core.server import FedMDServer
    from ..core.config import load_config_from_cli
    
    # 설정 로드
    config, _ = load_config_from_cli()
    
    # FedMD 서버 생성 (모의)
    class MockFedMDServer:
        def register_client(self, client_info):
            return {"success": True, "round_id": 1, "public_indices": [1, 2, 3]}
        
        def collect_logits(self, client_id, round_id, logits):
            return {"success": True}
        
        def get_soft_targets(self, round_id):
            return {"success": True, "soft_targets": {1: [0.1, 0.9]}}
        
        def report_metrics(self, client_id, round_id, metrics):
            return {"success": True}
        
        def get_round_status(self, round_id):
            return {"status": "COLLECTING", "num_clients_registered": 1}
        
        def get_health_status(self):
            return {"healthy": True, "status": "OK"}
    
    # gRPC 서버 생성 및 시작
    mock_server = MockFedMDServer()
    grpc_config = config.comms.grpc.dict() if config.comms else {"address": "127.0.0.1:50051"}
    
    grpc_server = FedMDGrpcServer(mock_server, grpc_config)
    
    if grpc_server.start():
        print("gRPC 서버가 시작되었습니다. Ctrl+C로 중지하세요.")
        try:
            grpc_server.wait_for_termination()
        except KeyboardInterrupt:
            grpc_server.stop()
    else:
        print("gRPC 서버 시작 실패")
