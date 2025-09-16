"""
FedMD 클라이언트 실행 스크립트
"""
import sys
import time
import signal
import argparse
from pathlib import Path
import logging

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.config import load_config_from_cli, create_arg_parser
from core.client import ClientNode
from comms.grpc_client import FedMDGrpcClient
from core.utils import setup_logger


class FedMDClientRunner:
    """FedMD 클라이언트 실행기"""
    
    def __init__(self, config, args):
        self.config = config
        self.args = args
        self.logger = setup_logger(f"FedMDClientRunner-{args.client_id}", 
                                 level=getattr(logging, args.log_level))
        
        # 클라이언트 인스턴스
        self.client = None
        
        # 시그널 핸들러 설정
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """시그널 핸들러"""
        self.logger.info(f"시그널 {signum} 수신, 클라이언트 종료 중...")
        sys.exit(0)
    
    def start(self):
        """클라이언트 시작"""
        try:
            self.logger.info(f"FedMD 클라이언트 시작: {self.args.client_id}")
            
            # 통신 클라이언트 생성
            comm_client = None
            if self.config.comms:
                grpc_config = self.config.comms.grpc.dict()
                comm_client = FedMDGrpcClient(
                    server_address=grpc_config["address"],
                    client_id=self.args.client_id,
                    timeout_sec=self.config.comms.timeout_sec,
                    logger=self.logger
                )
                
                # 서버 연결
                if not comm_client.connect():
                    raise Exception("서버 연결 실패")
                
                self.logger.info(f"서버 연결 성공: {grpc_config['address']}")
            
            # 클라이언트 노드 생성
            self.client = ClientNode(
                client_id=self.args.client_id,
                config=self.config.dict(),
                comm_client=comm_client,
                logger=self.logger
            )
            
            # 라운드 실행
            total_rounds = self.args.rounds or self.config.rounds.total
            self.logger.info(f"총 {total_rounds} 라운드 실행 시작")
            
            results = self.client.run(total_rounds)
            
            # 결과 요약
            successful_rounds = sum(1 for r in results if r["success"])
            self.logger.info(f"클라이언트 실행 완료: {successful_rounds}/{len(results)} 라운드 성공")
            
            # 최종 메트릭 출력
            if results:
                last_result = results[-1]
                if last_result["success"] and "metrics" in last_result:
                    metrics = last_result["metrics"]
                    self.logger.info(f"최종 메트릭: {metrics}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"클라이언트 실행 중 오류: {e}")
            raise
    
    def run(self):
        """클라이언트 실행"""
        try:
            results = self.start()
            return True
        except KeyboardInterrupt:
            self.logger.info("사용자에 의해 중단됨")
            return False
        except Exception as e:
            self.logger.error(f"클라이언트 실행 실패: {e}")
            return False


def main():
    """메인 함수"""
    # 인자 파서 생성
    parser = create_arg_parser()
    parser.add_argument(
        "--client_id",
        type=str,
        required=True,
        help="클라이언트 ID"
    )
    parser.add_argument(
        "--rounds",
        type=int,
        help="실행할 라운드 수 (기본값: 설정 파일에서 로드)"
    )
    parser.add_argument(
        "--log_level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="로그 레벨"
    )
    
    # 인자 파싱
    args = parser.parse_args()
    
    try:
        # 설정 로드
        config, _ = load_config_from_cli()
        
        # 클라이언트 실행기 생성 및 실행
        runner = FedMDClientRunner(config, args)
        success = runner.run()
        
        sys.exit(0 if success else 1)
        
    except Exception as e:
        print(f"클라이언트 실행 실패: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
