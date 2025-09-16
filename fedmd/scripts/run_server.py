"""
FedMD 서버 실행 스크립트
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
from core.server import FedMDServer
from comms.grpc_server import FedMDGrpcServer
from core.utils import setup_logger


class FedMDServerRunner:
    """FedMD 서버 실행기"""
    
    def __init__(self, config, args):
        self.config = config
        self.args = args
        self.logger = setup_logger("FedMDServerRunner", level=getattr(logging, args.log_level))
        
        # 서버 인스턴스
        self.fedmd_server = None
        self.grpc_server = None
        
        # 시그널 핸들러 설정
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """시그널 핸들러"""
        self.logger.info(f"시그널 {signum} 수신, 서버 종료 중...")
        self.stop()
        sys.exit(0)
    
    def start(self):
        """서버 시작"""
        try:
            self.logger.info("FedMD 서버 시작")
            
            # FedMD 서버 생성
            self.fedmd_server = FedMDServer(self.config.dict(), self.logger)
            
            # gRPC 서버 생성
            if self.config.comms:
                grpc_config = self.config.comms.grpc.dict()
                self.grpc_server = FedMDGrpcServer(
                    self.fedmd_server, 
                    grpc_config, 
                    self.logger
                )
                
                # gRPC 서버 시작
                if not self.grpc_server.start():
                    raise Exception("gRPC 서버 시작 실패")
                
                self.logger.info(f"gRPC 서버 시작됨: {grpc_config['address']}")
            
            # 라운드 실행
            total_rounds = self.args.rounds or self.config.rounds.total
            self.logger.info(f"총 {total_rounds} 라운드 실행 시작")
            
            self.fedmd_server.run(total_rounds)
            
            self.logger.info("모든 라운드 완료")
            
        except Exception as e:
            self.logger.error(f"서버 실행 중 오류: {e}")
            raise
    
    def stop(self):
        """서버 중지"""
        if self.grpc_server:
            self.grpc_server.stop()
            self.logger.info("gRPC 서버 중지됨")
        
        self.logger.info("FedMD 서버 중지됨")
    
    def run(self):
        """서버 실행"""
        try:
            self.start()
        except KeyboardInterrupt:
            self.logger.info("사용자에 의해 중단됨")
        except Exception as e:
            self.logger.error(f"서버 실행 실패: {e}")
            return False
        finally:
            self.stop()
        
        return True


def main():
    """메인 함수"""
    # 인자 파서 생성
    parser = create_arg_parser()
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
        
        # 서버 실행기 생성 및 실행
        runner = FedMDServerRunner(config, args)
        success = runner.run()
        
        sys.exit(0 if success else 1)
        
    except Exception as e:
        print(f"서버 실행 실패: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
