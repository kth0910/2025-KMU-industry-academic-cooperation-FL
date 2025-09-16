"""
gRPC 프로토콜 스텁 생성 스크립트
"""
import subprocess
import sys
from pathlib import Path
import logging

from ..core.utils import setup_logger


def generate_grpc_stubs():
    """gRPC 스텁 생성"""
    logger = setup_logger("ProtoGenerator")
    
    # 경로 설정
    project_root = Path(__file__).parent.parent
    proto_dir = project_root / "comms" / "proto"
    output_dir = project_root / "comms"
    
    proto_file = proto_dir / "fedmd.proto"
    
    if not proto_file.exists():
        logger.error(f"프로토콜 파일을 찾을 수 없습니다: {proto_file}")
        return False
    
    # gRPC 도구 명령어
    cmd = [
        sys.executable, "-m", "grpc_tools.protoc",
        f"-I{proto_dir}",
        f"--python_out={output_dir}",
        f"--grpc_python_out={output_dir}",
        str(proto_file)
    ]
    
    logger.info(f"gRPC 스텁 생성 중: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info("gRPC 스텁 생성 완료")
        logger.debug(f"출력: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"gRPC 스텁 생성 실패: {e}")
        logger.error(f"오류 출력: {e.stderr}")
        return False


def create_init_files():
    """__init__.py 파일 생성"""
    logger = setup_logger("InitFileCreator")
    
    comms_dir = Path(__file__).parent.parent / "comms"
    
    # comms/__init__.py
    comms_init = comms_dir / "__init__.py"
    if not comms_init.exists():
        with open(comms_init, 'w') as f:
            f.write('"""FedMD 통신 모듈"""\n')
        logger.info(f"생성됨: {comms_init}")
    
    # comms/proto/__init__.py
    proto_init = comms_dir / "proto" / "__init__.py"
    if not proto_init.exists():
        with open(proto_init, 'w') as f:
            f.write('"""gRPC 프로토콜 정의"""\n')
        logger.info(f"생성됨: {proto_init}")


def main():
    """메인 함수"""
    logger = setup_logger("ProtoGenMain")
    
    logger.info("gRPC 프로토콜 스텁 생성 시작")
    
    # __init__.py 파일 생성
    create_init_files()
    
    # gRPC 스텁 생성
    if generate_grpc_stubs():
        logger.info("gRPC 프로토콜 스텁 생성 완료")
        
        # 생성된 파일들 확인
        comms_dir = Path(__file__).parent.parent / "comms"
        generated_files = [
            "fedmd_pb2.py",
            "fedmd_pb2_grpc.py"
        ]
        
        for filename in generated_files:
            filepath = comms_dir / filename
            if filepath.exists():
                logger.info(f"생성됨: {filepath}")
            else:
                logger.warning(f"생성되지 않음: {filepath}")
        
        return True
    else:
        logger.error("gRPC 프로토콜 스텁 생성 실패")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
