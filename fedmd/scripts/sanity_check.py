"""
FedMD 시스템 스모크 테스트
단일 프로세스에서 서버와 클라이언트를 시뮬레이션
"""
import sys
import time
import argparse
from pathlib import Path
import logging
import json
from tqdm import tqdm

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.config import load_config_from_cli, create_arg_parser
from core.server import FedMDServer
from core.client import ClientNode
from core.utils import setup_logger, set_seed


class SanityChecker:
    """FedMD 시스템 스모크 테스트"""
    
    def __init__(self, config, args):
        self.config = config
        self.args = args
        self.logger = setup_logger("SanityChecker", level=getattr(logging, args.log_level))
        
        # 시드 설정
        set_seed(42)
        
        # 서버 및 클라이언트
        self.server = None
        self.clients = []
        
        # 테스트 결과
        self.test_results = {
            "server_startup": False,
            "client_registration": False,
            "data_loading": False,
            "model_building": False,
            "local_training": False,
            "logits_inference": False,
            "logits_upload": False,
            "soft_targets_generation": False,
            "soft_targets_reception": False,
            "knowledge_distillation": False,
            "metrics_evaluation": False,
            "end_to_end_round": False
        }
    
    def test_server_startup(self):
        """서버 시작 테스트"""
        self.logger.info("=== 서버 시작 테스트 ===")
        
        try:
            self.server = FedMDServer(self.config.dict(), self.logger)
            self.test_results["server_startup"] = True
            self.logger.info("✓ 서버 시작 성공")
            return True
        except Exception as e:
            self.logger.error(f"✗ 서버 시작 실패: {e}")
            return False
    
    def test_client_registration(self):
        """클라이언트 등록 테스트"""
        self.logger.info("=== 클라이언트 등록 테스트 ===")
        
        try:
            # 클라이언트 생성
            client_a = ClientNode("clientA", self.config.dict(), logger=self.logger)
            client_b = ClientNode("clientB", self.config.dict(), logger=self.logger)
            
            # 등록
            if not client_a.register():
                raise Exception("클라이언트 A 등록 실패")
            
            if not client_b.register():
                raise Exception("클라이언트 B 등록 실패")
            
            self.clients = [client_a, client_b]
            self.test_results["client_registration"] = True
            self.logger.info("✓ 클라이언트 등록 성공")
            return True
        except Exception as e:
            self.logger.error(f"✗ 클라이언트 등록 실패: {e}")
            return False
    
    def test_data_loading(self):
        """데이터 로딩 테스트"""
        self.logger.info("=== 데이터 로딩 테스트 ===")
        
        try:
            # 공용 인덱스 생성 (스모크 테스트용)
            public_indices = list(range(100))  # 100개 더미 인덱스
            public_indices_file = Path("data/public/public_indices.json")
            public_indices_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(public_indices_file, 'w') as f:
                json.dump(public_indices, f)
            
            self.logger.info(f"공용 인덱스 생성: {len(public_indices)}개")
            
            for client in self.clients:
                if not client.load_datasets():
                    raise Exception(f"클라이언트 {client.client_id} 데이터 로딩 실패")
            
            self.test_results["data_loading"] = True
            self.logger.info("✓ 데이터 로딩 성공")
            return True
        except Exception as e:
            self.logger.error(f"✗ 데이터 로딩 실패: {e}")
            return False
    
    def test_model_building(self):
        """모델 빌드 테스트"""
        self.logger.info("=== 모델 빌드 테스트 ===")
        
        try:
            for client in self.clients:
                if not client.build_model():
                    raise Exception(f"클라이언트 {client.client_id} 모델 빌드 실패")
            
            self.test_results["model_building"] = True
            self.logger.info("✓ 모델 빌드 성공")
            return True
        except Exception as e:
            self.logger.error(f"✗ 모델 빌드 실패: {e}")
            return False
    
    def test_local_training(self):
        """로컬 학습 테스트"""
        self.logger.info("=== 로컬 학습 테스트 ===")
        
        try:
            for client in self.clients:
                history = client.local_pretrain()
                if not history or "train_accuracy" not in history:
                    raise Exception(f"클라이언트 {client.client_id} 로컬 학습 실패")
            
            self.test_results["local_training"] = True
            self.logger.info("✓ 로컬 학습 성공")
            return True
        except Exception as e:
            self.logger.error(f"✗ 로컬 학습 실패: {e}")
            return False
    
    def test_logits_inference(self):
        """로짓 추론 테스트"""
        self.logger.info("=== 로짓 추론 테스트 ===")
        
        try:
            for client in self.clients:
                logits_data = client.infer_public_logits()
                if not logits_data or len(logits_data) == 0:
                    raise Exception(f"클라이언트 {client.client_id} 로짓 추론 실패")
            
            self.test_results["logits_inference"] = True
            self.logger.info("✓ 로짓 추론 성공")
            return True
        except Exception as e:
            self.logger.error(f"✗ 로짓 추론 실패: {e}")
            return False
    
    def test_logits_upload(self):
        """로짓 업로드 테스트"""
        self.logger.info("=== 로짓 업로드 테스트 ===")
        
        try:
            # 라운드 시작
            self.server.start_round(1)
            
            # 로짓 업로드
            for client in self.clients:
                logits_data = client.infer_public_logits()
                if not client.upload_logits(logits_data):
                    raise Exception(f"클라이언트 {client.client_id} 로짓 업로드 실패")
            
            self.test_results["logits_upload"] = True
            self.logger.info("✓ 로짓 업로드 성공")
            return True
        except Exception as e:
            self.logger.error(f"✗ 로짓 업로드 실패: {e}")
            return False
    
    def test_soft_targets_generation(self):
        """소프트 타겟 생성 테스트"""
        self.logger.info("=== 소프트 타겟 생성 테스트 ===")
        
        try:
            # 소프트 타겟 생성
            result = self.server.make_soft_targets()
            if not result["success"]:
                raise Exception("소프트 타겟 생성 실패")
            
            soft_targets = result["soft_targets"]
            if not soft_targets or len(soft_targets) == 0:
                raise Exception("소프트 타겟이 비어있음")
            
            self.test_results["soft_targets_generation"] = True
            self.logger.info("✓ 소프트 타겟 생성 성공")
            return True
        except Exception as e:
            self.logger.error(f"✗ 소프트 타겟 생성 실패: {e}")
            return False
    
    def test_soft_targets_reception(self):
        """소프트 타겟 수신 테스트"""
        self.logger.info("=== 소프트 타겟 수신 테스트 ===")
        
        try:
            for client in self.clients:
                soft_targets = client.receive_soft_targets()
                if soft_targets is None or len(soft_targets) == 0:
                    raise Exception(f"클라이언트 {client.client_id} 소프트 타겟 수신 실패")
            
            self.test_results["soft_targets_reception"] = True
            self.logger.info("✓ 소프트 타겟 수신 성공")
            return True
        except Exception as e:
            self.logger.error(f"✗ 소프트 타겟 수신 실패: {e}")
            return False
    
    def test_knowledge_distillation(self):
        """지식 증류 테스트"""
        self.logger.info("=== 지식 증류 테스트 ===")
        
        try:
            for client in self.clients:
                soft_targets = client.receive_soft_targets()
                if soft_targets is None:
                    continue
                
                history = client.distill(soft_targets)
                if not history or "distill_accuracy" not in history:
                    raise Exception(f"클라이언트 {client.client_id} 지식 증류 실패")
            
            self.test_results["knowledge_distillation"] = True
            self.logger.info("✓ 지식 증류 성공")
            return True
        except Exception as e:
            self.logger.error(f"✗ 지식 증류 실패: {e}")
            return False
    
    def test_metrics_evaluation(self):
        """메트릭 평가 테스트"""
        self.logger.info("=== 메트릭 평가 테스트 ===")
        
        try:
            for client in self.clients:
                metrics = client.evaluate()
                if not metrics or "accuracy" not in metrics:
                    raise Exception(f"클라이언트 {client.client_id} 메트릭 평가 실패")
                
                # 메트릭 보고
                if not client.report_metrics(metrics):
                    raise Exception(f"클라이언트 {client.client_id} 메트릭 보고 실패")
            
            self.test_results["metrics_evaluation"] = True
            self.logger.info("✓ 메트릭 평가 성공")
            return True
        except Exception as e:
            self.logger.error(f"✗ 메트릭 평가 실패: {e}")
            return False
    
    def test_end_to_end_round(self):
        """엔드투엔드 라운드 테스트"""
        self.logger.info("=== 엔드투엔드 라운드 테스트 ===")
        
        try:
            # 새로운 클라이언트로 전체 라운드 실행
            test_client = ClientNode("test_client", self.config.dict(), logger=self.logger)
            
            # 등록
            if not test_client.register():
                raise Exception("테스트 클라이언트 등록 실패")
            
            # 데이터 로딩
            if not test_client.load_datasets():
                raise Exception("테스트 클라이언트 데이터 로딩 실패")
            
            # 모델 빌드
            if not test_client.build_model():
                raise Exception("테스트 클라이언트 모델 빌드 실패")
            
            # 라운드 실행
            results = test_client.run_round()
            
            if not results["success"]:
                raise Exception(f"라운드 실행 실패: {results.get('error', '알 수 없는 오류')}")
            
            self.test_results["end_to_end_round"] = True
            self.logger.info("✓ 엔드투엔드 라운드 성공")
            return True
        except Exception as e:
            self.logger.error(f"✗ 엔드투엔드 라운드 실패: {e}")
            return False
    
    def run_all_tests(self):
        """모든 테스트 실행"""
        self.logger.info("FedMD 시스템 스모크 테스트 시작")
        
        tests = [
            ("서버 시작", self.test_server_startup),
            ("클라이언트 등록", self.test_client_registration),
            ("데이터 로딩", self.test_data_loading),
            ("모델 빌드", self.test_model_building),
            ("로컬 학습", self.test_local_training),
            ("로짓 추론", self.test_logits_inference),
            ("로짓 업로드", self.test_logits_upload),
            ("소프트 타겟 생성", self.test_soft_targets_generation),
            ("소프트 타겟 수신", self.test_soft_targets_reception),
            ("지식 증류", self.test_knowledge_distillation),
            ("메트릭 평가", self.test_metrics_evaluation),
            ("엔드투엔드 라운드", self.test_end_to_end_round)
        ]
        
        passed = 0
        total = len(tests)
        
        # 테스트 진행률 표시
        with tqdm(total=total, desc="🧪 FedMD 스모크 테스트", 
                 unit="테스트", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} 테스트 [{elapsed}<{remaining}]") as pbar:
            
            for test_name, test_func in tests:
                pbar.set_description(f"🧪 {test_name} 테스트 중")
                pbar.set_postfix_str("실행 중...")
                
                self.logger.info(f"\n--- {test_name} 테스트 ---")
                try:
                    if test_func():
                        passed += 1
                        pbar.set_postfix_str("✅ 통과")
                        self.logger.info(f"✓ {test_name} 통과")
                    else:
                        pbar.set_postfix_str("❌ 실패")
                        self.logger.error(f"✗ {test_name} 실패")
                except Exception as e:
                    pbar.set_postfix_str("❌ 오류")
                    self.logger.error(f"✗ {test_name} 오류: {e}")
                
                pbar.update(1)
        
        # 결과 요약
        pbar.set_description(f"🏁 테스트 완료: {passed}/{total} 통과")
        pbar.set_postfix_str(f"성공률: {passed/total*100:.1f}%")
        
        self.logger.info(f"\n=== 테스트 결과 요약 ===")
        self.logger.info(f"통과: {passed}/{total}")
        self.logger.info(f"실패: {total - passed}/{total}")
        
        # 상세 결과
        for test_name, result in self.test_results.items():
            status = "✓ 통과" if result else "✗ 실패"
            self.logger.info(f"{test_name}: {status}")
        
        # 결과 저장
        self._save_test_results()
        
        return passed == total
    
    def _save_test_results(self):
        """테스트 결과 저장"""
        results_file = Path("artifacts/sanity_check_results.json")
        results_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        
        self.logger.info(f"테스트 결과 저장됨: {results_file}")


def main():
    """메인 함수"""
    # 인자 파서 생성
    parser = create_arg_parser()
    
    # 인자 파싱
    args = parser.parse_args()
    
    try:
        # 설정 로드
        config, _ = load_config_from_cli()
        
        # 스모크 테스트 실행
        checker = SanityChecker(config, args)
        success = checker.run_all_tests()
        
        if success:
            print("\n🎉 모든 테스트 통과! FedMD 시스템이 정상적으로 작동합니다.")
            sys.exit(0)
        else:
            print("\n❌ 일부 테스트 실패. 로그를 확인하세요.")
            sys.exit(1)
            
    except Exception as e:
        print(f"스모크 테스트 실행 실패: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
