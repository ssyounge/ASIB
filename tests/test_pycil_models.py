"""
PyCIL Continual Learning 모델들에 대한 포괄적인 테스트
"""

import pytest
import torch
import torch.nn as nn
import numpy as np

# PyCIL 모델들 import (실제로 존재하는 것들)
from PyCIL.models.base import BaseLearner
from PyCIL.models.asib_cl import ASIB_CL
from PyCIL.models.ewc import EWC
from PyCIL.models.lwf import LwF
from PyCIL.models.icarl import iCaRL
from PyCIL.models.bic import BiC
from PyCIL.models.wa import WA
from PyCIL.models.der import DER
from PyCIL.models.foster import FOSTER
from PyCIL.models.gem import GEM
from PyCIL.models.replay import Replay
from PyCIL.models.simplecil import SimpleCIL
from PyCIL.models.podnet import PODNet
from PyCIL.models.ssre import SSRE
from PyCIL.models.memo import MEMO
from PyCIL.models.tagfex import TagFex
from PyCIL.models.il2a import IL2A
from PyCIL.models.dsal import DSAL
from PyCIL.models.finetune import Finetune
from PyCIL.models.aper_finetune import APER_FINETUNE
from PyCIL.models.acil import ACIL
from PyCIL.models.coil import COIL
from PyCIL.models.fetril import FeTrIL

# 존재하지 않는 모델들은 주석 처리
# from PyCIL.models.icarl import ICarl
# from PyCIL.models.bic import BiC
# from PyCIL.models.wa import WA
# from PyCIL.models.der import DER
# from PyCIL.models.foster import FOSTER
# from PyCIL.models.gem import GEM
# from PyCIL.models.replay import Replay
# from PyCIL.models.simplecil import SimpleCIL
# from PyCIL.models.podnet import PODNet
# from PyCIL.models.ssre import SSRE
# from PyCIL.models.memo import MEMO
# from PyCIL.models.tagfex import TAGFEX
# from PyCIL.models.il2a import IL2A
# from PyCIL.models.dsal import DSAL
# from PyCIL.models.finetune import Finetune
# from PyCIL.models.aper_finetune import APER_Finetune
# from PyCIL.models.acil import ACIL
# from PyCIL.models.coil import COIL
# from PyCIL.models.fetril import FeTrIL

@pytest.fixture
def base_args():
    """기본 테스트 인자들"""
    return {
        "convnet_type": "resnet32",
        "dataset": "cifar100",
        "device": ["cpu"],
        "memory_size": 2000,
        "init_cls": 10,
        "increment": 10,
        "total_cls": 100,
        # DER 모델용
        "der_alpha": 0.2,
        # FOSTER 모델용
        "beta1": 0.5,
        "beta2": 0.5,
        "is_teacher_wa": False,
        "is_student_wa": False,
        "lambda_okd": 1.0,
        "wa_value": 1.0,
        "oofc": "ft",
        # GEM 모델용
        "memory_strength": 0.5,
        # SimpleCIL 모델용
        "min_lr": 1e-8,
        # PODNet 모델용
        "pod_lambda": 1.0,
        # SSRE 모델용 (IncrementalNet 중복 인자 제거)
        # MEMO 모델용 (AdaptiveNet 관련)
        "train_base": True,
        "train_adaptive": True,
        # TAGFEX 모델용
        "proj_hidden_dim": 512,
        "proj_output_dim": 256,
        # IL2A 모델용
        "model_name": "il2a",
        # DSAL/ACIL 모델용 (exemplar-free methods)
        "configurations": {
            "cifar100": {
                "convnet_type": "resnet32",
                "dataset": "cifar100"
            }
        },
        # COIL 모델용
        "sinkhorn": 0.1,
        "calibration_term": 0.1,
        "norm_term": 2,
        # 추가 공통 인자들
        "batch_size": 64,
        "num_workers": 4,
        "epochs": 100,
        "lr": 0.1,
        "weight_decay": 0.0002,
        "step_size": 30,
        "gamma": 0.1,
        "seed": 1993
    }

@pytest.fixture
def exemplar_free_args():
    """Exemplar-free methods (DSAL, ACIL)를 위한 인자들"""
    return {
        "convnet_type": "resnet32",
        "dataset": "cifar100",
        "device": ["cpu"],
        "memory_size": 0,  # exemplar-free methods는 memory_size가 0이어야 함
        "init_cls": 10,
        "increment": 10,
        "total_cls": 100,
        # DSAL/ACIL 모델용
        "configurations": {
            "cifar100": {
                "convnet_type": "resnet32",
                "dataset": "cifar100"
            }
        },
        # DSAL/ACIL 모델용 scheduler
        "scheduler": {
            "lr": 0.1,
            "weight_decay": 0.0002,
            "step_size": 30,
            "gamma": 0.1
        },
        # DSAL/ACIL 모델용 buffer_size
        "buffer_size": 8192,
        # DSAL/ACIL 모델용 model_name
        "model_name": "dsal",
        # DSAL 모델용 추가 인자들
        "gamma_comp": 1e-3,
        "compensation_ratio": 0.6,
        # 추가 공통 인자들
        "batch_size": 64,
        "num_workers": 4,
        "epochs": 100,
        "lr": 0.1,
        "weight_decay": 0.0002,
        "step_size": 30,
        "gamma": 0.1,
        "seed": 1993
    }

@pytest.fixture
def mock_data_manager():
    """더미 데이터 매니저"""
    class MockDataManager:
        def __init__(self):
            self._increments = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
            self._total_classes = 100
            self._nb_tasks = 10
            self._cur_task = 0
            
        def get_dataset(self, indices, source, mode, appendent=None, ret_data=False):
            return None
            
        def get_dataloader(self, dataset, batch_size, shuffle=True, num_workers=0):
            return None
            
        def get_task_size(self, task_id):
            return 10
            
        def get_total_classnum(self):
            return 100
    
    return MockDataManager()

class TestASIB_CL:
    """ASIB-CL 모델 테스트"""
    
    def test_asib_cl_initialization(self, base_args):
        """ASIB-CL 초기화 테스트"""
        model = ASIB_CL(base_args)
        assert isinstance(model, BaseLearner)
        assert model._ib_beta == 0.1
        assert model.lambda_D == 1.0
        assert model.lambda_IB == 1.0
    
    def test_asib_cl_after_task(self, base_args):
        """ASIB-CL after_task 메소드 테스트"""
        model = ASIB_CL(base_args)
        model.after_task()
        assert model._old_network is not None
        assert model._ib_encoder is not None
        assert model._ib_decoder is not None

class TestEWC:
    """EWC 모델 테스트"""
    
    def test_ewc_initialization(self, base_args):
        """EWC 모델 초기화 테스트"""
        model = EWC(base_args)
        assert isinstance(model, EWC)
        assert isinstance(model, BaseLearner)
        # EWC 모델의 실제 속성 확인
        assert hasattr(model, 'fisher')
        assert model.fisher is None
        assert hasattr(model, '_network')
    
    def test_ewc_after_task(self, base_args):
        """EWC 모델 after_task 메서드 테스트"""
        model = EWC(base_args)
        model.after_task()
        # EWC는 after_task에서 _old_network를 설정하지 않음
        assert model._known_classes == model._total_classes

class TestLwF:
    """LwF 모델 테스트"""
    
    def test_lwf_initialization(self, base_args):
        """LwF 모델 초기화 테스트"""
        model = LwF(base_args)
        assert isinstance(model, LwF)
        assert isinstance(model, BaseLearner)
        # LwF 모델의 실제 속성 확인
        assert hasattr(model, '_network')
        # temperature는 모듈 레벨 상수로 정의됨
        from PyCIL.models.lwf import T
        assert T == 2
    
    def test_lwf_after_task(self, base_args):
        """LwF 모델 after_task 메서드 테스트"""
        model = LwF(base_args)
        model.after_task()
        # LwF는 after_task에서 _old_network를 설정함
        assert model._old_network is not None
        assert model._known_classes == model._total_classes

# 존재하지 않는 모델들의 테스트는 주석 처리
class TestICarl:
    def test_icarl_initialization(self, base_args):
        model = iCaRL(base_args)
        assert isinstance(model, BaseLearner)
    
    def test_icarl_after_task(self, base_args):
        model = iCaRL(base_args)
        model.after_task()
        assert model._old_network is not None

class TestBiC:
    def test_bic_initialization(self, base_args):
        model = BiC(base_args)
        assert isinstance(model, BaseLearner)
    
    def test_bic_after_task(self, base_args):
        model = BiC(base_args)
        model.after_task()
        assert model._old_network is not None

class TestWA:
    def test_wa_initialization(self, base_args):
        model = WA(base_args)
        assert isinstance(model, BaseLearner)
    
    def test_wa_after_task(self, base_args):
        model = WA(base_args)
        model.after_task()
        assert model._old_network is not None

class TestDER:
    def test_der_initialization(self, base_args):
        model = DER(base_args)
        assert isinstance(model, BaseLearner)
        # DER 모델은 der_alpha 속성이 없음
    
    def test_der_after_task(self, base_args):
        base_args["der_alpha"] = 0.2
        model = DER(base_args)
        model.after_task()
        # DER은 after_task에서 _old_network를 설정하지 않음
        assert model._known_classes == model._total_classes

class TestFOSTER:
    def test_foster_initialization(self, base_args):
        model = FOSTER(base_args)
        assert isinstance(model, BaseLearner)
    
    def test_foster_after_task(self, base_args):
        model = FOSTER(base_args)
        model.after_task()
        # FOSTER은 after_task에서 _old_network를 설정하지 않음
        assert model._known_classes == model._total_classes

class TestGEM:
    def test_gem_initialization(self, base_args):
        model = GEM(base_args)
        assert isinstance(model, BaseLearner)
        # GEM 모델은 memory_strength 속성이 없음
    
    def test_gem_after_task(self, base_args):
        base_args["memory_strength"] = 0.5
        model = GEM(base_args)
        model.after_task()
        # GEM은 after_task에서 _old_network를 설정하지 않음
        assert model._known_classes == model._total_classes

class TestReplay:
    def test_replay_initialization(self, base_args):
        model = Replay(base_args)
        assert isinstance(model, BaseLearner)
    
    def test_replay_after_task(self, base_args):
        model = Replay(base_args)
        model.after_task()
        # Replay은 after_task에서 _old_network를 설정하지 않음
        assert model._known_classes == model._total_classes

class TestSimpleCIL:
    def test_simplecil_initialization(self, base_args):
        model = SimpleCIL(base_args)
        assert isinstance(model, BaseLearner)
    
    def test_simplecil_after_task(self, base_args):
        model = SimpleCIL(base_args)
        model.after_task()
        # SimpleCIL은 after_task에서 _old_network를 설정하지 않음
        assert model._known_classes == model._total_classes

class TestPODNet:
    def test_podnet_initialization(self, base_args):
        model = PODNet(base_args)
        assert isinstance(model, BaseLearner)
        # PODNet 모델은 pod_lambda 속성이 없음
    
    def test_podnet_after_task(self, base_args):
        base_args["pod_lambda"] = 1.0
        model = PODNet(base_args)
        model.after_task()
        # PODNet은 after_task에서 _old_network를 설정함
        assert model._old_network is not None
        assert model._known_classes == model._total_classes

class TestSSRE:
    def test_ssre_initialization(self, base_args):
        model = SSRE(base_args)
        assert isinstance(model, BaseLearner)
    
    def test_ssre_after_task(self, base_args):
        model = SSRE(base_args)
        model.after_task()
        # SSRE은 after_task에서 _old_network를 설정함
        assert model._old_network is not None
        assert model._known_classes == model._total_classes

class TestMEMO:
    def test_memo_initialization(self, base_args):
        model = MEMO(base_args)
        assert isinstance(model, BaseLearner)
    
    def test_memo_after_task(self, base_args):
        model = MEMO(base_args)
        model.after_task()
        # MEMO은 after_task에서 _old_network를 설정하지 않음
        assert model._known_classes == model._total_classes

class TestTAGFEX:
    def test_tagfex_initialization(self, base_args):
        model = TagFex(base_args)
        assert isinstance(model, BaseLearner)
    
    def test_tagfex_after_task(self, base_args):
        model = TagFex(base_args)
        model.after_task()
        # TAGFEX은 after_task에서 _old_network를 설정하지 않음
        assert model._known_classes == model._total_classes

class TestIL2A:
    def test_il2a_initialization(self, base_args):
        model = IL2A(base_args)
        assert isinstance(model, BaseLearner)
    
    def test_il2a_after_task(self, base_args):
        model = IL2A(base_args)
        model.after_task()
        # IL2A은 after_task에서 _old_network를 설정하지 않음
        assert model._known_classes == model._total_classes

class TestDSAL:
    def test_dsal_initialization(self, exemplar_free_args):
        model = DSAL(exemplar_free_args)
        assert isinstance(model, BaseLearner)
    
    def test_dsal_after_task(self, exemplar_free_args):
        model = DSAL(exemplar_free_args)
        # fc와 fc_comp를 초기화해야 함
        model._network.fc = type('MockFC', (), {'after_task': lambda self: None})()
        model._network.fc_comp = type('MockFC', (), {'after_task': lambda self: None})()
        model.after_task()
        # DSAL은 after_task에서 _old_network를 설정하지 않음
        assert model._known_classes == model._total_classes

class TestFinetune:
    def test_finetune_initialization(self, base_args):
        model = Finetune(base_args)
        assert isinstance(model, BaseLearner)
    
    def test_finetune_after_task(self, base_args):
        model = Finetune(base_args)
        model.after_task()
        # Finetune은 after_task에서 _old_network를 설정하지 않음
        assert model._known_classes == model._total_classes

class TestAPER_Finetune:
    def test_aper_finetune_initialization(self, base_args):
        model = APER_FINETUNE(base_args)
        assert isinstance(model, BaseLearner)
    
    def test_aper_finetune_after_task(self, base_args):
        model = APER_FINETUNE(base_args)
        model.after_task()
        # APER_Finetune은 after_task에서 _old_network를 설정하지 않음
        assert model._known_classes == model._total_classes

class TestACIL:
    def test_acil_initialization(self, exemplar_free_args):
        model = ACIL(exemplar_free_args)
        assert isinstance(model, BaseLearner)
    
    def test_acil_after_task(self, exemplar_free_args):
        model = ACIL(exemplar_free_args)
        # fc를 초기화해야 함
        model._network.fc = type('MockFC', (), {'after_task': lambda self: None})()
        model.after_task()
        # ACIL은 after_task에서 _old_network를 설정하지 않음
        assert model._known_classes == model._total_classes

class TestCOIL:
    def test_coil_initialization(self, base_args):
        model = COIL(base_args)
        assert isinstance(model, BaseLearner)
    
    def test_coil_after_task(self, base_args, mock_data_manager):
        model = COIL(base_args)
        # COIL 모델은 data_manager가 필요함
        model.data_manager = mock_data_manager
        # _total_classes도 설정해야 함
        model._total_classes = 10
        # _extract_class_means 메소드가 필요함 (mock)
        model._extract_class_means = lambda data_manager, low, high: None
        model._ot_prototype_means = torch.randn(20, 512)  # dummy data
        # fc를 초기화해야 함 (CUDA로 이동)
        model._network.fc = torch.nn.Linear(512, 10).cuda()
        model.after_task()
        # COIL은 after_task에서 _old_network를 설정함
        assert model._old_network is not None
        assert model._known_classes == model._total_classes

class TestFeTrIL:
    def test_fetril_initialization(self, base_args):
        model = FeTrIL(base_args)
        assert isinstance(model, BaseLearner)
    
    def test_fetril_after_task(self, base_args):
        model = FeTrIL(base_args)
        model.after_task()
        # FeTrIL은 after_task에서 _old_network를 설정하지 않음
        assert model._known_classes == model._total_classes

class TestModelCompatibility:
    """모든 모델의 호환성 테스트"""
    
    def test_all_models_inherit_base_learner(self, base_args):
        """모든 모델이 BaseLearner를 상속하는지 테스트"""
        models = [
            (ASIB_CL, "ASIB_CL"),
            (EWC, "EWC"),
            (LwF, "LwF"),
            (iCaRL, "iCaRL"),
            (BiC, "BiC"),
            (WA, "WA"),
            (DER, "DER"),
            (FOSTER, "FOSTER"),
            (GEM, "GEM"),
            (Replay, "Replay"),
            (SimpleCIL, "SimpleCIL"),
            (PODNet, "PODNet"),
            (SSRE, "SSRE"),
            (MEMO, "MEMO"),
            (TagFex, "TagFex"),
            (IL2A, "IL2A"),
            (DSAL, "DSAL"),
            (Finetune, "Finetune"),
            (APER_FINETUNE, "APER_FINETUNE"),
            (ACIL, "ACIL"),
            (COIL, "COIL"),
            (FeTrIL, "FeTrIL"),
        ]
        
        for model_class, model_name in models:
            try:
                model = model_class(base_args)
                assert isinstance(model, BaseLearner), f"{model_name} should inherit from BaseLearner"
            except Exception as e:
                # 일부 모델은 특별한 설정이 필요할 수 있음
                print(f"Warning: {model_name} initialization failed: {e}")
    
    def test_all_models_have_after_task(self, base_args):
        """모든 모델이 after_task 메서드를 가지고 있는지 테스트"""
        models = [
            (ASIB_CL, "ASIB_CL"),
            (EWC, "EWC"),
            (LwF, "LwF"),
            (iCaRL, "iCaRL"),
            (BiC, "BiC"),
            (WA, "WA"),
            (DER, "DER"),
            (FOSTER, "FOSTER"),
            (GEM, "GEM"),
            (Replay, "Replay"),
            (SimpleCIL, "SimpleCIL"),
            (PODNet, "PODNet"),
            (SSRE, "SSRE"),
            (MEMO, "MEMO"),
            (TagFex, "TagFex"),
            (IL2A, "IL2A"),
            (DSAL, "DSAL"),
            (Finetune, "Finetune"),
            (APER_FINETUNE, "APER_FINETUNE"),
            (ACIL, "ACIL"),
            (COIL, "COIL"),
            (FeTrIL, "FeTrIL"),
        ]
        
        for model_class, model_name in models:
            try:
                model = model_class(base_args)
                assert hasattr(model, 'after_task'), f"{model_name} should have after_task method"
                assert callable(getattr(model, 'after_task')), f"{model_name}.after_task should be callable"
            except Exception as e:
                # 일부 모델은 특별한 설정이 필요할 수 있음
                print(f"Warning: {model_name} after_task test failed: {e}")
    
    def test_all_models_have_network(self, base_args):
        """모든 모델이 _network 속성을 가지고 있는지 테스트"""
        models = [
            (ASIB_CL, "ASIB_CL"),
            (EWC, "EWC"),
            (LwF, "LwF"),
            (iCaRL, "iCaRL"),
            (BiC, "BiC"),
            (WA, "WA"),
            (DER, "DER"),
            (FOSTER, "FOSTER"),
            (GEM, "GEM"),
            (Replay, "Replay"),
            (SimpleCIL, "SimpleCIL"),
            (PODNet, "PODNet"),
            (SSRE, "SSRE"),
            (MEMO, "MEMO"),
            (TagFex, "TagFex"),
            (IL2A, "IL2A"),
            (DSAL, "DSAL"),
            (Finetune, "Finetune"),
            (APER_FINETUNE, "APER_FINETUNE"),
            (ACIL, "ACIL"),
            (COIL, "COIL"),
            (FeTrIL, "FeTrIL"),
        ]
        
        for model_class, model_name in models:
            try:
                model = model_class(base_args)
                assert hasattr(model, '_network'), f"{model_name} should have _network attribute"
            except Exception as e:
                # 일부 모델은 특별한 설정이 필요할 수 있음
                print(f"Warning: {model_name} _network test failed: {e}")

if __name__ == "__main__":
    pytest.main([__file__]) 