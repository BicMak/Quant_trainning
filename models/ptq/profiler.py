import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import copy
from torch import Tensor

from models.ptq.quant_linear import QuantLinear

from .observer_config import QuantConfig, BitTypeConfig
from .bit_type import BitType
from .layer_quantizer.build import build_quantizer
import os

import models.ptq.observer_config as observer_config
from models.ptq.utils import init_observers

# observer_config를 받아서 QuantLayer 구성
class QAct(nn.Module):
    def __init__(self, 
                 quant_args:dict,
                 observer_config:QuantConfig):
        # observer 초기화
        super(QAct, self).__init__()

        #0. observer config copy
        self.observer_config = observer_config
        self.observer_type = observer_config.observer_type
        self.bit_type = BitType(
            bits=observer_config.bit_type.bits,
            signed=observer_config.bit_type.signed,
            name=observer_config.bit_type.name
        )
        self.calibration_mode = observer_config.calibration_mode
        self.mode = 'fp32'

        #1. set layer type & observer
        self.observer = init_observers(self.observer_type,
                                        self.bit_type,
                                        'activation',
                                        self.calibration_mode,
                                        self.observer_config)

        #2. quantizer build
        self.quantizer = build_quantizer(
            quantizer_str='uniform', 
            bit_type=self.bit_type,
            module_type='activation')

    
    def calibration(self, x):
        """Calibration 전용 - observer update만 수행"""
        with torch.no_grad():
            self.observer.update(x)
            output = x

        return output  # 필요하면 반환

    def compute_quant_params(self):
        """Calibration 끝나고 한 번 호출"""
        self.scaler, self.zero = self.observer.get_quantization_params()
        
        # quantizer에 quantization param 설정
        self.quantizer.update_quantization_params(
            self.scaler, self.zero
            )
 
        
        return (self.scaler, self.zero)


    def forward(self, x):
        # in inference x input is int8 tensor

        if self.mode == 'quantized':
            
            # 1. quantize activation variable
            x = self.quantizer.quant(x)
            
            return x
        
        else:  # fp32
            return x
            

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np

# 사용자가 제공한 QAct 클래스 및 의존성 라이브러리들이 로드되었다고 가정합니다.
# (QuantLinear, QuantConfig, BitTypeConfig 등)

def test_qact_attention_matmul(observer_type='MinmaxObserver'):
    print(f"\n{'='*70}")
    print(f"Testing QAct for Attention MatMul Result (QK^T) with {observer_type}")
    print(f"{'='*70}")

    # ========== 1. 초기 설정 (QuantLinear와 동일한 Config) ==========
    bit_config = BitTypeConfig(bits=8, signed=True, name='int8')
    obs_config = QuantConfig(
        calibration_mode='layer_wise',
        bit_type=bit_config,
        observer_type=observer_type
    )

    # QAct 인스턴스 생성 (MatMul 결과값 전용)
    qact_matmul = QAct(quant_args={}, observer_config=obs_config)

    # ========== 2. 상황 가정: Q, K는 이미 양자화되어 출력되었다고 가정 ==========
    # Batch=1, Head=8, Seq_len=128, Dim=64
    B, H, S, D = 1, 8, 128, 64
    
    # 실제 환경이라면 QuantLinear를 통과한 결과겠지만, 
    # 테스트를 위해 이미 정수 눈금에 맞춰진(Fake-quantized) Q, K 생성
    # (실제 하드웨어라면 INT8이겠지만 시뮬레이션이므로 FP32 내의 정수값)
    query = torch.randn(B, H, S, D).round() 
    key = torch.randn(B, H, S, D).round()
    scale_factor = D ** 0.5  # ViT의 root(d_k) 스케일링

    # ========== 3. Calibration 수행 (MatMul 출력값 분포 파악) ==========
    print("Step 1: Calibration - Observing MatMul outputs...")
    num_batches = 32
    
    for i in range(num_batches):
        # 매번 다른 데이터를 MatMul 한다고 가정
        q_sim = torch.randn(B, H, S, D).round()
        k_sim = torch.randn(B, H, S, D).round()
        
        # 실제 MatMul 구현 (Attention Score 계산)
        # result shape: [1, 8, 128, 128]
        matmul_result = torch.matmul(q_sim, k_sim.transpose(-2, -1)) / scale_factor
        
        # QAct의 calibration 매서드를 사용하여 출력값 통계 수집
        qact_matmul.calibration(matmul_result)

    # Quantization 파라미터 계산 (Scale, Zero-point 추출)
    scaler, zero = qact_matmul.compute_quant_params()
    print(f"Calibration Done. Computed Scaler: {scaler.item():.8f}, Zero: {zero.item():.2f}")

    # ========== 4. Inference 시뮬레이션 (값 뭉개기 확인) ==========
    print("\nStep 2: Inference - Applying Quantization to MatMul result...")
    qact_matmul.mode = 'quantized'

    # 테스트용 MatMul 수행
    original_matmul = torch.matmul(query, key.transpose(-2, -1)) / scale_factor
    
    # QAct 통과 (여기서 값이 8-bit 수준으로 뭉개짐)
    quantized_matmul = qact_matmul(original_matmul)

    # ========== 5. 결과 검증 및 분석 ==========
    # 1. MSE 오차 측정
    mse = F.mse_loss(original_matmul, quantized_matmul)
    
    # 2. 코사인 유사도 (방향성 유지 확인)
    cos_sim = F.cosine_similarity(original_matmul.flatten(), quantized_matmul.flatten(), dim=0)

    # 3. 데이터 뭉개짐 확인 (고유 값의 개수)
    # FP32는 고유값이 매우 많지만, 양자화되면 최대 256개(8-bit)로 제한됨
    unique_orig = len(torch.unique(original_matmul))
    unique_quant = len(torch.unique(quantized_matmul))

    print(f"\n[Analysis Results]")
    print(f"  - Original Unique Values: {unique_orig}")
    print(f"  - Quantized Unique Values: {unique_quant} (Should be <= 256)")
    print(f"  - MSE Loss: {mse.item():.10f}")
    print(f"  - Cosine Similarity: {cos_sim.item():.10f}")
    
    # 실제 값 샘플 비교
    print(f"\n[Value Comparison Sample]")
    print(f"  - Original (First 5): {original_matmul.flatten()[:5].tolist()}")
    print(f"  - Quantized (First 5): {quantized_matmul.flatten()[:5].tolist()}")

    if cos_sim > 0.99:
        print(f"\n✅ {observer_type} Test Passed! (High similarity maintained)")
    else:
        print(f"\n❌ {observer_type} Test Warning: Similarity is lower than expected.")

if __name__ == "__main__":
    # 작성한 QAct의 동작을 관찰하기 위해 여러 Observer로 테스트
    observers = ['MinmaxObserver', 'PercentileObserver']
    
    for obs in observers:
        try:
            test_qact_attention_matmul(observer_type=obs)
        except Exception as e:
            print(f"Error testing {obs}: {e}")