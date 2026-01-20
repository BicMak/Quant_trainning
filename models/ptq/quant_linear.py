import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import copy
from torch import Tensor
import os

from .observer_config import QuantConfig, BitTypeConfig
from .bit_type import BitType
from .layer_quantizer.build import build_quantizer
from .utils import init_observers

# observer_config를 받아서 QuantLayer 구성
class QuantLinear(nn.Module):
    def __init__(self, 
                 quant_args:dict,
                 input_module:nn.Module,
                 observer_config:QuantConfig):
        # observer 초기화
        super(QuantLinear, self).__init__()

        #0. observer config copy
        self.input_module = input_module
        self.observer_config = observer_config
        self.observer_type = observer_config.observer_type
        self.bit_type = BitType(
            bits=observer_config.bit_type.bits,
            signed=observer_config.bit_type.signed,
            name=observer_config.bit_type.name
        )
        self.calibration_mode = observer_config.calibration_mode
        self.quant_weight = None
        self.mode = 'fp32'


        #1. set layer type & observer
        self.observer = init_observers(self.observer_type,
                                        self.bit_type,
                                        'linear_weight',
                                        self.calibration_mode,
                                        self.observer_config)
        self.output_observer = copy.deepcopy(self.observer) #deepcopy for output observer

        #2. quantizer build
        self.quantizer = build_quantizer(
            quantizer_str='uniform', 
            bit_type=self.bit_type,
            module_type='linear_weight')
        self.output_quantizer = copy.deepcopy(self.quantizer)

        #2. layer initialization  
        self.fwd_kwargs = dict()
        self.fwd_func = F.linear

        self.weight = self.input_module.weight.clone().detach()

        if self.input_module.bias != None:
            self.bias = self.input_module.bias.clone().detach()
        else:
            self.bias = torch.zeros(self.input_module.weight.size(0)).to(
                self.input_module.weight.device
            )

    
    def calibration(self, x):
        """Calibration 전용 - observer update만 수행"""
        with torch.no_grad():
            self.observer.update(self.weight)
            output = self.fwd_func(x, self.weight, self.bias, **self.fwd_kwargs)
            self.output_observer.update(output)

        return output  # 필요하면 반환

    def compute_quant_params(self):
        """Calibration 끝나고 한 번 호출"""
        self.scaler, self.zero = self.observer.get_quantization_params()
        self.output_scaler, self.output_zero = self.output_observer.get_quantization_params()
        
        # quantizer에 quantization param 설정
        self.quantizer.update_quantization_params(
            self.scaler, self.zero
            )
        self.output_quantizer.update_quantization_params(
            self.output_scaler, self.output_zero
            )   
        
        # weight quantization, save quant_weight
        self.quant_weight = torch.clamp(
            torch.round(self.weight / self.scaler) + self.zero,
            min=self.bit_type.lower_bound,
            max=self.bit_type.upper_bound
        )

        return (self.scaler, self.zero), (self.output_scaler, self.output_zero)


    def forward(self, x):
        # in inference x input is int8 tensor

        if self.mode == 'quantized':
            
            # 1. dequantize weights (int8 -> fp32)
            dequant_weight = self.quantizer.forward(self.quant_weight)

            # 2. fake quantization in fp32
            x = self.fwd_func(x,dequant_weight, self.bias, **self.fwd_kwargs)
            
            # 3. Output fake quantization
            x = self.output_quantizer.forward(x)
            
            return x
        
        else:  # fp32
            return self.fwd_func(x, self.weight, self.bias, **self.fwd_kwargs)  
            


def test_quant_linear(observer_type='PercentileObserver'):
    # ========== 1. Config 설정 ==========
    bit_config = BitTypeConfig(bits=8, signed=True, name='int8')
    observer_config = QuantConfig(
        calibration_mode='layer_wise',
        bit_type=bit_config,
        observer_type=observer_type
    )

    # ========== 2. 간단한 MLP 모델 생성 ==========
    class SimpleMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(784, 512)
            self.fc2 = nn.Linear(512, 256)
            self.fc3 = nn.Linear(256, 128)
            self.fc4 = nn.Linear(128, 10)
            
        def forward(self, x):
            x = self.fc1(x)
            x = self.fc2(x)
            x = self.fc3(x)
            x = self.fc4(x)
            return x
    
    mlp = SimpleMLP()
    print(f"\n{'='*60}")
    print(f"Testing with {observer_type}")
    print(f"{'='*60}")
    print("Original MLP:")
    for name, module in mlp.named_modules():
        if isinstance(module, nn.Linear):
            print(f"  {name}: {module}")

    # ========== 3. QuantLinear로 변환 ==========
    quant_layers = []
    for name, layer in mlp.named_modules():
        if isinstance(layer, nn.Linear):
            quant_layer = QuantLinear(
                quant_args={},
                input_module=layer,
                observer_config=observer_config
            )
            quant_layers.append(quant_layer)
            print(f"Converted: {name}")

    print(f"\nQuantLinear 개수: {len(quant_layers)}")

    # ========== 4. 더미 Calibration 데이터 생성 ==========
    num_batches = 64
    batch_size = 32
    input_dim = 784  # 28x28 flattened

    print(f"\n더미 데이터 생성: {num_batches} batches × {batch_size} samples × {input_dim} features")
    calib_data = [
        torch.randn(batch_size, input_dim) 
        for _ in range(num_batches)
    ]

    # ========== 5. Calibration 수행 ==========
    print("\n=== Calibration 시작 ===")
    for batch_idx, x in enumerate(calib_data):
        out = x
        for layer_idx, layer in enumerate(quant_layers):
            out = layer.calibration(out)
            # ReLU activation (마지막 레이어 제외)
            if layer_idx < len(quant_layers) - 1:
                out = F.relu(out)
        
        if (batch_idx + 1) % 10 == 0:
            print(f"Batch {batch_idx + 1}/{num_batches} 완료")

    print("\n=== Calibration 완료 ===")

    # ========== 6. Observer 통계 확인 ==========
    print("\n=== Observer 통계 ===")
    for i, layer in enumerate(quant_layers):
        print(f"\nLayer {i} (fc{i+1}):")
        print(f"  Observer type: {type(layer.observer).__name__}")
        print(f"  Weight min_val: {layer.observer.min_val:.6f}")
        print(f"  Weight max_val: {layer.observer.max_val:.6f}")
        print(f"  Output min_val: {layer.output_observer.min_val:.6f}")
        print(f"  Output max_val: {layer.output_observer.max_val:.6f}")
        print(f"  Weight shape: {layer.weight.shape}")
        print(f"  Weight range: [{layer.weight.min().item():.4f}, {layer.weight.max().item():.4f}]")

    # ========== 7. Quantization Params 계산 ==========
    print("\n=== Quantization Parameters 계산 ===")
    for i, layer in enumerate(quant_layers):
        (w_scale, w_zero), (out_scale, out_zero) = layer.compute_quant_params()
        print(f"\nLayer {i} (fc{i+1}):")
        print(f"  Weight - scale: {w_scale:.8f}, zero: {w_zero:.2f}")
        print(f"  Output - scale: {out_scale:.8f}, zero: {out_zero:.2f}")

    # ========== 8. FP32 vs Fake Quantization 비교 ==========
    print("\n=== FP32 vs Fake Quantization 출력 비교 ===")
    test_input = torch.randn(1, input_dim)
    
    # FP32 forward
    out_fp32 = test_input
    for layer in quant_layers:
        layer.mode = 'fp32'
        out_fp32 = layer(out_fp32)
        if layer != quant_layers[-1]:
            out_fp32 = F.relu(out_fp32)
    
    # Fake Quantization forward
    out_quant = test_input
    for layer in quant_layers:
        layer.mode = 'quantized'
        out_quant = layer(out_quant)
        if layer != quant_layers[-1]:
            out_quant = F.relu(out_quant)
    
    print(f"FP32 output: {out_fp32[0, :5]}")  # 처음 5개만
    print(f"Fake Quant output: {out_quant[0, :5]}")
    print(f"MSE: {F.mse_loss(out_fp32, out_quant).item():.6f}")
    print(f"Max diff: {(out_fp32 - out_quant).abs().max().item():.6f}")

    print(f"\n=== {observer_type} 테스트 완료 ===\n")


def test_with_profiler():
    """Profiler 테스트 - 레이어별 출력 QSNR 측정"""
    from .layer_profiler import StatProfiler, TimeProfiler, HistProfiler

    print("="*60)
    print("Testing QuantLinear with Profilers (Layer-wise Output QSNR)")
    print("="*60)

    # 1. Config 설정
    bit_config = BitTypeConfig(bits=8, signed=True, name='int8')
    observer_config = QuantConfig(
        calibration_mode='layer_wise',
        bit_type=bit_config,
        observer_type='PercentileObserver'
    )

    # 2. SimpleMLP 생성 및 QuantLinear 변환
    class SimpleMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(784, 512)
            self.fc2 = nn.Linear(512, 256)
            self.fc3 = nn.Linear(256, 128)
            self.fc4 = nn.Linear(128, 10)

    mlp = SimpleMLP()

    # QuantLinear로 변환
    quant_layers = {}
    for name, layer in mlp.named_modules():
        if isinstance(layer, nn.Linear):
            quant_layers[name] = QuantLinear(
                quant_args={},
                input_module=layer,
                observer_config=observer_config
            )

    print(f"Converted {len(quant_layers)} layers: {list(quant_layers.keys())}")

    # 3. 더미 데이터
    num_batches = 16
    batch_size = 32
    calib_data = [torch.randn(batch_size, 784) for _ in range(num_batches)]

    # 4. Calibration
    print("\n=== Calibration ===")
    for x in calib_data:
        out = x
        for name in ['fc1', 'fc2', 'fc3', 'fc4']:
            out = quant_layers[name].calibration(out)
            if name != 'fc4':
                out = F.relu(out)

    # Quantization params 계산
    for name, layer in quant_layers.items():
        layer.compute_quant_params()
    print("Calibration 완료")

    # 5. 레이어별 출력 QSNR 측정
    print("\n=== 레이어별 출력 QSNR (Output Quality) ===")
    test_input = torch.randn(1, 784)

    layer_stats = {}
    x_fp32 = test_input
    x_quant = test_input

    for name in ['fc1', 'fc2', 'fc3', 'fc4']:
        layer = quant_layers[name]

        # FP32 forward
        layer.mode = 'fp32'
        out_fp32 = layer(x_fp32)

        # Quantized forward
        layer.mode = 'quantized'
        out_quant = layer(x_quant)

        # 출력 비교 (이게 진짜 중요!)
        stats = StatProfiler.compute(out_fp32, out_quant)
        layer_stats[name] = stats

        print(f"\n{name} Output:")
        print(f"  MSE: {stats['mse']:.6f}")
        print(f"  Cosine Sim: {stats['cosine_sim']:.4f}")
        print(f"  QSNR: {stats['qsnr']:.2f} dB")  # 출력이니까 QSNR이 의미있음

        # 다음 레이어 입력 준비 (ReLU 적용)
        if name != 'fc4':
            x_fp32 = F.relu(out_fp32)
            x_quant = F.relu(out_quant)

    # 6. 최종 출력 비교
    print("\n=== 전체 모델 출력 비교 ===")

    # 전체 FP32
    out = test_input
    for name in ['fc1', 'fc2', 'fc3', 'fc4']:
        quant_layers[name].mode = 'fp32'
        out = quant_layers[name](out)
        if name != 'fc4':
            out = F.relu(out)
    final_fp32 = out

    # 전체 Quantized
    out = test_input
    for name in ['fc1', 'fc2', 'fc3', 'fc4']:
        quant_layers[name].mode = 'quantized'
        out = quant_layers[name](out)
        if name != 'fc4':
            out = F.relu(out)
    final_quant = out

    final_stats = StatProfiler.compute(final_fp32, final_quant)
    print(f"Final MSE: {final_stats['mse']:.6f}")
    print(f"Final Cosine Sim: {final_stats['cosine_sim']:.4f}")
    print(f"Final QSNR: {final_stats['qsnr']:.2f} dB")

    print("\n=== 모든 Profiler 테스트 완료 ===")


if __name__ == "__main__":
    # 기본 테스트
    observer_types = ['MinmaxObserver', 'PercentileObserver']

    print("="*60)
    print("Testing QuantLinear")
    print("="*60)

    for observer_type in observer_types:
        try:
            test_quant_linear(observer_type=observer_type)
        except Exception as e:
            print(f"\n{observer_type} 테스트 실패")
            print(f"Error: {e}\n")

    # Profiler 테스트
    try:
        test_with_profiler()
    except Exception as e:
        print(f"\nProfiler 테스트 실패: {e}")

    print("="*60)
    print("모든 테스트 완료")
    print("="*60)