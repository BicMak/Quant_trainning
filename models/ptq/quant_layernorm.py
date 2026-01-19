import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import copy
from torch import Tensor
import os

from .observer_config import ObserverConfig, BitTypeConfig
from .bit_type import BitType
from .layer_quantizer.build import build_quantizer
from .utils import init_observers

# observer_config를 받아서 QuantLayer 구성
class QLayerNorm(nn.Module):
    def __init__(self, 
                 quant_args:dict,
                 input_module:nn.Module,
                 observer_config:ObserverConfig):
        # observer 초기화
        super(QLayerNorm, self).__init__()

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

        #
        self.normalized_shape = self.input_module.normalized_shape
        self.eps = self.input_module.eps
        self.num_channels = self.input_module.weight.size(0)

        #LayerNorm specific params
        self.scale = None
        self.zero = None
        self.s_base = None

        #1. set layer type & observer
        self.output_observer = init_observers(self.observer_type,
                                        self.bit_type,
                                        'activation',
                                        self.calibration_mode,
                                        self.observer_config)

        #2. quantizer build
        self.output_quantizer = build_quantizer(
            quantizer_str='uniform', 
            bit_type=self.bit_type,
            module_type='activation')

        #2. layer initialization  
        self.fwd_kwargs = dict()
        self.fwd_func = F.layer_norm

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
            mean = x.mean(-1, keepdim=True)
            var = x.var(-1, keepdim=True, unbiased=False)
            x_normalized = (x - mean) / torch.sqrt(var + self.eps)
            x = self.fwd_func(x_normalized, self.weight, self.bias, **self.fwd_kwargs)
            output = self.output_observer.update(x)

        return output  # 필요하면 반환
    
    def _find_alpha(self, calib_loader):
        """alpha 탐색용 내부 메서드"""
        all_alphas = []
        for x in calib_loader:
            with torch.no_grad():
                out = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
                alpha = self.find_best_ptf(out, self.s_base)
                all_alphas.append(alpha)
        # 배치별 alpha 중 최빈값 또는 평균
        return torch.stack(all_alphas).float().mean(dim=0).round().int() 

    def compute_quant_params(self,calib_loader):
        """Calibration 끝나고 한 번 호출"""
        scale, self.zero = self.output_observer.get_quantization_params()

        # scale: layer-wise (mean 또는 max), zero : layer-wise
        self.s_base = scale.mean()
        self.scale = scale.fill_(self.s_base)  # shape 유지하면서 값만 통일
        self.output_quantizer.update_quantization_params(
            self.scale, self.zero
            )   
        
        if calib_loader is not None:
            self.alpha = self._find_alpha(calib_loader)
        else:
            # calib_loader 없으면 alpha=0 (기본값, PTF 미적용)
            self.alpha = torch.zeros(self.num_channels, dtype=torch.int32)
        

        return (self.scale, self.zero)


    def forward(self, x):
        # in inference x input is int8 tensor

        if self.mode == 'quantized':
            
            # 1. dequantize weights (int8 -> fp32)
            mean = x.mean(-1, keepdim=True)
            var = x.var(-1, keepdim=True, unbiased=False)
            x_normalized = (x - mean) / torch.sqrt(var + 1e-5)

            # 2. fake quantization in fp32
            x = self.fwd_func(x_normalized, self.weight, self.bias, **self.fwd_kwargs)

            # 3. Output fake quantization
            x = self.output_quantizer.forward(x)
            
            return x
        
        else:  # fp32
            return self.fwd_func(x, self.weight, self.bias, **self.fwd_kwargs)  
            


def test_quant_linear(observer_type='PercentileObserver'):
    # ========== 1. Config 설정 ==========
    bit_config = BitTypeConfig(bits=8, signed=True, name='int8')
    observer_config = ObserverConfig(
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
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
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


if __name__ == "__main__":
    observer_types = ['MinmaxObserver', 'PercentileObserver', 'OmseObserver', 'KVObserver']
    
    print("="*60)
    print("Testing All Observer Types")
    print("="*60)
    
    for observer_type in observer_types:
        try:
            test_quant_linear(observer_type=observer_type)
        except Exception as e:
            print(f"\n❌ {observer_type} 테스트 실패")
            print(f"Error: {e}\n")
    
    print("="*60)
    print("모든 테스트 완료")
    print("="*60)