import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import copy
from torch import Tensor

from observer_config import ObserverConfig, BitTypeConfig
from layer_observer.minmax import MinmaxObserver
from layer_observer.percentile import PercentileObserver
from layer_observer.omse import OmseObserver
from layer_observer.kv_divergence import KVObserver
from layer_observer.bit_type import BitType
import os

import observer_config

ACTIVATION_MAP = {
    nn.ReLU: F.relu,
    nn.ReLU6: F.relu6,
    nn.GELU: F.gelu,
    nn.SiLU: F.silu,
    nn.Sigmoid: torch.sigmoid,
    nn.Tanh: torch.tanh,
    nn.LeakyReLU: lambda x, m: F.leaky_relu(x, m.negative_slope),
    nn.Hardswish: F.hardswish,
}

def init_observers(observer_type, bit_type, 
                   module_type, calibration_mode,
                   observer_config):
    #2. observer initialization
    if observer_type == 'MinmaxObserver':
        observer = MinmaxObserver(
            bit_type=bit_type, 
            module_type=module_type,
            calibration_mode=calibration_mode
        )
    elif observer_type == 'PercentileObserver':
        observer = PercentileObserver(
            bit_type=bit_type,
            module_type=module_type,
            calibration_mode=calibration_mode,
            percentile_alpha=observer_config.percentile_alpha,
            percentile_sigma=observer_config.percentile_sigma
        )
    elif observer_type == 'OmseObserver':
        observer = OmseObserver(
            bit_type=bit_type,
            module_type=module_type,
            calibration_mode=calibration_mode
        )
    elif observer_type == 'KLObserver':
        observer = KVObserver(
            bit_type=bit_type,
            module_type=module_type,
            calibration_mode=calibration_mode,
            kl_bins=observer_config.kl_bins
        )

    return observer

# observer_config를 받아서 QuantLayer 구성
class QuantLinear(nn.Module):
    def __init__(self, 
                 quant_args:dict,
                 input_module:nn.Module,
                 observer_config:ObserverConfig):
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

        #2. layer initialization  
        self.fwd_kwargs = dict()
        self.fwd_func = F.linear

        self.weight = self.input_module.weight.clone().detach()
        self.output_observer = copy.deepcopy(self.observer) #deepcopy for output observer

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
            dequant_weight = (self.quant_weight - self.zero) * self.scaler

            # 2. fake quantization in fp32
            x = self.fwd_func(x,dequant_weight, self.bias, **self.fwd_kwargs)
            
            # 3. Output fake quantization
            output_q = torch.clamp(
                torch.round(x / self.output_scaler) + self.output_zero,
                min=self.bit_type.lower_bound,
                max=self.bit_type.upper_bound
            )
            
            # 4. Output dequantization (다시 float로)
            x = (output_q - self.output_zero) * self.output_scaler

            return x
        
        else:  # fp32
            return self.fwd_func(x, self.weight, self.bias, **self.fwd_kwargs)  
            


# observer_config를 받아서 QuantLayer 구성
class QuantConv(nn.Module):
    def __init__(self, 
                 quant_args:dict,
                 input_module:nn.Module,
                 observer_config:ObserverConfig):
        # observer 초기화
        super(QuantConv, self).__init__()

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
                                        'conv_weight',
                                        self.calibration_mode,
                                        self.observer_config)

        #2. layer initialization  
        self.fwd_kwargs = dict(
            stride=self.input_module.stride,
            padding=self.input_module.padding,
            dilation=self.input_module.dilation,
            groups=self.input_module.groups,
        )
        self.fwd_func = F.conv2d

        self.weight = self.input_module.weight.clone().detach()
        self.output_observer = copy.deepcopy(self.observer) #deepcopy for output observer

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
            dequant_weight = (self.quant_weight - self.zero) * self.scaler

            # 2. fake quantization in fp32
            x = self.fwd_func(x,dequant_weight, self.bias, **self.fwd_kwargs)
            
            # 3. Output fake quantization
            output_q = torch.clamp(
                torch.round(x / self.output_scaler) + self.output_zero,
                min=self.bit_type.lower_bound,
                max=self.bit_type.upper_bound
            )
            
            # 4. Output dequantization (다시 float로)
            x = (output_q - self.output_zero) * self.output_scaler

            return x
        
        else:  # fp32
            return self.fwd_func(x, self.weight, self.bias, **self.fwd_kwargs)  
            



        



def test_quant_linear():
    # ========== 1. Config 설정 ==========
    bit_config = BitTypeConfig(bits=8, signed=False, name='int8')
    observer_config = ObserverConfig(
        calibration_mode='layer_wise',
        bit_type=bit_config,
        observer_type='PercentileObserver'
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

    print("\n=== 테스트 완료 ===")

def test_quant_conv():
    # ========== 1. Config 설정 ==========
    bit_config = BitTypeConfig(bits=8, signed=False, name='int8')
    observer_config = ObserverConfig(
        calibration_mode='layer_wise',
        bit_type=bit_config,
        observer_type='MinmaxObserver'
    )

    # ========== 2. 간단한 CNN 모델 생성 ==========
    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
            
        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = self.conv3(x)
            return x
    
    cnn = SimpleCNN()
    print("Original CNN:")
    for name, module in cnn.named_modules():
        if isinstance(module, nn.Conv2d):
            print(f"  {name}: {module}")

    # ========== 3. QuantConv로 변환 ==========
    quant_layers = []
    for name, layer in cnn.named_modules():
        if isinstance(layer, nn.Conv2d):
            quant_layer = QuantConv(
                quant_args={},
                input_module=layer,
                observer_config=observer_config
            )
            quant_layers.append(quant_layer)
            print(f"Converted: {name}")

    print(f"\nQuantConv 개수: {len(quant_layers)}")

    # ========== 4. 더미 Calibration 데이터 생성 ==========
    num_batches = 64
    batch_size = 32
    channels = 3
    height = 32
    width = 32

    print(f"\n더미 데이터 생성: {num_batches} batches × {batch_size} samples × {channels}x{height}x{width}")
    calib_data = [
        torch.randn(batch_size, channels, height, width) 
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
        print(f"\nLayer {i} (conv{i+1}):")
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
        print(f"\nLayer {i} (conv{i+1}):")
        print(f"  Weight - scale: {w_scale:.8f}, zero: {w_zero:.2f}")
        print(f"  Output - scale: {out_scale:.8f}, zero: {out_zero:.2f}")

    # ========== 8. FP32 vs Fake Quantization 비교 ==========
    print("\n=== FP32 vs Fake Quantization 출력 비교 ===")
    test_input = torch.randn(1, channels, height, width)
    
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
    
    print(f"FP32 output shape: {out_fp32.shape}")
    print(f"Fake Quant output shape: {out_quant.shape}")
    print(f"FP32 output (sample): {out_fp32[0, 0, 0, :5]}")  # 첫 번째 채널, 첫 행, 처음 5개
    print(f"Fake Quant output (sample): {out_quant[0, 0, 0, :5]}")
    print(f"MSE: {F.mse_loss(out_fp32, out_quant).item():.6f}")
    print(f"Max diff: {(out_fp32 - out_quant).abs().max().item():.6f}")

    print("\n=== 테스트 완료 ===")


if __name__ == "__main__":
    print("="*60)
    print("Testing QuantLinear")
    print("="*60)
    test_quant_linear()
    
    print("\n\n")
    print("="*60)
    print("Testing QuantConv")
    print("="*60)
    test_quant_conv()