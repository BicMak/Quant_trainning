import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import copy
from torch import Tensor

from models.ptq.observer_config import QuantConfig, BitTypeConfig
from layer_observer.minmax import MinmaxObserver
from layer_observer.percentile import PercentileObserver
from layer_observer.omse import OmseObserver
from layer_observer.kv_divergence import KVObserver
from models.ptq.bit_type import BitType
import os

import models.ptq.observer_config as observer_config

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
                 observer_config:QuantConfig):
        # observer 초기화
        super(QuantLayer, self).__init__()

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

        #1. set layer type & observer
        init_observers(self.observer_type,
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
    
    # def forward(self, x):
    #     """추론 전용 - 깔끔하게"""
    #     if self.mode == 'fp32':
    #         return self.fwd_func(x, self.weight, self.bias, **self.fwd_kwargs)
        
    #     elif self.mode == 'quantized':
    #         dequant_weight = (self.quant_weight - self.zero) * self.scaler
    #         return self.fwd_func(x, dequant_weight, self.bias, **self.fwd_kwargs)
    
    def compute_quant_params(self):
        """Calibration 끝나고 한 번 호출"""
        self.scaler, self.zero = self.observer.get_quantization_params()
        self.output_scaler, self.output_zero = self.output_observer.get_quantization_params()
        return (self.scaler, self.zero), (self.output_scaler, self.output_zero)

        



class QuantLayer(nn.Module):
    def __init__(self, 
                 quant_args:dict,
                 input_module:nn.Module,
                 observer_config:QuantConfig):
        # observer 초기화
        super(QuantLayer, self).__init__()

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
        self.is_activation = None
        #1. set layer type & observer
        self._init_observers()

        #2. layer initialization  
        self._layer_init()  

        #3. weight and bias save (only for conv, linear)
        if self.module_type in ['conv_weight', 'linear_weight']:
            self.weight = self.input_module.weight.clone().detach()
            self.output_observer = copy.deepcopy(self.observer) #deepcopy for output observer

            if self.input_module.bias != None:
                self.bias = self.input_module.bias.clone().detach()
            else:
                self.bias = torch.zeros(self.input_module.weight.size(0)).to(
                    self.input_module.weight.device
                )

    def _init_observers(self):
        #1. check module type
        if isinstance(self.input_module, nn.Conv2d):
            self.module_type = "conv_weight"
        elif isinstance(self.input_module, nn.Linear):
            self.module_type = "linear_weight"
        else:
            self.module_type = "activation"

        #2. observer initialization
        if self.observer_type == 'MinmaxObserver':
            self.observer = MinmaxObserver(
                bit_type=self.bit_type, 
                module_type=self.module_type,
                calibration_mode=self.calibration_mode
            )
        elif self.observer_type == 'PercentileObserver':
            self.observer = PercentileObserver(
                bit_type=self.bit_type,
                module_type=self.module_type,
                calibration_mode=self.calibration_mode,
                percentile_alpha=self.observer_config.percentile_alpha,
                percentile_sigma=self.observer_config.percentile_sigma
            )
        elif self.observer_type == 'OmseObserver':
            self.observer = OmseObserver(
                bit_type=self.bit_type,
                module_type=self.module_type,
                calibration_mode=self.calibration_mode
            )
        elif self.observer_type == 'KLObserver':
            self.observer = KVObserver(
                bit_type=self.bit_type,
                module_type=self.module_type,
                calibration_mode=self.calibration_mode,
                kl_bins=self.observer_config.kl_bins
            )

    def _layer_init(self):
        if isinstance(self.input_module, nn.Linear):
            self.fwd_kwargs = dict()
            self.fwd_func = F.linear
            self.is_lin = True
            self.is_activation = False

        elif isinstance(self.input_module, nn.Conv2d):
            self.fwd_kwargs = dict(
                stride=self.input_module.stride,
                padding=self.input_module.padding,
                dilation=self.input_module.dilation,
                groups=self.input_module.groups,
            )
            self.fwd_func = F.conv2d
            self.is_lin = False  
            self.is_activation = False

        else:
            if type(self.input_module) in ACTIVATION_MAP:
                if isinstance(self.input_module, nn.LeakyReLU):
                    # 파라미터 있는 경우
                    self.act_func = lambda x: F.leaky_relu(x, self.input_module.negative_slope)
                else:
                    self.act_func = ACTIVATION_MAP[type(self.input_module)]
  
    
    def calibration(self, x):
        """Calibration 전용 - observer update만 수행"""
        if self.is_activation == False:
            # weight observer update
            with torch.no_grad():
                self.observer.update(self.weight)
                output = self.fwd_func(x, self.weight, self.bias, **self.fwd_kwargs)
                self.output_observer.update(output)
        else:
            output = self.act_func(x)
            self.observer.update(output)
            
        return output  # 필요하면 반환
    
    # def forward(self, x):
    #     """추론 전용 - 깔끔하게"""
    #     if self.mode == 'fp32':
    #         return self.fwd_func(x, self.weight, self.bias, **self.fwd_kwargs)
        
    #     elif self.mode == 'quantized':
    #         dequant_weight = (self.quant_weight - self.zero) * self.scaler
    #         return self.fwd_func(x, dequant_weight, self.bias, **self.fwd_kwargs)
    
    def compute_quant_params(self):
        """Calibration 끝나고 한 번 호출"""
        self.scaler, self.zero = self.observer.get_quantization_params()
        if self.is_activation == False:
            self.output_scaler, self.output_zero = self.output_observer.get_quantization_params()
            return (self.scaler, self.zero), (self.output_scaler, self.output_zero)
        else:
            return (self.scaler, self.zero), None
        
def main():
    # ========== 1. Config 설정 ==========
    bit_config = BitTypeConfig(bits=8, signed=False, name='int8')
    observer_config = QuantConfig(
        calibration_mode='layer_wise',
        bit_type=bit_config,
        observer_type='MinmaxObserver'
    )

    # ========== 2. AlexNet 로드 ==========
    alexnet = models.alexnet(pretrained=False)
    print("Original AlexNet:")
    print(alexnet.features)

    # ========== 3. QuantLayer로 변환 ==========
    quant_layers = []
    for layer in alexnet.features:
        if isinstance(layer, (nn.Conv2d, nn.ReLU, nn.MaxPool2d)):
            if isinstance(layer, nn.MaxPool2d):
                # MaxPool은 일단 스킵 (나중에 처리)
                continue
            quant_layer = QuantLayer(
                quant_args={},  # 빈 dict (나중에 제거 가능)
                input_module=layer,
                observer_config=observer_config
            )
            quant_layers.append(quant_layer)

    print(f"\nQuantLayer 개수: {len(quant_layers)}")

    # ========== 4. 더미 Calibration 데이터 생성 ==========
    num_batches = 64
    batch_size = 32
    img_size = 224
    channels = 3

    print(f"\n더미 데이터 생성: {num_batches} batches × {batch_size} samples")
    calib_data = [
        torch.randn(batch_size, channels, img_size, img_size) 
        for _ in range(num_batches)
    ]

    # ========== 5. Calibration 수행 ==========
    print("\n=== Calibration 시작 ===")
    for batch_idx, x in enumerate(calib_data):
        out = x
        for layer_idx, layer in enumerate(quant_layers):
            out = layer.calibration(out)
        
        if (batch_idx + 1) % 10 == 0:
            print(f"Batch {batch_idx + 1}/{num_batches} 완료")

    print("\n=== Calibration 완료 ===")

    # ========== 6. Observer 통계 확인 ==========
    print("\n=== Observer 통계 ===")
    for i, layer in enumerate(quant_layers):
        print(f"\nLayer {i} ({layer.module_type}):")
        print(f"  Observer type: {type(layer.observer).__name__}")
        print(f"  min_val: {layer.observer.min_val}")
        print(f"  max_val: {layer.observer.max_val}")
        
        if layer.module_type in ['conv_weight', 'linear_weight']:
            print(f"  Weight shape: {layer.weight.shape}")
            print(f"  Weight range: [{layer.weight.min().item():.4f}, {layer.weight.max().item():.4f}]")

    # ========== 7. Quantization Params 계산 ==========
    print("\n=== Quantization Parameters 계산 ===")
    for i, layer in enumerate(quant_layers):
        result = layer.compute_quant_params()
        print(f"\nLayer {i}:")
        if layer.is_activation == False:
            (w_scale, w_zero), (out_scale, out_zero) = result
            print(f"  Weight - scale: {w_scale}, zero: {w_zero}")
            print(f"  Output - scale: {out_scale}, zero: {out_zero}")
        else:
            (act_scale, act_zero), _ = result
            print(f"  Activation - scale: {act_scale}, zero: {act_zero}")

    print("\n=== 테스트 완료 ===")


if __name__ == "__main__":
    main()