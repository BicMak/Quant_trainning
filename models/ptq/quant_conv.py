import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch import Tensor
import os

from quant_config import QuantConfig, BitTypeConfig
from .bit_type import BitType
from .layer_quantizer.build import build_quantizer
from .utils import init_observers
from .layer_profiler.profiler import profiler


class QuantConv2d(nn.Module):
    def __init__(self,
                 input_module: nn.Conv2d,
                 quant_config: QuantConfig,
                 layer_name: str = 'qconv',
                 enable_profiling: bool = False):
        super().__init__()

        # quant_config에서 설정 추출

        self.layer_name = layer_name
        self.enable_profiling = enable_profiling
        self.quant_config = quant_config
        self.observer_type = quant_config.observer_type
        self.bit_type = BitType(
            bits=quant_config.bit_type.bits,
            symmetric=quant_config.bit_type.symmetric,
            name=quant_config.bit_type.name
        )

        # Hardcoded weight bit type (symmetric=True for weights)
        self.weight_bit_type = BitType(
            bits=quant_config.bit_type.bits,
            symmetric=True,  # Hardcoded for weights
            name=f"int{quant_config.bit_type.bits}_weight"
        )

        self.calibration_mode = quant_config.calibration_mode
        self.quant_weight = None
        self.mode = 'fp32'

        #1. set layer type & observer
        self.observer = init_observers("MinmaxObserver",  # Hardcoded
                                        self.weight_bit_type,  # Hardcoded symmetric=True
                                        'conv_weight',  # Hardcoded
                                        self.calibration_mode,
                                        self.quant_config)
        # output_observer는 별도로 초기화 (activation 타입으로)
        self.output_observer = init_observers(self.observer_type,
                                              self.bit_type,
                                              'activation',
                                              self.calibration_mode,
                                              self.quant_config)

        #2. quantizer build
        self.quantizer = build_quantizer(
            quantizer_str='uniform',
            bit_type=self.weight_bit_type,  # Use hardcoded weight bit type
            module_type='conv_weight')
        self.output_quantizer = build_quantizer(
            quantizer_str='uniform',
            bit_type=self.bit_type,
            module_type='activation')

        #3. profiler 초기화
        self.profiler = None
        self.weight_profiler = None
        if self.enable_profiling:
            self.profiler = profiler(layer_name + '_output')
            self.weight_profiler = profiler(layer_name + '_weight')

        #4. layer initialization
        self.fwd_kwargs = dict(
            stride=input_module.stride,
            padding=input_module.padding,
            dilation=input_module.dilation,
            groups=input_module.groups,
        )
        self.fwd_func = F.conv2d

        self.weight = input_module.weight.clone().detach()
        if input_module.bias is not None:
            self.bias = input_module.bias.clone().detach()
        else:
            self.bias = torch.zeros(input_module.weight.size(0)).to(
                input_module.weight.device
            )

    def calibration(self, x):
        """Calibration 전용 - observer update만 수행"""
        with torch.no_grad():
            self.observer.update(self.weight)
            output = self.fwd_func(x, self.weight, self.bias, **self.fwd_kwargs)
            self.output_observer.update(output)

        return output

    def compute_quant_params(self):
        """Calibration 끝나고 한 번 호출"""
        self.scaler, self.zero = self.observer.get_quantization_params()
        self.output_scaler, self.output_zero = self.output_observer.get_quantization_params()

        # Device 통일: weight의 device로 모든 파라미터 이동
        weight_device = self.weight.device
        self.scaler = self.scaler.to(weight_device)
        self.zero = self.zero.to(weight_device)
        self.output_scaler = self.output_scaler.to(weight_device)
        self.output_zero = self.output_zero.to(weight_device)

        # quantizer에 quantization param 설정
        self.quantizer.update_quantization_params(
            self.scaler, self.zero
        )
        self.output_quantizer.update_quantization_params(
            self.output_scaler, self.output_zero
        )

        # weight quantization, save quant_weight
        # scaler와 zero를 올바른 shape으로 reshape
        range_shape = self.quantizer.get_reshape_range(self.weight)
        scaler_reshaped = self.scaler.reshape(range_shape)
        zero_reshaped = self.zero.reshape(range_shape)

        self.quant_weight = torch.clamp(
            torch.round(self.weight / scaler_reshaped) + zero_reshaped,
            min=self.weight_bit_type.lower_bound,
            max=self.weight_bit_type.upper_bound
        )

        # profiler에 weight 업데이트 (FP32 vs Quantized weight 비교)
        if self.enable_profiling and self.weight_profiler is not None:
            # Manual dequantization: (quant_weight - zero) * scale
            dequant_weight = (self.quant_weight - zero_reshaped) * scaler_reshaped
            self.weight_profiler.update_weight(self.weight, dequant_weight.detach())

        return (self.scaler, self.zero), (self.output_scaler, self.output_zero)

    def get_profiler(self):
        """Get profiler objects (output and weight)"""
        if self.enable_profiling:
            return {
                'output': self.profiler,
                'weight': self.weight_profiler
            }
        return None

    def get_profiling_results(self):
        """
        Get all profiling results after forward pass.

        Returns:
            dict: Dictionary containing statistics, histogram, time, and memory records
                  for both output and weight quantization
                  Returns None if profiling is not enabled

        Usage:
            layer.forward(x)  # Automatically updates profiler
            results = layer.get_profiling_results()
            print(results['output']['statistics']['qsnr'])
            print(results['weight']['statistics']['qsnr'])
        """
        if not self.enable_profiling:
            return None

        results = {}

        # Output profiling results
        if self.profiler is not None:
            try:
                output_stats = self.profiler.get_statistic() if self.profiler.weight is not None else None
                output_hist = self.profiler.get_hist() if self.profiler.weight is not None else None
            except (ValueError, AttributeError):
                output_stats = None
                output_hist = None

            results['output'] = {
                'statistics': output_stats,
                'histogram': output_hist,
                'time': self.profiler.get_time_record(),
                'memory': self.profiler.get_memory_record()
            }

        # Weight profiling results
        if self.weight_profiler is not None:
            try:
                weight_stats = self.weight_profiler.get_statistic() if self.weight_profiler.weight is not None else None
                weight_hist = self.weight_profiler.get_hist() if self.weight_profiler.weight is not None else None
            except (ValueError, AttributeError):
                weight_stats = None
                weight_hist = None

            results['weight'] = {
                'statistics': weight_stats,
                'histogram': weight_hist,
                'time': self.weight_profiler.get_time_record(),
                'memory': self.weight_profiler.get_memory_record()
            }

        return results

    def reset_profiling(self):
        """Reset profiling data for both output and weight"""
        if self.enable_profiling:
            if self.profiler is not None:
                self.profiler.reset_time_profiler()
                self.profiler.reset_memory_profiler()
                self.profiler.weight = None
                self.profiler.quant_weight = None

            if self.weight_profiler is not None:
                self.weight_profiler.reset_time_profiler()
                self.weight_profiler.reset_memory_profiler()
                self.weight_profiler.weight = None
                self.weight_profiler.quant_weight = None


    def forward(self, x):
        if self.mode == 'quantized':
            if self.enable_profiling and self.profiler is not None:
                # Measure quantization time
                with self.profiler.measure_time():
                    # 1. dequantize weights (int8 -> fp32)
                    dequant_weight = self.quantizer.forward(self.quant_weight)

                    # 2. Conv operation in fp32
                    out_fp32 = self.fwd_func(x, dequant_weight, self.bias, **self.fwd_kwargs)

                    # Store FP32 output before quantization
                    fp32_output = out_fp32.clone().detach()

                    # 3. Output fake quantization
                    x = self.output_quantizer.forward(out_fp32)

                    # Update profiler with FP32 vs Quantized outputs
                    self.profiler.update_weight(fp32_output, x.detach())
            else:
                # 1. dequantize weights (int8 -> fp32)
                dequant_weight = self.quantizer.forward(self.quant_weight)

                # 2. conv operation in fp32
                x = self.fwd_func(x, dequant_weight, self.bias, **self.fwd_kwargs)

                # 3. Output fake quantization
                x = self.output_quantizer.forward(x)

            return x

        else:  # fp32
            if self.enable_profiling and self.profiler is not None:
                # FP32 mode - measure time without quantization
                with self.profiler.measure_time():
                    pass  # No quantization, just time measurement

            return self.fwd_func(x, self.weight, self.bias, **self.fwd_kwargs)

