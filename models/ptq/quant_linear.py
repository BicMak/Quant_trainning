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

class QuantLinear(nn.Module):
    def __init__(self,
                 input_module: nn.Module,
                 out_config: QuantConfig,
                 weight_config: QuantConfig,
                 layer_name: str = 'qlinear'):
        super().__init__()

        # Config 설정 (input_module 참조 저장하지 않음 - 메모리 절약)
        self.layer_name = layer_name
        self.out_config = out_config
        self.weight_config = weight_config
        self.enable_profiling = out_config.enable_profiler
        self.output_quant_enable = out_config.output_quant_enable

        # Bit type 설정
        self.out_bit_type = BitType(
            bits=out_config.bit_type.bits,
            symmetric=out_config.bit_type.symmetric,
            name=out_config.bit_type.name
        )
        self.weight_bit_type = BitType(
            bits=weight_config.bit_type.bits,
            symmetric=weight_config.bit_type.symmetric,
            name=weight_config.bit_type.name
        )

        self.quant_weight = None
        self.mode = 'fp32'

        # Quantization params - register_buffer로 ONNX initializer로 인식되게 함
        self.register_buffer('weight_scaler', None)
        self.register_buffer('weight_zero', None)
        self.register_buffer('output_scaler', None)
        self.register_buffer('output_zero', None)

        # 1. Observers 초기화
        # Weight observer (weight config 사용)
        self.weight_observer = init_observers(
            weight_config.observer_type,
            self.weight_bit_type,
            'linear_weight',
            weight_config.calibration_mode,
            weight_config
        )

        # Output observer (output config 사용)
        self.output_observer = init_observers(
            out_config.observer_type,
            self.out_bit_type,
            'activation',
            out_config.calibration_mode,
            out_config
        )

        # 2. Quantizers 초기화
        self.weight_quantizer = build_quantizer(
            quantizer_str=weight_config.quantization_method.lower(),
            bit_type=self.weight_bit_type,
            module_type='linear_weight'
        )
        self.output_quantizer = build_quantizer(
            quantizer_str=out_config.quantization_method.lower(),
            bit_type=self.out_bit_type,
            module_type='activation'
        )

        # 3. Profiler 초기화 (output과 weight 각각)
        self.output_profiler = None
        self.weight_profiler = None
        if self.enable_profiling:
            self.output_profiler = profiler(layer_name + '_output')
            self.weight_profiler = profiler(layer_name + '_weight')

        # 4. Layer 초기화 - 필요한 값만 복사하고 원본 참조는 저장하지 않음
        self.fwd_kwargs = dict()
        self.fwd_func = F.linear

        self.weight = input_module.weight.clone().detach()

        if input_module.bias is not None:
            self.bias = input_module.bias.clone().detach()
        else:
            self.bias = torch.zeros(input_module.weight.size(0)).to(
                input_module.weight.device
            )

        # Weight는 고정값이므로 바로 quantize
        self.quantize_weight()

    def quantize_weight(self):
        """
        Weight quantization 전용 메서드.
        Weight는 고정값이므로 초기화 시 한 번만 실행.
        """
        with torch.no_grad():
            # Weight observer 업데이트
            self.weight_observer.update(self.weight)

            # Quantization params 계산
            weight_scaler, weight_zero = self.weight_observer.get_quantization_params()

            # Device 통일
            weight_device = self.weight.device
            weight_scaler = weight_scaler.to(weight_device)
            weight_zero = weight_zero.to(weight_device)

            # Quantizer에 params 설정
            self.weight_quantizer.update_quantization_params(weight_scaler, weight_zero)

            # Weight quantization 수행
            range_shape = self.weight_quantizer.get_reshape_range(self.weight)
            scaler_reshaped = weight_scaler.reshape(range_shape)
            zero_reshaped = weight_zero.reshape(range_shape)

            self.quant_weight = torch.clamp(
                torch.round(self.weight / scaler_reshaped) + zero_reshaped,
                min=self.weight_bit_type.lower_bound,
                max=self.weight_bit_type.upper_bound
            )

            # Weight profiler 업데이트
            if self.enable_profiling and self.weight_profiler is not None:
                dequant_weight = (self.quant_weight - zero_reshaped) * scaler_reshaped
                self.weight_profiler.update_weight(self.weight, dequant_weight.detach())

            # Quantization params 저장
            self.weight_scaler = weight_scaler
            self.weight_zero = weight_zero

    def calibration(self, x):
        """
        Calibration 전용 - output activation만 처리.
        Weight는 이미 quantize_weight()에서 처리됨.
        """
        with torch.no_grad():
            # FP32 forward로 output 계산
            output = self.fwd_func(x, self.weight, self.bias, **self.fwd_kwargs)

            # Output observer 업데이트 (activation calibration)
            # output_quant_enable이 True인 경우에만 observer 업데이트
            if self.output_quant_enable:
                self.output_observer.update(output)

        return output

    def compute_output_quant_params(self):
        """
        Calibration 끝난 후 output quantization params 계산.
        Weight는 이미 quantize_weight()에서 처리됨.

        Returns:
            tuple: (weight_params, output_params)
                - weight_params: (scaler, zero) - 이미 계산된 값
                - output_params: (scaler, zero) - 새로 계산한 값 (output_quant_enable=False면 None)
        """
        # Output quantization이 활성화된 경우에만 params 계산
        if self.output_quant_enable:
            # Output quantization params 계산
            output_scaler, output_zero = self.output_observer.get_quantization_params()

            # Device 통일
            weight_device = self.weight.device
            output_scaler = output_scaler.to(weight_device)
            output_zero = output_zero.to(weight_device)

            # Output quantizer에 params 설정
            self.output_quantizer.update_quantization_params(output_scaler, output_zero)

            # Params 저장
            self.output_scaler = output_scaler
            self.output_zero = output_zero
        else:
            # Output quantization이 비활성화된 경우 None 반환
            self.output_scaler = None
            self.output_zero = None

        # Weight params는 이미 quantize_weight()에서 계산됨
        return (self.weight_scaler, self.weight_zero), (self.output_scaler, self.output_zero)

    def get_profiler(self):
        """
        Get profiler objects (output and weight).

        Returns:
            dict or None: {'output': output_profiler, 'weight': weight_profiler}
        """
        if self.enable_profiling:
            return {
                'output': self.output_profiler,
                'weight': self.weight_profiler
            }
        return None

    def get_profiling_results(self):
        """
        Get all profiling results after forward pass.

        Returns:
            dict: Dictionary with separate results for output and weight
                {
                    'output': {'statistics': ..., 'histogram': ..., 'time': ..., 'memory': ...},
                    'weight': {'statistics': ..., 'histogram': ..., 'time': ..., 'memory': ...}
                }
                Returns None if profiling is not enabled

        Usage:
            # Weight는 초기화 시 자동으로 profiling됨
            layer = QuantLinear(...)  # quantize_weight() 자동 호출

            # Calibration으로 output 수집
            layer.calibration(x)
            layer.compute_output_quant_params()

            # Inference에서 profiling
            layer.mode = 'quantized'
            layer.forward(x)

            # 결과 조회
            results = layer.get_profiling_results()
            print(results['output']['statistics']['qsnr'])
            print(results['weight']['statistics']['qsnr'])
        """
        if not self.enable_profiling:
            return None

        results = {}

        # Output profiling results
        if self.output_profiler is not None:
            try:
                output_stats = self.output_profiler.get_statistic() if len(self.output_profiler.weight_batch_list) > 0 else None
                output_hist = self.output_profiler.get_hist() if len(self.output_profiler.weight_batch_list) > 0 else None
            except (ValueError, AttributeError):
                output_stats = None
                output_hist = None

            results['output'] = {
                'statistics': output_stats,
                'histogram': output_hist,
                'time': self.output_profiler.get_time_record(),
                'memory': self.output_profiler.get_memory_record()
            }

        # Weight profiling results
        if self.weight_profiler is not None:
            try:
                weight_stats = self.weight_profiler.get_statistic() if len(self.weight_profiler.weight_batch_list) > 0 else None
                weight_hist = self.weight_profiler.get_hist() if len(self.weight_profiler.weight_batch_list) > 0 else None
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
            if self.output_profiler is not None:
                self.output_profiler.reset_time_profiler()
                self.output_profiler.reset_memory_profiler()
                self.output_profiler.clear_batches()

            if self.weight_profiler is not None:
                self.weight_profiler.reset_time_profiler()
                self.weight_profiler.reset_memory_profiler()
                self.weight_profiler.clear_batches()


    def forward(self, x):
        """
        Forward pass with separate weight and output quantization.

        Mode:
            - 'fp32': FP32 inference (no quantization)
            - 'quantized': Quantized inference (weight quantization always, output quantization if enabled)
        """
        if self.mode == 'quantized':
            # 1. Dequantize weights (int8 -> fp32)
            # NOTE: quant_weight is already quantized, so we only need dequantize()
            # DO NOT use forward() which calls quant() then dequantize() (double quantization!)
            dequant_weight = self.weight_quantizer.dequantize(self.quant_weight)

            # 2. Linear operation in fp32
            out_fp32 = self.fwd_func(x, dequant_weight, self.bias, **self.fwd_kwargs)

            # 3. Output quantization (선택적)
            if self.output_quant_enable:
                # Output quantization이 활성화된 경우
                if self.enable_profiling and self.output_profiler is not None:
                    with self.output_profiler.measure_time():
                        # Store FP32 output before quantization
                        fp32_output = out_fp32.clone().detach()

                        # Output fake quantization
                        output = self.output_quantizer.forward(out_fp32)

                        # Update output profiler with FP32 vs Quantized outputs
                        self.output_profiler.update_weight(fp32_output, output.detach())
                else:
                    # Output fake quantization
                    output = self.output_quantizer.forward(out_fp32)
            else:
                # Output quantization이 비활성화된 경우: FP32 출력 그대로 사용
                output = out_fp32

                # Profiling이 활성화된 경우에도 시간만 측정
                if self.enable_profiling and self.output_profiler is not None:
                    with self.output_profiler.measure_time():
                        pass  # No quantization, just time measurement

            return output

        else:  # fp32
            if self.enable_profiling and self.output_profiler is not None:
                # FP32 mode - measure time without quantization
                with self.output_profiler.measure_time():
                    pass  # No quantization, just time measurement

            return self.fwd_func(x, self.weight, self.bias, **self.fwd_kwargs)


