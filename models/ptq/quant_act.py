import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from torch import Tensor
import os

from quant_config import QuantConfig, BitTypeConfig
from .bit_type import BitType
from .layer_quantizer.build import build_quantizer
from .utils import init_observers

class QAct(nn.Module):
    def __init__(self,
                 quant_config:QuantConfig,
                 act_module:nn.Module = None):
        super().__init__()

        # quant_config에서 설정 추출
        self.act_module = act_module

        self.quant_config = quant_config
        self.observer_type = quant_config.observer_type
        self.bit_type = BitType(
            bits=quant_config.bit_type.bits,
            signed=quant_config.bit_type.signed,
            name=quant_config.bit_type.name
        )
        self.calibration_mode = quant_config.calibration_mode
        self.mode = 'fp32'

        #1. set layer type & observer
        self.observer = init_observers(self.observer_type,
                                        self.bit_type,
                                        'activation',
                                        self.calibration_mode,
                                        self.quant_config)

        #2. quantizer build
        self.quantizer = build_quantizer(
            quantizer_str='uniform',
            bit_type=self.bit_type,
            module_type='activation')


    def calibration(self, x):
        """Calibration 전용 - observer update만 수행"""
        with torch.no_grad():
            if self.act_module is not None:
                x = self.act_module(x)
            self.observer.update(x)
            output = x

        return output

    def compute_quant_params(self):
        """Calibration 끝나고 한 번 호출"""
        self.scaler, self.zero = self.observer.get_quantization_params()

        # quantizer에 quantization param 설정
        self.quantizer.update_quantization_params(
            self.scaler, self.zero
            )

        return (self.scaler, self.zero)


    def forward(self, x):
        if self.act_module is not None:
            x = self.act_module(x)

        if self.mode == 'quantized':
            x = self.quantizer.quant(x)
        
        return x

