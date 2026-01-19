import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import copy
from torch import Tensor

from models.ptq.observer_config import ObserverConfig, BitTypeConfig
from layer_observer.minmax import MinmaxObserver

from models.ptq.bit_type import BitType
import os

from layer_quantizer.uniform import UniformQuantizer
from layer_quantizer.build import build_quantizer


def test_quant_linear():
    # ========== 1. Config 설정 ==========
    bit_config = BitTypeConfig(bits=8, signed=False, name='int8')
    observer_config = ObserverConfig(
        calibration_mode='layer_wise',
        bit_type=bit_config,
        observer_type='PercentileObserver'
    )
    # ====================================

    # ========== 2. QuantLinear 생성 ==========
    build_quantizer(quantizer_str='uniform', 
                    bit_type=bit_config,
                    observer=observer_config,
                    module_type='linear_weight')



if __name__ == "__main__":
    print("="*60)
    print("Testing QuantLinear")
    print("="*60)
    test_quant_linear()
    
