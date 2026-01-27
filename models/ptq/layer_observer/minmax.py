# Copyright (c) MEGVII Inc. and its affiliates.
# All Rights Reserved.
import torch
import torch.nn as nn
from ..bit_type import BitType 
from .base import BaseObserver

class MinmaxObserver(BaseObserver):
    def __init__(self,
                 bit_type,
                 module_type,
                 calibration_mode,
                 num_heads=None,
                 head_dim=None):
        super(MinmaxObserver, self).__init__(bit_type,
                                             module_type,
                                             calibration_mode,
                                             num_heads=num_heads,
                                             head_dim=head_dim)
        self.symmetric = self.bit_type.symmetric
        self.max_val = None
        self.min_val = None
        self.device = None

    def update(self, v):
        #1. update self.max_val and self.min_val

        #device check 1st time
        if self.device is None:
            self.device = v.device.type
        
        # test device match
        if self.device != v.device.type:
            raise ValueError(
                "Device type mismatch in observer."
                f"Expected device type: {self.device}, but got: {v.device.type}")

        v = self.reshape_tensor(v)
        cur_max = v.max(axis=1).values
        if self.max_val is None:
            self.max_val = cur_max
        else:
            self.max_val = torch.max(cur_max, self.max_val)

        cur_min = v.min(axis=1).values
        if self.min_val is None:
            self.min_val = cur_min
        else:
            self.min_val = torch.min(cur_min, self.min_val)

        if self.calibration_mode == 'layer_wise':
            self.max_val = self.max_val.max()
            self.min_val = self.min_val.min()
        # head_wise: 이미 reshape_tensor에서 (num_heads, ...) 형태로 변환됨
        # 별도 처리 불필요 (channel_wise와 동일하게 head별 min/max 유지)

    def get_quantization_params(self, *args, **kwargs):
        max_val = self.max_val
        min_val = self.min_val

        qmax = self.bit_type.upper_bound
        qmin = self.bit_type.lower_bound

        scale = torch.ones_like(max_val, dtype=torch.float32)
        zero_point = torch.zeros_like(max_val, dtype=torch.int64)

        if self.symmetric:
            #symmetric quant paras
            max_val = torch.max(-min_val, max_val)
            scale = max_val / (float(qmax - qmin) / 2)
            scale.clamp_(self.eps)
            zero_point = torch.zeros_like(max_val, dtype=torch.int64)
        else:
            #asymmetric quant paras
            scale = (max_val - min_val) / float(qmax - qmin)
            scale.clamp_(self.eps)
            zero_point = qmin - torch.round(min_val / scale)
            zero_point.clamp_(qmin, qmax)
        
        return scale, zero_point

