# Copyright (c) MEGVII Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn as nn

from .base import BaseQuantizer


class UniformQuantizer(BaseQuantizer):
    """
    Uniform Quantizer for standard linear quantization.

    Implements: q = round(x / scale + zero_point)
                x' = (q - zero_point) * scale
    """

    def __init__(self, bit_type, module_type):
        super(UniformQuantizer, self).__init__(
            bit_type=bit_type,
            module_type=module_type,
        )
        self.scale = None
        self.zero_point = None

    def update_quantization_params(self, scale, zero_point):
        self.scale, self.zero_point = scale, zero_point

    def quant(self, inputs, scale=None, zero_point=None):
        
        #check the None of parameter
        if scale is None:
            scale = self.scale
        if zero_point is None:
            zero_point = self.zero_point

        scale = scale.to(inputs.device)
        zero_point = zero_point.to(inputs.device)
        

        range_shape = self.get_reshape_range(inputs)
        scale = scale.reshape(range_shape)
        zero_point = zero_point.reshape(range_shape)
        outputs = inputs / scale + zero_point
        outputs = outputs.round().clamp(self.bit_type.lower_bound,
                                        self.bit_type.upper_bound)
        
        return outputs

    def dequantize(self, inputs, scale=None, zero_point=None):
        if scale is None:
            scale = self.scale
        if zero_point is None:
            zero_point = self.zero_point
        
        scale = scale.to(inputs.device)
        zero_point = zero_point.to(inputs.device)
                    
        range_shape = self.get_reshape_range(inputs)
        scale = scale.reshape(range_shape)
        zero_point = zero_point.reshape(range_shape)
        outputs = (inputs - zero_point) * scale
        return outputs
    

    