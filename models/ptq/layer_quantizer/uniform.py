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

    def __init__(self, bit_type, module_type, num_heads=None, head_dim=None):
        super(UniformQuantizer, self).__init__(
            bit_type=bit_type,
            module_type=module_type,
            num_heads=num_heads,
            head_dim=head_dim,
        )
        # register_buffer로 ONNX initializer로 인식되게 함
        self.register_buffer('scale', None)
        self.register_buffer('zero_point', None)

    def update_quantization_params(self, scale, zero_point):
        # buffer 업데이트
        self.scale = scale.clone() if isinstance(scale, torch.Tensor) else torch.tensor(scale)
        self.zero_point = zero_point.clone() if isinstance(zero_point, torch.Tensor) else torch.tensor(zero_point)

    def quant(self, inputs, scale=None, zero_point=None):
        if scale is None:
            scale = self.scale
        if zero_point is None:
            zero_point = self.zero_point

        scale = scale.to(inputs.device)
        zero_point = zero_point.to(inputs.device)

        # Standard quantization
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

        # Standard dequantization
        range_shape = self.get_reshape_range(inputs)
        scale = scale.reshape(range_shape)
        zero_point = zero_point.reshape(range_shape)
        outputs = (inputs - zero_point) * scale
        return outputs
    

    