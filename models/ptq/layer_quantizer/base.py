# Copyright (c) MEGVII Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn as nn


class BaseQuantizer(nn.Module):

    def __init__(self, bit_type, module_type, num_heads=None, head_dim=None):
        super(BaseQuantizer, self).__init__()
        self.bit_type = bit_type
        self.module_type = module_type
        self.num_heads = num_heads
        self.head_dim = head_dim

    def get_reshape_range(self, inputs):
        range_shape = None
        if self.module_type == 'conv_weight':
            range_shape = (-1, 1, 1, 1)
        elif self.module_type == 'linear_weight':
            range_shape = (-1, 1)
        elif self.module_type == 'activation':
            if len(inputs.shape) == 2:
                range_shape = (1, -1)
            elif len(inputs.shape) == 3:
                range_shape = (1, 1, -1)
            elif len(inputs.shape) == 4:
                range_shape = (1, -1, 1, 1)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        return range_shape

    def update_quantization_params(self, scale, zero_point):
        pass

    def quant(self, inputs, scale=None, zero_point=None):
        raise NotImplementedError

    def dequantize(self, inputs, scale=None, zero_point=None):
        raise NotImplementedError

    def forward(self, inputs):
        outputs = self.quant(inputs)
        outputs = self.dequantize(outputs)
        return outputs