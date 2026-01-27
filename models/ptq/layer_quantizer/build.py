# Copyright (c) MEGVII Inc. and its affiliates. All Rights Reserved.
from .log2 import Log2Quantizer
from .uniform import UniformQuantizer

str2quantizer = {'uniform': UniformQuantizer, 'log2': Log2Quantizer}

def build_quantizer(quantizer_str, bit_type, module_type, num_heads=None, head_dim=None):
    quantizer = str2quantizer[quantizer_str]
    return quantizer(bit_type, module_type, num_heads=num_heads, head_dim=head_dim)

