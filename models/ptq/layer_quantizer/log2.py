# Copyright (c) MEGVII Inc. and its affiliates. All Rights Reserved.
import torch

from .base import BaseQuantizer


class Log2Quantizer(BaseQuantizer):
    """
    Log2 Quantizer for I-BERT style softmax output.

    Quantizes values to power-of-2 representation for bit-shift friendly computation.
    Used after softmax where outputs are in (0, 1] range.
    """

    def __init__(self, bit_type, module_type, num_heads=None, head_dim=None):
        super(Log2Quantizer, self).__init__(
            bit_type=bit_type,
            module_type=module_type,
            num_heads=num_heads,
            head_dim=head_dim,
        )
        self.softmax_mask = None

    def quant(self, inputs, scale=None, zero_point=None):
        """
        Quantize softmax outputs to log2 representation.

        Args:
            inputs: Softmax output tensor in (0, 1] range
            scale: Not used (for interface compatibility)
            zero_point: Not used (for interface compatibility)
        """
        rounds = torch.round(-1 * inputs.log2())
        self.softmax_mask = rounds >= 2**self.bit_type.bits
        outputs = torch.clamp(rounds, 0, 2**self.bit_type.bits - 1)
        return outputs

    def dequantize(self, inputs, scale=None, zero_point=None):
        """
        Dequantize log2 representation back to probability.

        Args:
            inputs: Log2 quantized tensor
            scale: Not used (for interface compatibility)
            zero_point: Not used (for interface compatibility)
        """
        outputs = 2**(-1 * inputs)
        outputs[self.softmax_mask] = 0
        return outputs