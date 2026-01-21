import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from .ptq.quant_linear import QuantLinear
from .ptq.quant_intSoft import QuantIntSoft
from .ptq.quant_act import QAct
from .ptq.quant_layernorm import QLayerNorm
from quant_config import QuantConfig, BitTypeConfig

from torch.nn.modules.container import Sequential


class QuantTimmVitBlock(nn.Module):
    """
    Quantized ViT Block for timm models.
    timm의 vit_base_patch16_224 등에서 사용.
    """

    def __init__(self,
                 block,  # timm.models.vision_transformer.Block
                 quant_config: QuantConfig):
        super().__init__()

        self.original_block = block

        # === Attention 부분 ===
        # timm은 qkv가 하나로 합쳐져 있음 (in_features → 3 * out_features)
        self.attn_qkv = QuantLinear(
            input_module=block.attn.qkv,
            quant_config=quant_config
        )
        self.attn_qkv_output = QAct(
            act_module = None,
            quant_config=quant_config
        )
        self.attn_qkv = QuantLinear(
            input_module=block.attn.qkv,
            quant_config=quant_config
        )

        #
        self.sv_attn= QAct(
            input_module=None,
            quant_config=quant_config
        )
        self.residual1= QAct(
            input_module=None,
            quant_config=quant_config
        )
        self.residual2= QAct(
            input_module=None,
            quant_config=quant_config
        )

        # === MLP 부분 ===
        self.mlp_fc1 = QuantLinear(
            input_module=block.mlp.fc1,
            quant_config=quant_config
        )
        self.mlp_act = QAct(
            act_module = block.mlp.act,
            quant_config=quant_config
        )
        self.mlp_fc2 = QuantLinear(
            input_module=block.mlp.fc2,
            quant_config=quant_config
        )
        self.mlp_act2 = QAct(
            act_module = block.mlp.act,
            quant_config=quant_config
        )

        # === LayerNorm ===
        self.norm1 = QLayerNorm(
            input_module=block.norm1,
            quant_config=quant_config
        )
        self.norm2 = QLayerNorm(
            input_module=block.norm2,
            quant_config=quant_config
        )

        #QintSoftmax 
        self.intSoft = QuantIntSoft(
            input_module = None,
            quant_config = quant_config
        ) 


        # === 그대로 쓰는 것들 ===
        self.attn_drop = block.attn.attn_drop
        self.proj_drop = block.attn.proj_drop
        
        self.drop_path = block.drop_path

        # Attention 파라미터
        self.num_heads = block.attn.num_heads
        self.head_dim = block.attn.head_dim
        self.scale = block.attn.scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with quantized layers."""
        # === Attention block ===
        B, N, C = x.shape

        # norm1 → qkv
        x_norm = self.norm1.forward(x)
        qkv = self.attn_qkv.forward(x_norm)
        qkv = self.attn_qkv_output.forward(qkv)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # (B, num_heads, N, head_dim)

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.intSoft.forward(attn)
        attn = self.attn_drop(attn)

        # Combine heads
        x_attn = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x_attn = self.attn_proj.forward(x_attn)
        x_attn = self.sv_attn.forward(x_attn)
        x_attn = self.proj_drop(x_attn)

        # Residual 1
        x = x + self.drop_path(x_attn)
        x = self.residual1.forward(x)

        # === MLP block ===
        x_norm = self.norm2.forward(x)
        x_mlp = self.mlp_fc1.forward(x_norm)
        x_mlp = self.mlp_act.forward(x_mlp)
        x_mlp = self.mlp_fc2.forward(x_mlp)

        # Residual 2
        x = x + self.drop_path(x_mlp)
        x = self.residual2.forward(x)

        return x

    def get_quantized_layers(self) -> Dict[str, nn.Module]:
        """Returns all quantized layers."""
        return {
            'attn_qkv': self.attn_qkv,
            'attn_proj': self.attn_proj,
            'mlp_fc1': self.mlp_fc1,
            'mlp_fc2': self.mlp_fc2,
            'norm1': self.norm1,
            'norm2': self.norm2,
        }

    def calibration(self, x: torch.Tensor) -> torch.Tensor:
        """Calibration pass to collect statistics."""
        B, N, C = x.shape

        # Power-of-Two Factor (PTF)
        # int8 _> int8
        x_norm = self.norm1.calibration(x)

        # Power-of-Two Factor (PTF)
        # int8 -> int32 -> int8
        qkv = self.attn_qkv.calibration(x_norm)
        qkv = self.attn_qkv_output.calibration(qkv)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)


        # Attn (K*Q)
        # int8 -> int32 
        attn = (q @ k.transpose(-2, -1)) * self.scale
        # Log-Int-Softmax (LIS)
        # int32 -> int8
        attn = self.intSoft.calibration(attn)
        attn = self.attn_drop(attn)

        # Attn (S*V)
        # int8 -> int32 -> int8
        x_attn = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x_attn = self.attn_proj.calibration(x_attn)
        x_attn = self.sv_attn.calibration(x_attn)
        x_attn = self.proj_drop(x_attn)
        
        # residual
        # int8 -> int8
        x = x + self.drop_path(x_attn)
        x = self.residual1.calibration(x)

        # MLP
        x_norm = self.norm2.calibration(x)
        x_mlp = self.mlp_fc1.calibration(x_norm)
        x_mlp = self.mlp_act.calibration(x_mlp)
        x_mlp = self.mlp_fc2.calibration(x_mlp)
        x_mlp = self.mlp_act2.calibration(x_mlp)

        x = x + self.drop_path(x_mlp)
        x = self.residual2.calibration(x)

        return x

    def compute_quant_params(self):
        """Compute quantization parameters for all layers."""
        for layer in self.get_quantized_layers().values():
            if hasattr(layer, 'compute_quant_params'):
                layer.compute_quant_params()

    def set_mode(self, mode: str):
        """Set quantization mode ('fp32' or 'quantized')."""
        for layer in self.get_quantized_layers().values():
            if hasattr(layer, 'mode'):
                layer.mode = mode




