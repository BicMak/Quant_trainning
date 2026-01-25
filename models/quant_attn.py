import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Union
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import os

from .ptq.quant_linear import QuantLinear
from .ptq.quant_intSoft import QuantIntSoft
from .ptq.quant_act import QAct
from .ptq.layer_profiler.profiler import profiler
from quant_config import QuantConfig, LayerQuantConfig
from utils.config_loader import load_config_from_yaml



class QAttn(nn.Module):
    """
    Quantized ViT Block for timm models.
    timm의 vit_base_patch16_224 등에서 사용.
    """

    def __init__(self,
                 block,  # timm.models.vision_transformer.Block
                 quant_config: Union[QuantConfig, LayerQuantConfig, str, Path] = None,
                 enable_profiling: bool = False):
        super().__init__()

        self.original_block = block #attn
        self.enable_profiling = enable_profiling

        # Config 처리: YAML 파일 경로, LayerQuantConfig, 또는 단일 QuantConfig
        if isinstance(quant_config, (str, Path)):
            # YAML 파일 경로가 주어진 경우
            layer_config = load_config_from_yaml(quant_config)
        elif isinstance(quant_config, LayerQuantConfig):
            # LayerQuantConfig 객체가 주어진 경우
            layer_config = quant_config
        elif isinstance(quant_config, QuantConfig):
            # 단일 QuantConfig가 주어진 경우 (기존 방식)
            layer_config = LayerQuantConfig(
                default_config=quant_config,
                layer_configs={}
            )
        else:
            raise ValueError(
                "quant_config must be a YAML file path, LayerQuantConfig, or QuantConfig"
            )

        # === Attention 부분 ===
        # timm은 qkv가 하나로 합쳐져 있음 (in_features → 3 * out_features)
        self.attn_qkv = QuantLinear(
            input_module=block.attn.qkv,
            quant_config=layer_config.get_config('attn_qkv')
        )
        self.attn_qkv_out = QAct(
            quant_config=layer_config.get_config('attn_qkv_out'),
            act_module=None
        )
        self.attn_proj = QuantLinear(
            input_module=block.attn.proj,
            quant_config=layer_config.get_config('attn_proj')
        )

        # kv_act: IntSoftmax에 스칼라 scale을 전달해야 하므로 layer_wise 필요
        # YAML에서 layer_wise로 설정되어 있을 것으로 예상
        self.attn_kv_out= QAct(
            quant_config=layer_config.get_config('attn_kv_out'),
            act_module=None
        )
        # attn_v_output: Attention @ Value 출력 양자화 (FQ-ViT qact2)
        self.attn_v_out = QAct(
            quant_config=layer_config.get_config('attn_v_out'),
            act_module=None
        )



        #QintSoftmax
        self.intSoft = QuantIntSoft(
            input_module = None,
            quant_config = layer_config.get_config('intSoft')
        )


        # Attention 파라미터
        self.num_heads = block.attn.num_heads
        self.head_dim = block.attn.head_dim
        self.attn_dim = self.head_dim * self.num_heads
        self.scale = block.attn.scale

        # === Profiler 초기화 ===
        self.profilers = {}
        if self.enable_profiling:
            self._init_profilers()

    def _init_profilers(self):
        """Initialize profilers for each quantized layer"""
        layer_names = [
            'attn_qkv', 'attn_qkv_out', 'attn_proj',
            'attn_kv_out', 'attn_v_out', 'intSoft'
        ]
        for name in layer_names:
            self.profilers[name] = profiler(name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with quantized layers."""
        # === Attention block ===
        B, N, C = x.shape

        # norm1 → qkv (FQ-ViT style: pass quantizers for scale propagation)
        # Note: norm1의 in_quantizer는 이전 블록의 출력이므로 None으로 처리
        # out_quantizer는 attn_qkv_output의 quantizer 사용
        qkv = self.attn_qkv.forward(x)
        qkv = self.attn_qkv_out.forward(qkv)

        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # (B, num_heads, N, head_dim)


        # Attention
        q *= self.scale
        attn = (q @ k.transpose(-2, -1)) 
        attn = self.attn_kv_out.forward(attn)

        # IntSoftmax에 scale 전달 (quantized mode에서는 kv_act.scaler 사용, fp32에서는 scale=None이어도 됨)
        scale_param = self.attn_kv_out.scaler if hasattr(self.attn_kv_out, 'scaler') else None
        attn = self.intSoft.forward(attn, scale=scale_param)

        x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        x = self.attn_v_out(x)
        x = self.attn_proj(x)

        return x



    def get_quantized_layers(self) -> Dict[str, nn.Module]:
        """Returns all quantized layers (including QAct)."""
        return {
            # Linear layers
            'attn_qkv': self.attn_qkv,
            'attn_proj': self.attn_proj,
            # QAct layers
            'attn_qkv_out': self.attn_qkv_out,
            'attn_kv_out': self.attn_kv_out,
            'attn_v_out': self.attn_v_out,
            # IntSoftmax
            'intSoft': self.intSoft,
        }

    def calibration(self, x: torch.Tensor) -> torch.Tensor:
        """Calibration pass to collect statistics."""
        B, N, C = x.shape

        qkv = self.attn_qkv.calibration(x)
        qkv = self.attn_qkv_out.calibration(qkv)

        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # (B, num_heads, N, head_dim)

        # Attention
        q *= self.scale
        attn = (q @ k.transpose(-2, -1)) 
        attn = self.attn_kv_out.calibration(attn)

        # IntSoftmax에 scale 전달 (quantized mode에서는 kv_act.scaler 사용, fp32에서는 scale=None이어도 됨)
        scale_param = self.attn_kv_out.scaler if hasattr(self.attn_kv_out, 'scaler') else None
        attn = self.intSoft.calibration(attn, scale=scale_param)

        x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        x = self.attn_v_out.calibration(x)
        x = self.attn_proj.calibration(x)

        return x

    def compute_quant_params(self, calib_loader=None):
        """Compute quantization parameters for all layers."""
        # IntSoftmax에 kv_act의 scale을 먼저 전달
        if hasattr(self.attn_kv_out, 'scaler') and self.attn_kv_out.scaler is not None:
            self.intSoft.input_scale = self.attn_kv_out.scaler

        # 모든 양자화 레이어 처리 (get_quantized_layers()에 다 포함됨)
        for name, layer in self.get_quantized_layers().items():
            if hasattr(layer, 'compute_quant_params'):
                # QLayerNorm은 calib_loader를 받음 (None이어도 scale은 계산됨)
                if 'norm' in name:
                    layer.compute_quant_params(calib_loader)
                else:
                    layer.compute_quant_params()

    def set_mode(self, mode: str):
        """Set quantization mode ('fp32' or 'quantized')."""
        # 모든 양자화 레이어의 mode 설정 (get_quantized_layers()에 다 포함됨)
        for layer in self.get_quantized_layers().values():
            if hasattr(layer, 'mode'):
                layer.mode = mode

    def get_profiler(self) -> Dict:
        """Get profiling results from all layers"""
        if not self.enable_profiling:
            return {}

        results = {}
        for layer in self.get_quantized_layers().values():
            results[layer.layer_name] = layer.get_profiler()
        return results

    def update_profiler_weights(self):
        """Update profiler with current quantized weights for analysis"""
        if not self.enable_profiling:
            return

        # Linear layers
        for layer_name in ['attn_qkv', 'attn_proj', 'mlp_fc1', 'mlp_fc2']:
            layer = getattr(self, layer_name)
            if hasattr(layer, 'weight') and hasattr(layer, 'quant_weight'):
                # quant_weight를 역양자화해서 FP32로 변환
                if layer.quant_weight is not None:
                    # scaler와 zero를 사용해서 역양자화
                    range_shape = layer.quantizer.get_reshape_range(layer.weight)
                    scaler_reshaped = layer.scaler.reshape(range_shape)
                    zero_reshaped = layer.zero.reshape(range_shape)

                    # 역양자화: dequant_weight = (quant_weight - zero) * scaler
                    dequant_weight = (layer.quant_weight - zero_reshaped) * scaler_reshaped

                    self.profilers[layer_name].update_weight(
                        layer.weight,
                        dequant_weight
                    )
                else:
                    self.profilers[layer_name].update_weight(
                        layer.weight,
                        layer.weight
                    )
