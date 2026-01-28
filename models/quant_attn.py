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
from quant_config import QuantConfig
from utils.config_loader import load_config_from_yaml



class QAttn(nn.Module):
    """
    Quantized ViT Block for timm models.
    timm의 vit_base_patch16_224 등에서 사용.
    """

    def __init__(self,
                 block,  # timm.models.vision_transformer.Block
                 config_path: Union[str, Path]):
        super().__init__()

        # original_block 참조 저장하지 않음 - 메모리 절약

        # Config 로드: YAML 파일에서 각 레이어별 config를 로드
        # 반환값: dict with per-layer configs
        configs = load_config_from_yaml(config_path)

        # Enable profiling은 config에서 가져옴
        # Per-layer format: attn_qkv는 dict with 'weight', 'output' keys
        # attn_q_out 등은 직접 QuantConfig
        if 'attn_qkv' in configs and isinstance(configs['attn_qkv'], dict):
            # Per-layer format
            self.enable_profiling = configs['attn_qkv']['weight'].enable_profiler
        else:
            # Global format (legacy)
            self.enable_profiling = configs.get('output', configs.get('weight')).enable_profiler

        # === Attention 부분 ===
        # timm은 qkv가 하나로 합쳐져 있음 (in_features → 3 * out_features)

        # Helper function to get config for a layer
        def get_layer_config(layer_name):
            if layer_name in configs:
                return configs[layer_name]
            # Fallback to global configs
            if layer_name in ['attn_qkv', 'attn_proj']:
                return {'weight': configs.get('weight'), 'output': configs.get('output')}
            else:
                return configs.get('activation', configs.get('output'))

        # QuantLinear layers
        qkv_config = get_layer_config('attn_qkv')
        self.attn_qkv = QuantLinear(
            input_module=block.attn.qkv,
            out_config=qkv_config['output'] if isinstance(qkv_config, dict) else configs['output'],
            weight_config=qkv_config['weight'] if isinstance(qkv_config, dict) else configs['weight'],
            layer_name='attn_qkv'
        )

        proj_config = get_layer_config('attn_proj')
        self.attn_proj = QuantLinear(
            input_module=block.attn.proj,
            out_config=proj_config['output'] if isinstance(proj_config, dict) else configs['output'],
            weight_config=proj_config['weight'] if isinstance(proj_config, dict) else configs['weight'],
            layer_name='attn_proj'
        )

        # Attention 파라미터
        num_heads = block.attn.num_heads
        head_dim = block.attn.head_dim

        # QKV 분리 후 각각 양자화하기 위한 QAct 레이어
        self.attn_q_out = QAct(
            quant_config=get_layer_config('attn_q_out'),
            act_module=None,
            layer_name='attn_q_out'
        )
        self.attn_k_out = QAct(
            quant_config=get_layer_config('attn_k_out'),
            act_module=None,
            layer_name='attn_k_out'
        )
        self.attn_v_input = QAct(
            quant_config=get_layer_config('attn_v_input'),
            act_module=None,
            layer_name='attn_v_input'
        )

        # kv_act: Q@K 출력 양자화 (attention scores before softmax)
        self.attn_kv_out = QAct(
            quant_config=get_layer_config('attn_kv_out'),
            act_module=None,
            layer_name='attn_kv_out'
        )
        # attn_v_output: Attention @ Value 출력 양자화 (FQ-ViT qact2)
        self.attn_v_out = QAct(
            quant_config=get_layer_config('attn_v_out'),
            act_module=None,
            layer_name='attn_v_out'
        )

        # QintSoftmax
        self.intSoft = QuantIntSoft(
            input_module=None,
            quant_config=get_layer_config('intSoft'),
            layer_name='intSoft'
        )

        # Attention 파라미터 (self에 저장)
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.attn_dim = head_dim * num_heads
        self.scale = block.attn.scale

        # === Profiler 초기화 ===
        self.profilers = {}
        if self.enable_profiling:
            self._init_profilers()

    def _init_profilers(self):
        """Initialize profilers for each quantized layer"""
        layer_names = [
            'attn_qkv', 'attn_proj',
            'attn_q_out', 'attn_k_out', 'attn_v_input',
            'attn_kv_out', 'attn_v_out', 'intSoft'
        ]
        for name in layer_names:
            self.profilers[name] = profiler(name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with quantized layers."""
        # === Attention block ===
        B, N, C = x.shape

        # QKV projection (output_quant_enable=False이므로 FP32 출력)
        # qkv shape: (B, N, 3*C) = (B, N, 2304)
        qkv = self.attn_qkv.forward(x)

        # QKV 분리: Linear 직후 바로 슬라이싱
        # qkv: (B, N, 2304) → q, k, v 각각 (B, N, 768)
        q = qkv[:, :, :C]              # (B, N, C)
        k = qkv[:, :, C:2*C]           # (B, N, C)
        v = qkv[:, :, 2*C:]            # (B, N, C)

        # Q, K, V 각각 양자화 (아직 head 분리 전)
        q = self.attn_q_out.forward(q)  # (B, N, C)
        k = self.attn_k_out.forward(k)  # (B, N, C)
        v = self.attn_v_input.forward(v)  # (B, N, C)

        # 양자화 후 head 분리
        # (B, N, C) → (B, num_heads, N, head_dim)
        q = q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # 1. 여기서 반드시 self.scale을 곱해야 함 (Calibration과 일치)
        attn = (q @ k.transpose(-2, -1)) * self.scale 
        
        # 2. Scale이 적용된 '적절한 범위'의 데이터를 양자화 레이어에 전달
        attn = self.attn_kv_out.forward(attn)

        # IntSoftmax에 scale 전달 (quantized mode에서는 kv_act.scaler 사용, fp32에서는 scale=None이어도 됨)
        scale_param = self.attn_kv_out.scaler if hasattr(self.attn_kv_out, 'scaler') else None
        attn = self.intSoft.forward(attn, scale=scale_param)

        x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        x = self.attn_v_out.forward(x)
        x = self.attn_proj.forward(x)

        return x



    def get_quantized_layers(self) -> Dict[str, nn.Module]:
        """Returns all quantized layers (including QAct)."""
        return {
            # Linear layers
            'attn_qkv': self.attn_qkv,
            'attn_proj': self.attn_proj,
            # QKV output QAct layers
            'attn_q_out': self.attn_q_out,
            'attn_k_out': self.attn_k_out,
            'attn_v_input': self.attn_v_input,
            # Attention QAct layers
            'attn_kv_out': self.attn_kv_out,
            'attn_v_out': self.attn_v_out,
            # IntSoftmax
            'intSoft': self.intSoft,
        }

    def calibration(self, x: torch.Tensor) -> torch.Tensor:
        """Calibration pass to collect statistics."""
        B, N, C = x.shape

        # QKV projection calibration (output_quant_enable=False이므로 FP32 출력)
        # qkv shape: (B, N, 3*C) = (B, N, 2304)
        qkv = self.attn_qkv.calibration(x)

        # QKV 분리: Linear 직후 바로 슬라이싱
        # qkv: (B, N, 2304) → q, k, v 각각 (B, N, 768)
        q = qkv[:, :, :C]              # (B, N, C)
        k = qkv[:, :, C:2*C]           # (B, N, C)
        v = qkv[:, :, 2*C:]            # (B, N, C)

        # Q, K, V 각각 calibration (아직 head 분리 전)
        q = self.attn_q_out.calibration(q)  # (B, N, C)
        k = self.attn_k_out.calibration(k)  # (B, N, C)
        v = self.attn_v_input.calibration(v)  # (B, N, C)

        # calibration 후 head 분리
        # (B, N, C) → (B, num_heads, N, head_dim)
        q = q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_kv_out.calibration(attn)

        # IntSoftmax에 scale 전달 (quantized mode에서는 kv_act.scaler 사용, fp32에서는 scale=None이어도 됨)
        scale_param = self.attn_kv_out.scaler if hasattr(self.attn_kv_out, 'scaler') else None
        attn = self.intSoft.calibration(attn, input_scale=scale_param)

        x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        x = self.attn_v_out.calibration(x)
        x = self.attn_proj.calibration(x)

        return x

    def compute_quant_params(self, calib_loader=None):
        """Compute quantization parameters for all layers."""
        # QuantLinear layers: compute_output_quant_params() 사용
        # (weight는 이미 초기화 시 quantize됨)
        # output_quant_enable=False이므로 output params는 None 반환
        self.attn_qkv.compute_output_quant_params()
        self.attn_proj.compute_output_quant_params()

        # QKV output QAct layers: compute_quant_params() 사용
        self.attn_q_out.compute_quant_params()
        self.attn_k_out.compute_quant_params()
        self.attn_v_input.compute_quant_params()

        # Attention QAct layers: compute_quant_params() 사용
        self.attn_kv_out.compute_quant_params()
        self.attn_v_out.compute_quant_params()

        # IntSoftmax에 kv_act의 scale을 전달하고 compute
        if hasattr(self.attn_kv_out, 'scaler') and self.attn_kv_out.scaler is not None:
            self.intSoft.input_scale = self.attn_kv_out.scaler
        self.intSoft.compute_quant_params()

    def set_mode(self, mode: str):
        """Set quantization mode ('fp32' or 'quantized')."""
        # 모든 양자화 레이어의 mode 설정 (get_quantized_layers()에 다 포함됨)
        for layer in self.get_quantized_layers().values():
            if hasattr(layer, 'mode'):
                layer.mode = mode

    def set_profiling(self, enable: bool):
        """Enable/disable profiling for all layers."""
        self.enable_profiling = enable
        for layer in self.get_quantized_layers().values():
            if hasattr(layer, 'enable_profiling'):
                layer.enable_profiling = enable

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

