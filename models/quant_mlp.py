import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Union
from pathlib import Path

from .ptq.quant_linear import QuantLinear
from .ptq.quant_act import QAct
from .ptq.layer_profiler.profiler import profiler
from quant_config import QuantConfig
from utils.config_loader import load_config_from_yaml


class QMlp(nn.Module):
    """
    Quantized MLP layer for ViT Block.

    Structure:
        Input → fc1 (Linear) → GELU → fc2 (Linear) → Output

    Quantization:
        - fc1: weight quantization only (output_quant_enable=False)
        - gelu: activation quantization (별도 QAct)
        - fc2: weight + output quantization (output_quant_enable=True)
    """

    def __init__(self,
                 block,  # timm.models.vision_transformer.Block
                 config_path: Union[str, Path]):
        super().__init__()

        # original_block 참조 저장하지 않음 - 메모리 절약

        # Config 로드
        configs = load_config_from_yaml(config_path)

        # Enable profiling 설정
        if 'mlp_fc1' in configs and isinstance(configs['mlp_fc1'], dict):
            self.enable_profiling = configs['mlp_fc1']['weight'].enable_profiler
        else:
            self.enable_profiling = configs.get('output', configs.get('weight')).enable_profiler

        # Helper function to get config for a layer
        def get_layer_config(layer_name):
            if layer_name in configs:
                return configs[layer_name]
            # Fallback to global configs
            if layer_name in ['mlp_fc1', 'mlp_fc2']:
                return {'weight': configs.get('weight'), 'output': configs.get('output')}
            else:
                return configs.get('activation', configs.get('output'))

        # === MLP Layers ===
        # FC1: 768 → 3072 (hidden expansion)
        fc1_config = get_layer_config('mlp_fc1')
        self.fc1 = QuantLinear(
            input_module=block.mlp.fc1,
            out_config=fc1_config['output'] if isinstance(fc1_config, dict) else configs['output'],
            weight_config=fc1_config['weight'] if isinstance(fc1_config, dict) else configs['weight'],
            layer_name='mlp_fc1'
        )

        # FC2: 3072 → 768 (hidden reduction)
        fc2_config = get_layer_config('mlp_fc2')
        self.fc2 = QuantLinear(
            input_module=block.mlp.fc2,
            out_config=fc2_config['output'] if isinstance(fc2_config, dict) else configs['output'],
            weight_config=fc2_config['weight'] if isinstance(fc2_config, dict) else configs['weight'],
            layer_name='mlp_fc2'
        )

        # GELU activation quantization (fc1 출력 양자화)
        self.gelu = QAct(
            quant_config=get_layer_config('mlp_gelu'),
            act_module=block.mlp.act,  # timm의 GELU 모듈
            layer_name='mlp_gelu'
        )

        # === Profiler 초기화 ===
        self.profilers = {}
        if self.enable_profiling:
            self._init_profilers()

    def _init_profilers(self):
        """Initialize profilers for each quantized layer"""
        layer_names = ['mlp_fc1', 'mlp_gelu', 'mlp_fc2']
        for name in layer_names:
            self.profilers[name] = profiler(name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with quantized layers.

        Flow:
            x → fc1 (weight quant only) → gelu (act quant) → fc2 (weight + output quant) → output
        """
        # FC1: weight quantization only
        x = self.fc1.forward(x)

        # GELU with activation quantization
        x = self.gelu.forward(x)

        # FC2: weight + output quantization
        x = self.fc2.forward(x)

        return x

    def get_quantized_layers(self) -> Dict[str, nn.Module]:
        """Returns all quantized layers."""
        return {
            'mlp_fc1': self.fc1,
            'mlp_gelu': self.gelu,
            'mlp_fc2': self.fc2,
        }

    def calibration(self, x: torch.Tensor) -> torch.Tensor:
        """Calibration pass to collect statistics."""
        # FC1 calibration
        x = self.fc1.calibration(x)

        # GELU calibration
        x = self.gelu.calibration(x)

        # FC2 calibration
        x = self.fc2.calibration(x)

        return x

    def compute_quant_params(self):
        """Compute quantization parameters for all layers."""
        # QuantLinear layers
        self.fc1.compute_output_quant_params()
        self.fc2.compute_output_quant_params()

        # GELU QAct layer
        self.gelu.compute_quant_params()

    def set_mode(self, mode: str):
        """Set quantization mode ('fp32' or 'quantized')."""
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

        for layer_name in ['mlp_fc1', 'mlp_fc2']:
            layer = self.fc1 if layer_name == 'mlp_fc1' else self.fc2
            if hasattr(layer, 'weight') and hasattr(layer, 'quant_weight'):
                if layer.quant_weight is not None:
                    range_shape = layer.quantizer.get_reshape_range(layer.weight)
                    scaler_reshaped = layer.scaler.reshape(range_shape)
                    zero_reshaped = layer.zero.reshape(range_shape)

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
