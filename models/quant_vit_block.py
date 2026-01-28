"""
QVitBlock: Quantized ViT Block
- QAttn + QMlp + QLayerNorm + Residual 연결 통합
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Union
from pathlib import Path

from .quant_attn import QAttn
from .quant_mlp import QMlp
from .ptq.quant_layernorm import QLayerNorm
from .ptq.quant_act import QAct
from .ptq.layer_profiler.profiler import profiler
from quant_config import QuantConfig
from utils.config_loader import load_config_from_yaml


class QVitBlock(nn.Module):
    """
    Quantized ViT Block for timm models.

    Structure:
        x → norm1 → QAttn → (+x) → residual1 → norm2 → QMlp → (+residual1) → residual2 → output

    Components:
        - norm1, norm2: QLayerNorm
        - attn: QAttn (QKV Linear + Q/K/V activation quant + IntSoftmax + Proj)
        - mlp: QMlp (FC1 + GELU + FC2)
        - residual1, residual2: Optional residual quantization (QAct)
    """

    def __init__(self,
                 block,  # timm.models.vision_transformer.Block
                 attn_config_path: Union[str, Path],
                 mlp_config_path: Union[str, Path],
                 block_config_path: Union[str, Path] = None):
        """
        Args:
            block: timm ViT block (e.g., model.blocks[0])
            attn_config_path: Path to attention config YAML
            mlp_config_path: Path to MLP config YAML
            block_config_path: Optional path to block-level config (norm, residual)
                              If None, uses default settings
        """
        super().__init__()


        # Load block-level config (for norm and residual)
        if block_config_path is not None:
            block_configs = load_config_from_yaml(block_config_path)
        else:
            block_configs = None

        # === LayerNorm ===
        # norm1: before attention
        if block_configs and 'norm1' in block_configs:
            norm1_config = block_configs['norm1']
        else:
            # Default norm config
            from quant_config import BitTypeConfig
            norm1_config = QuantConfig(
                calibration_mode='layer_wise',
                bit_type=BitTypeConfig(bits=8, symmetric=False, name='int8'),
                observer_type='MinmaxObserver',
                quantization_method='Uniform',
                enable_profiler=True
            )

        self.norm1 = QLayerNorm(
            input_module=block.norm1,
            quant_config=norm1_config,
            layer_name='norm1'
        )

        # norm2: before MLP
        if block_configs and 'norm2' in block_configs:
            norm2_config = block_configs['norm2']
        else:
            from quant_config import BitTypeConfig
            norm2_config = QuantConfig(
                calibration_mode='layer_wise',
                bit_type=BitTypeConfig(bits=8, symmetric=False, name='int8'),
                observer_type='MinmaxObserver',
                quantization_method='Uniform',
                enable_profiler=True
            )

        self.norm2 = QLayerNorm(
            input_module=block.norm2,
            quant_config=norm2_config,
            layer_name='norm2'
        )

        # === Attention ===
        self.attn = QAttn(block, attn_config_path)

        # === MLP ===
        self.mlp = QMlp(block, mlp_config_path)

        # === Residual Quantization (Optional) ===
        # residual1: after attention + skip connection
        if block_configs and 'residual1' in block_configs:
            self.residual1 = QAct(
                quant_config=block_configs['residual1'],
                act_module=None,
                layer_name='residual1'
            )
            self.residual1_quant_enable = True
        else:
            self.residual1 = None
            self.residual1_quant_enable = False

        # residual2: after MLP + skip connection
        if block_configs and 'residual2' in block_configs:
            self.residual2 = QAct(
                quant_config=block_configs['residual2'],
                act_module=None,
                layer_name='residual2'
            )
            self.residual2_quant_enable = True
        else:
            self.residual2 = None
            self.residual2_quant_enable = False


        # Enable profiling flag
        self.enable_profiling = self.attn.enable_profiling or self.mlp.enable_profiling

        # === Profiler 초기화 ===
        self.profilers = {}
        if self.enable_profiling:
            self._init_profilers()

    def _init_profilers(self):
        """Initialize profilers for block-level layers"""
        layer_names = ['norm1', 'norm2']
        if self.residual1_quant_enable:
            layer_names.append('residual1')
        if self.residual2_quant_enable:
            layer_names.append('residual2')

        for name in layer_names:
            self.profilers[name] = profiler(name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with quantized layers.

        Flow:
            x → norm1 → attn → (+x) → [residual1] → norm2 → mlp → (+) → [residual2] → output
        """
        # === Attention Block ===
        # norm1
        x_norm1 = self.norm1.forward(x)

        # Attention
        x_attn = self.attn.forward(x_norm1)

        # Residual connection 1 (skip connection)
        x = x + x_attn

        # Optional residual1 quantization
        if self.residual1_quant_enable and self.residual1 is not None:
            x = self.residual1.forward(x)

        # === MLP Block ===
        # norm2
        x_norm2 = self.norm2.forward(x)

        # MLP
        x_mlp = self.mlp.forward(x_norm2)

        # Residual connection 2 (skip connection)
        x = x + x_mlp

        # Optional residual2 quantization
        if self.residual2_quant_enable and self.residual2 is not None:
            x = self.residual2.forward(x)

        return x

    def calibration(self, x: torch.Tensor) -> torch.Tensor:
        """Calibration pass to collect statistics."""
        # === Attention Block ===
        # norm1 calibration
        x_norm1 = self.norm1.calibration(x)

        # Attention calibration
        x_attn = self.attn.calibration(x_norm1)

        # Residual connection 1
        x = x + x_attn

        # Optional residual1 calibration
        if self.residual1_quant_enable and self.residual1 is not None:
            x = self.residual1.calibration(x)

        # === MLP Block ===
        # norm2 calibration
        x_norm2 = self.norm2.calibration(x)

        # MLP calibration
        x_mlp = self.mlp.calibration(x_norm2)

        # Residual connection 2
        x = x + x_mlp

        # Optional residual2 calibration
        if self.residual2_quant_enable and self.residual2 is not None:
            x = self.residual2.calibration(x)

        return x

    def compute_quant_params(self, calib_loader=None):
        """Compute quantization parameters for all layers."""
        # LayerNorm (PTF 적용 시 calib_loader 필요)
        self.norm1.compute_quant_params(calib_loader)
        self.norm2.compute_quant_params(calib_loader)

        # Attention
        self.attn.compute_quant_params()

        # MLP
        self.mlp.compute_quant_params()

        # Residual (optional)
        if self.residual1_quant_enable and self.residual1 is not None:
            self.residual1.compute_quant_params()
        if self.residual2_quant_enable and self.residual2 is not None:
            self.residual2.compute_quant_params()

    def set_mode(self, mode: str):
        """Set quantization mode ('fp32' or 'quantized')."""
        # LayerNorm
        self.norm1.mode = mode
        self.norm2.mode = mode

        # Attention
        self.attn.set_mode(mode)

        # MLP
        self.mlp.set_mode(mode)

        # Residual (optional)
        if self.residual1_quant_enable and self.residual1 is not None:
            self.residual1.mode = mode
        if self.residual2_quant_enable and self.residual2 is not None:
            self.residual2.mode = mode

    def set_profiling(self, enable: bool):
        """Enable/disable profiling for all layers."""
        self.enable_profiling = enable

        # LayerNorm
        self.norm1.enable_profiling = enable
        self.norm2.enable_profiling = enable

        # Attention
        self.attn.set_profiling(enable)

        # MLP
        self.mlp.set_profiling(enable)

        # Residual (optional)
        if self.residual1 is not None:
            self.residual1.enable_profiling = enable
        if self.residual2 is not None:
            self.residual2.enable_profiling = enable

    def get_quantized_layers(self) -> Dict[str, nn.Module]:
        """Returns all quantized layers."""
        layers = {
            'norm1': self.norm1,
            'norm2': self.norm2,
        }

        # Add attention layers
        for name, layer in self.attn.get_quantized_layers().items():
            layers[f'attn_{name}'] = layer

        # Add MLP layers
        for name, layer in self.mlp.get_quantized_layers().items():
            layers[f'mlp_{name}'] = layer

        # Add residual layers (optional)
        if self.residual1_quant_enable and self.residual1 is not None:
            layers['residual1'] = self.residual1
        if self.residual2_quant_enable and self.residual2 is not None:
            layers['residual2'] = self.residual2

        return layers

    def get_profiler(self) -> Dict:
        """Get profiling results from all layers (dict format)"""
        results = {}

        # Block-level profilers
        results['norm1'] = self.norm1.get_profiler()
        results['norm2'] = self.norm2.get_profiler()

        if self.residual1_quant_enable and self.residual1 is not None:
            results['residual1'] = self.residual1.get_profiler()
        if self.residual2_quant_enable and self.residual2 is not None:
            results['residual2'] = self.residual2.get_profiler()

        # Attention profilers
        attn_profilers = self.attn.get_profiler()
        for name, prof in attn_profilers.items():
            results[f'attn_{name}'] = prof

        # MLP profilers
        mlp_profilers = self.mlp.get_profiler()
        for name, prof in mlp_profilers.items():
            results[f'mlp_{name}'] = prof

        return results

    def get_profiler_list(self) -> list:
        """
        Get profiler objects as a list with layer names.

        Returns:
            List of tuples: [(layer_name, profiler_object), ...]
            - QuantLinear layers: (name_weight, weight_profiler), (name_output, output_profiler)
            - QAct/QLayerNorm layers: (name, profiler)
            순서: norm1 → attn layers → residual1 → norm2 → mlp layers → residual2
        """
        from .ptq.layer_profiler.profiler import profiler as ProfilerClass

        def _extract_profilers(name_prefix: str, prof_data):
            """Extract profiler objects from various formats"""
            result = []
            if prof_data is None:
                return result

            # Case 1: profiler 객체 자체
            if isinstance(prof_data, ProfilerClass):
                result.append((name_prefix, prof_data))
            # Case 2: {output: ..., weight: ...} dict (QuantLinear)
            elif isinstance(prof_data, dict):
                if 'weight' in prof_data and prof_data['weight'] is not None:
                    result.append((f'{name_prefix}_weight', prof_data['weight']))
                if 'output' in prof_data and prof_data['output'] is not None:
                    result.append((f'{name_prefix}_output', prof_data['output']))
            return result

        profiler_list = []

        # 1. norm1
        profiler_list.extend(_extract_profilers('norm1', self.norm1.get_profiler()))

        # 2. Attention profilers (순서대로)
        attn_profilers = self.attn.get_profiler()
        for name, prof in attn_profilers.items():
            profiler_list.extend(_extract_profilers(f'attn_{name}', prof))

        # 3. residual1 (optional)
        if self.residual1_quant_enable and self.residual1 is not None:
            profiler_list.extend(_extract_profilers('residual1', self.residual1.get_profiler()))

        # 4. norm2
        profiler_list.extend(_extract_profilers('norm2', self.norm2.get_profiler()))

        # 5. MLP profilers (순서대로)
        mlp_profilers = self.mlp.get_profiler()
        for name, prof in mlp_profilers.items():
            profiler_list.extend(_extract_profilers(f'mlp_{name}', prof))

        # 6. residual2 (optional)
        if self.residual2_quant_enable and self.residual2 is not None:
            profiler_list.extend(_extract_profilers('residual2', self.residual2.get_profiler()))

        return profiler_list

    def get_profiler_names(self) -> list:
        """Get list of layer names that have profilers enabled."""
        return [name for name, _ in self.get_profiler_list()]

    def get_profiler_by_name(self, layer_name: str):
        """Get specific profiler by layer name."""
        profiler_dict = self.get_profiler()
        return profiler_dict.get(layer_name, None)

    def print_layer_summary(self):
        """Print summary of all quantized layers"""
        print("\n" + "=" * 70)
        print("QVitBlock Layer Summary")
        print("=" * 70)

        layers = self.get_quantized_layers()
        print(f"\nTotal quantized layers: {len(layers)}")

        print("\n[Block-level layers]")
        print(f"  norm1: {type(self.norm1).__name__}")
        print(f"  norm2: {type(self.norm2).__name__}")
        if self.residual1_quant_enable:
            print(f"  residual1: {type(self.residual1).__name__}")
        if self.residual2_quant_enable:
            print(f"  residual2: {type(self.residual2).__name__}")

        print("\n[Attention layers]")
        for name, layer in self.attn.get_quantized_layers().items():
            print(f"  {name}: {type(layer).__name__}")

        print("\n[MLP layers]")
        for name, layer in self.mlp.get_quantized_layers().items():
            print(f"  {name}: {type(layer).__name__}")

        print("=" * 70)
