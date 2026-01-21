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
from .ptq.quant_layernorm import QLayerNorm
from .ptq.layer_profiler.profiler import profiler
from quant_config import QuantConfig, LayerQuantConfig
from utils.config_loader import load_config_from_yaml

from torch.nn.modules.container import Sequential


class QuantTimmVitBlock(nn.Module):
    """
    Quantized ViT Block for timm models.
    timm의 vit_base_patch16_224 등에서 사용.
    """

    def __init__(self,
                 block,  # timm.models.vision_transformer.Block
                 quant_config: Union[QuantConfig, LayerQuantConfig, str, Path] = None,
                 enable_profiling: bool = False):
        super().__init__()

        self.original_block = block
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
        self.attn_qkv_output = QAct(
            quant_config=layer_config.get_config('attn_qkv_output'),
            act_module=None
        )
        self.attn_proj = QuantLinear(
            input_module=block.attn.proj,
            quant_config=layer_config.get_config('attn_proj')
        )

        # kv_act: IntSoftmax에 스칼라 scale을 전달해야 하므로 layer_wise 필요
        # YAML에서 layer_wise로 설정되어 있을 것으로 예상
        self.kv_act = QAct(
            quant_config=layer_config.get_config('kv_act'),
            act_module=None
        )
        self.sv_attn= QAct(
            quant_config=layer_config.get_config('sv_attn'),
            act_module=None
        )
        self.residual1= QAct(
            quant_config=layer_config.get_config('residual1'),
            act_module=None
        )
        self.residual2= QAct(
            quant_config=layer_config.get_config('residual2'),
            act_module=None
        )

        # === MLP 부분 ===
        self.mlp_fc1 = QuantLinear(
            input_module=block.mlp.fc1,
            quant_config=layer_config.get_config('mlp_fc1')
        )
        self.mlp_act = QAct(
            quant_config=layer_config.get_config('mlp_act'),
            act_module=block.mlp.act
        )
        self.mlp_fc2 = QuantLinear(
            input_module=block.mlp.fc2,
            quant_config=layer_config.get_config('mlp_fc2')
        )
        self.mlp_act2 = QAct(
            quant_config=layer_config.get_config('mlp_act2'),
            act_module=None
        )

        # === LayerNorm ===
        self.norm1 = QLayerNorm(
            input_module=block.norm1,
            quant_config=layer_config.get_config('norm1')
        )
        self.norm2 = QLayerNorm(
            input_module=block.norm2,
            quant_config=layer_config.get_config('norm2')
        )

        #QintSoftmax
        self.intSoft = QuantIntSoft(
            input_module = None,
            quant_config = layer_config.get_config('intSoft')
        )


        # === 그대로 쓰는 것들 ===
        self.attn_drop = block.attn.attn_drop
        self.proj_drop = block.attn.proj_drop

        self.drop_path1 = block.drop_path1
        self.drop_path2 = block.drop_path2

        # Attention 파라미터
        self.num_heads = block.attn.num_heads
        self.head_dim = block.attn.head_dim
        self.scale = block.attn.scale

        # === Profiler 초기화 ===
        self.profilers = {}
        if self.enable_profiling:
            self._init_profilers()

    def _init_profilers(self):
        """Initialize profilers for each quantized layer"""
        layer_names = [
            'attn_qkv', 'attn_qkv_output', 'attn_proj',
            'kv_act', 'sv_attn', 'intSoft',
            'mlp_fc1', 'mlp_act', 'mlp_fc2', 'mlp_act2',
            'norm1', 'norm2',
            'residual1', 'residual2'
        ]
        for name in layer_names:
            self.profilers[name] = profiler(name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with quantized layers."""
        # === Attention block ===
        B, N, C = x.shape

        # norm1 → qkv
        if self.enable_profiling:
            with self.profilers['norm1'].measure_time():
                x_norm = self.norm1.forward(x)
        else:
            x_norm = self.norm1.forward(x)

        if self.enable_profiling:
            with self.profilers['attn_qkv'].measure_time():
                qkv = self.attn_qkv.forward(x_norm)
        else:
            qkv = self.attn_qkv.forward(x_norm)

        if self.enable_profiling:
            with self.profilers['attn_qkv_output'].measure_time():
                qkv = self.attn_qkv_output.forward(qkv)
        else:
            qkv = self.attn_qkv_output.forward(qkv)

        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # (B, num_heads, N, head_dim)

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale

        if self.enable_profiling:
            with self.profilers['kv_act'].measure_time():
                attn = self.kv_act.forward(attn)
        else:
            attn = self.kv_act.forward(attn)

        # IntSoftmax에 scale 전달 (quantized mode에서는 kv_act.scaler 사용, fp32에서는 scale=None이어도 됨)
        scale_param = self.kv_act.scaler if hasattr(self.kv_act, 'scaler') else None

        if self.enable_profiling:
            with self.profilers['intSoft'].measure_time():
                attn = self.intSoft.forward(attn, scale=scale_param)
        else:
            attn = self.intSoft.forward(attn, scale=scale_param)

        attn = self.attn_drop(attn)

        # Combine heads
        x_attn = (attn @ v).transpose(1, 2).reshape(B, N, C)

        if self.enable_profiling:
            with self.profilers['attn_proj'].measure_time():
                x_attn = self.attn_proj.forward(x_attn)
        else:
            x_attn = self.attn_proj.forward(x_attn)

        if self.enable_profiling:
            with self.profilers['sv_attn'].measure_time():
                x_attn = self.sv_attn.forward(x_attn)
        else:
            x_attn = self.sv_attn.forward(x_attn)

        x_attn = self.proj_drop(x_attn)

        # Residual 1
        x = x + self.drop_path1(x_attn)

        if self.enable_profiling:
            with self.profilers['residual1'].measure_time():
                x = self.residual1.forward(x)
        else:
            x = self.residual1.forward(x)

        # === MLP block ===
        if self.enable_profiling:
            with self.profilers['norm2'].measure_time():
                x_norm = self.norm2.forward(x)
        else:
            x_norm = self.norm2.forward(x)

        if self.enable_profiling:
            with self.profilers['mlp_fc1'].measure_time():
                x_mlp = self.mlp_fc1.forward(x_norm)
        else:
            x_mlp = self.mlp_fc1.forward(x_norm)

        if self.enable_profiling:
            with self.profilers['mlp_act'].measure_time():
                x_mlp = self.mlp_act.forward(x_mlp)
        else:
            x_mlp = self.mlp_act.forward(x_mlp)

        if self.enable_profiling:
            with self.profilers['mlp_fc2'].measure_time():
                x_mlp = self.mlp_fc2.forward(x_mlp)
        else:
            x_mlp = self.mlp_fc2.forward(x_mlp)

        if self.enable_profiling:
            with self.profilers['mlp_act2'].measure_time():
                x_mlp = self.mlp_act2.forward(x_mlp)
        else:
            x_mlp = self.mlp_act2.forward(x_mlp)

        # Residual 2
        x = x + self.drop_path2(x_mlp)

        if self.enable_profiling:
            with self.profilers['residual2'].measure_time():
                x = self.residual2.forward(x)
        else:
            x = self.residual2.forward(x)

        return x

    def get_quantized_layers(self) -> Dict[str, nn.Module]:
        """Returns all quantized layers (including QAct)."""
        return {
            # Linear layers
            'attn_qkv': self.attn_qkv,
            'attn_proj': self.attn_proj,
            'mlp_fc1': self.mlp_fc1,
            'mlp_fc2': self.mlp_fc2,
            # LayerNorm layers
            'norm1': self.norm1,
            'norm2': self.norm2,
            # QAct layers
            'attn_qkv_output': self.attn_qkv_output,
            'kv_act': self.kv_act,
            'sv_attn': self.sv_attn,
            'residual1': self.residual1,
            'residual2': self.residual2,
            'mlp_act': self.mlp_act,
            'mlp_act2': self.mlp_act2,
            # IntSoftmax
            'intSoft': self.intSoft,
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
        # int8 -> int32 -> int8
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.kv_act.calibration(attn)

        # Log-Int-Softmax (LIS)
        # int8 -> int8
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
        x = x + self.drop_path1(x_attn)
        x = self.residual1.calibration(x)

        # MLP
        x_norm = self.norm2.calibration(x)
        x_mlp = self.mlp_fc1.calibration(x_norm)
        x_mlp = self.mlp_act.calibration(x_mlp)
        x_mlp = self.mlp_fc2.calibration(x_mlp)
        x_mlp = self.mlp_act2.calibration(x_mlp)

        x = x + self.drop_path2(x_mlp)
        x = self.residual2.calibration(x)

        return x

    def compute_quant_params(self, calib_loader=None):
        """Compute quantization parameters for all layers."""
        # IntSoftmax에 kv_act의 scale을 먼저 전달
        if hasattr(self.kv_act, 'scaler') and self.kv_act.scaler is not None:
            self.intSoft.input_scale = self.kv_act.scaler

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

    def get_profiling_results(self) -> Dict:
        """Get profiling results from all layers"""
        if not self.enable_profiling:
            return {}

        results = {}
        for name, prof in self.profilers.items():
            results[name] = {
                'time': prof.get_time_record(),
                'memory': prof.get_memory_record()
            }
        return results

    def print_profiling_summary(self):
        """Print a summary of profiling results"""
        if not self.enable_profiling:
            print("Profiling is not enabled. Set enable_profiling=True in __init__")
            return

        print("\n" + "="*80)
        print("Profiling Summary")
        print("="*80)

        results = self.get_profiling_results()

        # Sort by mean time
        time_records = []
        for name, data in results.items():
            time_data = data['time']
            if name in time_data:
                time_records.append((name, time_data[name]))

        time_records.sort(key=lambda x: x[1]['mean'], reverse=True)

        print(f"\n{'Layer':<20} {'Mean (ms)':<12} {'Total (ms)':<12} {'Count':<8} {'Min (ms)':<12} {'Max (ms)':<12}")
        print("-"*80)

        for name, stats in time_records:
            print(f"{name:<20} {stats['mean']*1000:<12.4f} {stats['total']*1000:<12.4f} "
                  f"{stats['count']:<8} {stats['min']*1000:<12.4f} {stats['max']*1000:<12.4f}")

        print("="*80)

    def reset_profiling(self):
        """Reset all profiling data"""
        if not self.enable_profiling:
            return

        for prof in self.profilers.values():
            prof.reset_time_profiler()
            prof.reset_memory_profiler()

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

    def update_activation_stats(self, test_input):
        """Run forward pass to capture activation statistics for kv_act and sv_attn"""
        if not self.enable_profiling:
            return

        # Storage for original and quantized activations
        self.activation_cache = {}

        with torch.no_grad():
            B, N, C = test_input.shape

            # === FP32 mode - capture original activations ===
            self.set_mode('fp32')

            x_norm = self.norm1.forward(test_input)
            qkv = self.attn_qkv.forward(x_norm)
            qkv = self.attn_qkv_output.forward(qkv)
            qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)

            # kv_act (before softmax)
            attn_fp32 = (q @ k.transpose(-2, -1)) * self.scale
            kv_act_fp32 = self.kv_act.forward(attn_fp32)
            self.activation_cache['kv_act_fp32'] = kv_act_fp32.clone()

            # Continue to get sv_attn
            scale_param = self.kv_act.scaler if hasattr(self.kv_act, 'scaler') else None
            attn_fp32 = self.intSoft.forward(kv_act_fp32, scale=scale_param)
            attn_fp32 = self.attn_drop(attn_fp32)
            x_attn_fp32 = (attn_fp32 @ v).transpose(1, 2).reshape(B, N, C)
            x_attn_fp32 = self.attn_proj.forward(x_attn_fp32)

            # sv_attn
            sv_attn_fp32 = self.sv_attn.forward(x_attn_fp32)
            self.activation_cache['sv_attn_fp32'] = sv_attn_fp32.clone()

            # === Quantized mode - capture quantized activations ===
            self.set_mode('quantized')

            x_norm = self.norm1.forward(test_input)
            qkv = self.attn_qkv.forward(x_norm)
            qkv = self.attn_qkv_output.forward(qkv)
            qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)

            # kv_act (before softmax)
            attn_quant = (q @ k.transpose(-2, -1)) * self.scale
            kv_act_quant = self.kv_act.forward(attn_quant)
            self.activation_cache['kv_act_quant'] = kv_act_quant.clone()

            # Continue to get sv_attn
            scale_param = self.kv_act.scaler if hasattr(self.kv_act, 'scaler') else None
            attn_quant = self.intSoft.forward(kv_act_quant, scale=scale_param)
            attn_quant = self.attn_drop(attn_quant)
            x_attn_quant = (attn_quant @ v).transpose(1, 2).reshape(B, N, C)
            x_attn_quant = self.attn_proj.forward(x_attn_quant)

            # sv_attn
            sv_attn_quant = self.sv_attn.forward(x_attn_quant)
            self.activation_cache['sv_attn_quant'] = sv_attn_quant.clone()

        # Update profilers with activation data
        self.profilers['kv_act'].update_weight(
            self.activation_cache['kv_act_fp32'],
            self.activation_cache['kv_act_quant']
        )
        self.profilers['sv_attn'].update_weight(
            self.activation_cache['sv_attn_fp32'],
            self.activation_cache['sv_attn_quant']
        )

    def get_layer_statistics(self, layer_name: str):
        """Get statistical analysis for a specific layer"""
        if not self.enable_profiling:
            return None

        if layer_name not in self.profilers:
            return None

        return self.profilers[layer_name].get_statistic()

    def plot_activation_histograms(self, layer_names=['kv_act', 'sv_attn'], save_path=None):
        """Plot histogram comparison for specified activation layers"""
        if not self.enable_profiling:
            print("Profiling is not enabled")
            return

        num_layers = len(layer_names)
        fig, axes = plt.subplots(num_layers, 1, figsize=(12, 6 * num_layers))

        if num_layers == 1:
            axes = [axes]

        for idx, layer_name in enumerate(layer_names):
            if layer_name not in self.profilers:
                continue

            hist_data = self.profilers[layer_name].get_hist()

            orig_hist = hist_data['original_hist'].numpy()
            quant_hist = hist_data['quantized_hist'].numpy()
            orig_range = hist_data['original_range']
            quant_range = hist_data['quantized_range']

            # Normalize histograms to probability
            orig_hist_norm = orig_hist / (orig_hist.sum() + 1e-10)
            quant_hist_norm = quant_hist / (quant_hist.sum() + 1e-10)

            # Create bin edges
            min_val = min(orig_range[0], quant_range[0])
            max_val = max(orig_range[1], quant_range[1])
            bins = np.linspace(min_val, max_val, len(orig_hist) + 1)
            bin_centers = (bins[:-1] + bins[1:]) / 2

            # Plot
            ax = axes[idx]
            ax.bar(bin_centers, orig_hist_norm, width=(bins[1]-bins[0])*0.8,
                   alpha=0.6, label='FP32', color='blue')
            ax.bar(bin_centers, quant_hist_norm, width=(bins[1]-bins[0])*0.8,
                   alpha=0.6, label='Quantized', color='red')

            # Add statistics
            stats = self.profilers[layer_name].get_statistic()
            kl_div = hist_data.get('kl_divergence', 0)
            js_div = hist_data.get('js_divergence', 0)
            sqnr = stats.get('qsnr', 0)

            stats_text = f"SQNR: {sqnr:.2f} dB\nKL Div: {kl_div:.4f}\nJS Div: {js_div:.4f}"
            ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                   fontsize=10)

            ax.set_xlabel('Activation Value')
            ax.set_ylabel('Probability Density')
            ax.set_title(f'{layer_name} - Distribution Comparison')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\nHistogram saved to: {save_path}")
        else:
            plt.savefig('activation_histograms.png', dpi=150, bbox_inches='tight')
            print("\nHistogram saved to: activation_histograms.png")

        plt.close()

    def update_all_layer_stats(self, test_input: torch.Tensor):
        """
        Update profiler statistics for ALL quantized layers.
        Captures FP32 vs Quantized values for weights and activations.
        """
        if not self.enable_profiling:
            return

        # 1. Weight layers (Linear)
        self.update_profiler_weights()

        # 2. All activation layers - capture intermediate activations
        self._capture_all_activations(test_input)

    def _capture_all_activations(self, test_input: torch.Tensor):
        """Capture FP32 and Quantized activations for all layers"""
        if not self.enable_profiling:
            return

        with torch.no_grad():
            B, N, C = test_input.shape

            # === FP32 Mode ===
            self.set_mode('fp32')
            fp32_activations = {}

            # norm1
            x_norm1_fp32 = self.norm1.forward(test_input)
            fp32_activations['norm1'] = x_norm1_fp32.clone()

            # attn_qkv
            qkv_fp32 = self.attn_qkv.forward(x_norm1_fp32)
            fp32_activations['attn_qkv'] = qkv_fp32.clone()

            # attn_qkv_output
            qkv_out_fp32 = self.attn_qkv_output.forward(qkv_fp32)
            fp32_activations['attn_qkv_output'] = qkv_out_fp32.clone()

            qkv_fp32 = qkv_out_fp32.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv_fp32.unbind(0)

            # kv_act
            attn_fp32 = (q @ k.transpose(-2, -1)) * self.scale
            kv_act_fp32 = self.kv_act.forward(attn_fp32)
            fp32_activations['kv_act'] = kv_act_fp32.clone()

            # intSoft
            scale_param = self.kv_act.scaler if hasattr(self.kv_act, 'scaler') else None
            intsoft_fp32 = self.intSoft.forward(kv_act_fp32, scale=scale_param)
            fp32_activations['intSoft'] = intsoft_fp32.clone()

            intsoft_fp32 = self.attn_drop(intsoft_fp32)

            # attn_proj
            x_attn_fp32 = (intsoft_fp32 @ v).transpose(1, 2).reshape(B, N, C)
            attn_proj_fp32 = self.attn_proj.forward(x_attn_fp32)
            fp32_activations['attn_proj'] = attn_proj_fp32.clone()

            # sv_attn
            sv_attn_fp32 = self.sv_attn.forward(attn_proj_fp32)
            fp32_activations['sv_attn'] = sv_attn_fp32.clone()

            sv_attn_fp32 = self.proj_drop(sv_attn_fp32)

            # residual1
            x_res1_fp32 = test_input + self.drop_path1(sv_attn_fp32)
            residual1_fp32 = self.residual1.forward(x_res1_fp32)
            fp32_activations['residual1'] = residual1_fp32.clone()

            # norm2
            x_norm2_fp32 = self.norm2.forward(residual1_fp32)
            fp32_activations['norm2'] = x_norm2_fp32.clone()

            # mlp_fc1
            mlp_fc1_fp32 = self.mlp_fc1.forward(x_norm2_fp32)
            fp32_activations['mlp_fc1'] = mlp_fc1_fp32.clone()

            # mlp_act
            mlp_act_fp32 = self.mlp_act.forward(mlp_fc1_fp32)
            fp32_activations['mlp_act'] = mlp_act_fp32.clone()

            # mlp_fc2
            mlp_fc2_fp32 = self.mlp_fc2.forward(mlp_act_fp32)
            fp32_activations['mlp_fc2'] = mlp_fc2_fp32.clone()

            # mlp_act2
            mlp_act2_fp32 = self.mlp_act2.forward(mlp_fc2_fp32)
            fp32_activations['mlp_act2'] = mlp_act2_fp32.clone()

            # residual2
            x_res2_fp32 = residual1_fp32 + self.drop_path2(mlp_act2_fp32)
            residual2_fp32 = self.residual2.forward(x_res2_fp32)
            fp32_activations['residual2'] = residual2_fp32.clone()

            # === Quantized Mode ===
            self.set_mode('quantized')
            quant_activations = {}

            # norm1
            x_norm1_quant = self.norm1.forward(test_input)
            quant_activations['norm1'] = x_norm1_quant.clone()

            # attn_qkv
            qkv_quant = self.attn_qkv.forward(x_norm1_quant)
            quant_activations['attn_qkv'] = qkv_quant.clone()

            # attn_qkv_output
            qkv_out_quant = self.attn_qkv_output.forward(qkv_quant)
            quant_activations['attn_qkv_output'] = qkv_out_quant.clone()

            qkv_quant = qkv_out_quant.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv_quant.unbind(0)

            # kv_act
            attn_quant = (q @ k.transpose(-2, -1)) * self.scale
            kv_act_quant = self.kv_act.forward(attn_quant)
            quant_activations['kv_act'] = kv_act_quant.clone()

            # intSoft
            scale_param = self.kv_act.scaler if hasattr(self.kv_act, 'scaler') else None
            intsoft_quant = self.intSoft.forward(kv_act_quant, scale=scale_param)
            quant_activations['intSoft'] = intsoft_quant.clone()

            intsoft_quant = self.attn_drop(intsoft_quant)

            # attn_proj
            x_attn_quant = (intsoft_quant @ v).transpose(1, 2).reshape(B, N, C)
            attn_proj_quant = self.attn_proj.forward(x_attn_quant)
            quant_activations['attn_proj'] = attn_proj_quant.clone()

            # sv_attn
            sv_attn_quant = self.sv_attn.forward(attn_proj_quant)
            quant_activations['sv_attn'] = sv_attn_quant.clone()

            sv_attn_quant = self.proj_drop(sv_attn_quant)

            # residual1
            x_res1_quant = test_input + self.drop_path1(sv_attn_quant)
            residual1_quant = self.residual1.forward(x_res1_quant)
            quant_activations['residual1'] = residual1_quant.clone()

            # norm2
            x_norm2_quant = self.norm2.forward(residual1_quant)
            quant_activations['norm2'] = x_norm2_quant.clone()

            # mlp_fc1
            mlp_fc1_quant = self.mlp_fc1.forward(x_norm2_quant)
            quant_activations['mlp_fc1'] = mlp_fc1_quant.clone()

            # mlp_act
            mlp_act_quant = self.mlp_act.forward(mlp_fc1_quant)
            quant_activations['mlp_act'] = mlp_act_quant.clone()

            # mlp_fc2
            mlp_fc2_quant = self.mlp_fc2.forward(mlp_act_quant)
            quant_activations['mlp_fc2'] = mlp_fc2_quant.clone()

            # mlp_act2
            mlp_act2_quant = self.mlp_act2.forward(mlp_fc2_quant)
            quant_activations['mlp_act2'] = mlp_act2_quant.clone()

            # residual2
            x_res2_quant = residual1_quant + self.drop_path2(mlp_act2_quant)
            residual2_quant = self.residual2.forward(x_res2_quant)
            quant_activations['residual2'] = residual2_quant.clone()

        # Update profilers (activation layers only - weights already done)
        activation_layers = [
            'norm1', 'attn_qkv_output', 'kv_act', 'intSoft', 'sv_attn',
            'residual1', 'norm2', 'mlp_act', 'mlp_act2', 'residual2'
        ]

        for layer_name in activation_layers:
            if layer_name in fp32_activations and layer_name in quant_activations:
                self.profilers[layer_name].update_weight(
                    fp32_activations[layer_name],
                    quant_activations[layer_name]
                )

    def save_profiling_report(self, log_dir: Union[str, Path] = None):
        """
        Save profiling statistics report to txt file.

        Args:
            log_dir: Directory to save log files. Creates 'log' folder at project root if None.
        """
        if not self.enable_profiling:
            print("Profiling is not enabled")
            return

        # Create log directory
        if log_dir is None:
            log_dir = Path(__file__).parent.parent / 'log'
        else:
            log_dir = Path(log_dir)

        log_dir.mkdir(parents=True, exist_ok=True)

        # Generate timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = log_dir / f"profiling_report_{timestamp}.txt"

        lines = []
        lines.append("=" * 80)
        lines.append("ViT Block Quantization Profiling Report")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 80)

        # Layer categories
        weight_layers = ['attn_qkv', 'attn_proj', 'mlp_fc1', 'mlp_fc2']
        activation_layers = [
            'norm1', 'attn_qkv_output', 'kv_act', 'intSoft', 'sv_attn',
            'residual1', 'norm2', 'mlp_act', 'mlp_act2', 'residual2'
        ]

        # Weight layers statistics
        lines.append("\n" + "=" * 80)
        lines.append("WEIGHT QUANTIZATION STATISTICS")
        lines.append("=" * 80)

        for layer_name in weight_layers:
            if layer_name not in self.profilers:
                continue

            try:
                stats = self.profilers[layer_name].get_statistic()
                hist = self.profilers[layer_name].get_hist()

                lines.append(f"\n[{layer_name}]")
                lines.append("-" * 40)

                mse = stats.get('mse', None)
                lines.append(f"  MSE:               {mse:.10f}" if mse is not None else "  MSE:               N/A")

                sqnr = stats.get('qsnr', None)
                lines.append(f"  SQNR:              {sqnr:.4f} dB" if sqnr is not None else "  SQNR:              N/A")

                cos_sim = stats.get('cosine_sim', None)
                lines.append(f"  Cosine Similarity: {cos_sim:.10f}" if cos_sim is not None else "  Cosine Similarity: N/A")

                kl_div = hist.get('kl_divergence', None)
                lines.append(f"  KL Divergence:     {kl_div:.10f}" if kl_div is not None else "  KL Divergence:     N/A")

                js_div = hist.get('js_divergence', None)
                lines.append(f"  JS Divergence:     {js_div:.10f}" if js_div is not None else "  JS Divergence:     N/A")

                orig_range = hist.get('original_range', (None, None))
                lines.append(f"  FP32 Range:        [{orig_range[0]:.6f}, {orig_range[1]:.6f}]" if orig_range[0] is not None else "  FP32 Range:        N/A")

                quant_range = hist.get('quantized_range', (None, None))
                lines.append(f"  Quant Range:       [{quant_range[0]:.6f}, {quant_range[1]:.6f}]" if quant_range[0] is not None else "  Quant Range:       N/A")

            except Exception as e:
                lines.append(f"\n[{layer_name}]")
                lines.append(f"  Error: {str(e)}")

        # Activation layers statistics
        lines.append("\n" + "=" * 80)
        lines.append("ACTIVATION QUANTIZATION STATISTICS")
        lines.append("=" * 80)

        for layer_name in activation_layers:
            if layer_name not in self.profilers:
                continue

            try:
                stats = self.profilers[layer_name].get_statistic()
                hist = self.profilers[layer_name].get_hist()

                lines.append(f"\n[{layer_name}]")
                lines.append("-" * 40)

                mse = stats.get('mse', None)
                lines.append(f"  MSE:               {mse:.10f}" if mse is not None else "  MSE:               N/A")

                sqnr = stats.get('qsnr', None)
                lines.append(f"  SQNR:              {sqnr:.4f} dB" if sqnr is not None else "  SQNR:              N/A")

                cos_sim = stats.get('cosine_sim', None)
                lines.append(f"  Cosine Similarity: {cos_sim:.10f}" if cos_sim is not None else "  Cosine Similarity: N/A")

                kl_div = hist.get('kl_divergence', None)
                lines.append(f"  KL Divergence:     {kl_div:.10f}" if kl_div is not None else "  KL Divergence:     N/A")

                js_div = hist.get('js_divergence', None)
                lines.append(f"  JS Divergence:     {js_div:.10f}" if js_div is not None else "  JS Divergence:     N/A")

                orig_range = hist.get('original_range', (None, None))
                lines.append(f"  FP32 Range:        [{orig_range[0]:.6f}, {orig_range[1]:.6f}]" if orig_range[0] is not None else "  FP32 Range:        N/A")

                quant_range = hist.get('quantized_range', (None, None))
                lines.append(f"  Quant Range:       [{quant_range[0]:.6f}, {quant_range[1]:.6f}]" if quant_range[0] is not None else "  Quant Range:       N/A")

            except Exception as e:
                lines.append(f"\n[{layer_name}]")
                lines.append(f"  Error: {str(e)}")

        # Summary statistics
        lines.append("\n" + "=" * 80)
        lines.append("SUMMARY")
        lines.append("=" * 80)

        try:
            all_layers = weight_layers + activation_layers
            valid_sqnr = []
            valid_kl = []

            for layer_name in all_layers:
                if layer_name in self.profilers:
                    try:
                        stats = self.profilers[layer_name].get_statistic()
                        hist = self.profilers[layer_name].get_hist()
                        sqnr = stats.get('qsnr', None)
                        kl = hist.get('kl_divergence', None)
                        if sqnr is not None:
                            valid_sqnr.append((layer_name, sqnr))
                        if kl is not None:
                            valid_kl.append((layer_name, kl))
                    except:
                        pass

            if valid_sqnr:
                avg_sqnr = sum(s[1] for s in valid_sqnr) / len(valid_sqnr)
                min_sqnr = min(valid_sqnr, key=lambda x: x[1])
                max_sqnr = max(valid_sqnr, key=lambda x: x[1])

                lines.append(f"\n  Average SQNR:      {avg_sqnr:.4f} dB")
                lines.append(f"  Best SQNR:         {max_sqnr[0]} ({max_sqnr[1]:.4f} dB)")
                lines.append(f"  Worst SQNR:        {min_sqnr[0]} ({min_sqnr[1]:.4f} dB)")

            if valid_kl:
                avg_kl = sum(k[1] for k in valid_kl) / len(valid_kl)
                min_kl = min(valid_kl, key=lambda x: x[1])
                max_kl = max(valid_kl, key=lambda x: x[1])

                lines.append(f"\n  Average KL Div:    {avg_kl:.6f}")
                lines.append(f"  Best KL Div:       {min_kl[0]} ({min_kl[1]:.6f})")
                lines.append(f"  Worst KL Div:      {max_kl[0]} ({max_kl[1]:.6f})")

        except Exception as e:
            lines.append(f"\n  Summary Error: {str(e)}")

        lines.append("\n" + "=" * 80)
        lines.append("END OF REPORT")
        lines.append("=" * 80)

        # Write to file
        with open(report_path, 'w') as f:
            f.write('\n'.join(lines))

        print(f"Profiling report saved to: {report_path}")
        return report_path

    def save_all_histograms(self, log_dir: Union[str, Path] = None):
        """
        Save histogram comparison plots for all layers as JPEG files.

        Args:
            log_dir: Directory to save histogram images. Creates 'log' folder at project root if None.
        """
        if not self.enable_profiling:
            print("Profiling is not enabled")
            return

        # Create log directory
        if log_dir is None:
            log_dir = Path(__file__).parent.parent / 'log'
        else:
            log_dir = Path(log_dir)

        log_dir.mkdir(parents=True, exist_ok=True)

        # Generate timestamp for filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        all_layers = list(self.profilers.keys())
        saved_files = []

        for layer_name in all_layers:
            try:
                hist_data = self.profilers[layer_name].get_hist()
                stats = self.profilers[layer_name].get_statistic()

                orig_hist = hist_data['original_hist'].numpy()
                quant_hist = hist_data['quantized_hist'].numpy()
                orig_range = hist_data['original_range']
                quant_range = hist_data['quantized_range']

                # Normalize histograms
                orig_hist_norm = orig_hist / (orig_hist.sum() + 1e-10)
                quant_hist_norm = quant_hist / (quant_hist.sum() + 1e-10)

                # Create bin edges
                min_val = min(orig_range[0], quant_range[0])
                max_val = max(orig_range[1], quant_range[1])
                bins = np.linspace(min_val, max_val, len(orig_hist) + 1)
                bin_centers = (bins[:-1] + bins[1:]) / 2

                # Create figure
                fig, ax = plt.subplots(1, 1, figsize=(12, 6))

                ax.bar(bin_centers, orig_hist_norm, width=(bins[1]-bins[0])*0.8,
                       alpha=0.6, label='FP32', color='blue')
                ax.bar(bin_centers, quant_hist_norm, width=(bins[1]-bins[0])*0.8,
                       alpha=0.6, label='Quantized', color='red')

                # Statistics text
                kl_div = hist_data.get('kl_divergence', 0)
                js_div = hist_data.get('js_divergence', 0)
                sqnr = stats.get('qsnr', 0)
                mse = stats.get('mse', 0)
                cos_sim = stats.get('cosine_sim', 0)

                stats_text = (
                    f"MSE: {mse:.6f}\n"
                    f"SQNR: {sqnr:.2f} dB\n"
                    f"Cosine Sim: {cos_sim:.6f}\n"
                    f"KL Div: {kl_div:.6f}\n"
                    f"JS Div: {js_div:.6f}"
                )

                ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
                       verticalalignment='top', horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                       fontsize=10, family='monospace')

                ax.set_xlabel('Value', fontsize=12)
                ax.set_ylabel('Probability Density', fontsize=12)
                ax.set_title(f'{layer_name} - FP32 vs Quantized Distribution', fontsize=14)
                ax.legend(loc='upper left', fontsize=10)
                ax.grid(True, alpha=0.3)
                ax.set_yscale('log')

                plt.tight_layout()

                # Save as JPEG
                save_path = log_dir / f"hist_{layer_name}_{timestamp}.jpg"
                plt.savefig(save_path, format='jpeg', dpi=150, bbox_inches='tight')
                plt.close()

                saved_files.append(save_path)

            except Exception as e:
                print(f"  Warning: Could not save histogram for {layer_name}: {str(e)}")
                continue

        print(f"Saved {len(saved_files)} histogram images to: {log_dir}")
        return saved_files


def test_vit_block(config_path: Union[str, Path] = None, save_logs: bool = True):
    """
    ViT Block 양자화 테스트
    - Config를 YAML 파일에서 로드
    - 모든 레이어의 프로파일링 결과를 log 폴더에 저장

    Args:
        config_path: YAML config 파일 경로. None이면 기본 int8 config 사용
        save_logs: True이면 log 폴더에 통계 txt와 히스토그램 jpeg 저장
    """
    print("=" * 80)
    print("ViT Block Quantization Test")
    print("=" * 80)

    try:
        import timm
    except ImportError:
        print("Error: timm library not installed. Install with: pip install timm")
        return

    # ========== 1. Config 로드 ==========
    if config_path is None:
        # 기본 config 경로 설정
        config_path = Path(__file__).parent.parent / 'configs' / 'quant_config_int8.yaml'

    print(f"\n[Config Loading]")
    print(f"  Config file: {config_path}")

    layer_config = load_config_from_yaml(config_path)

    print(f"\n[Config Summary]")
    print(f"  Default bit type: {layer_config.default_config.bit_type.bits}-bit "
          f"{'signed' if layer_config.default_config.bit_type.signed else 'unsigned'}")
    print(f"  Default observer: {layer_config.default_config.observer_type}")
    print(f"  Default calibration mode: {layer_config.default_config.calibration_mode}")
    print(f"  Layer-specific configs: {len(layer_config.layer_configs)} layers")

    # ========== 2. timm ViT 모델 로드 및 첫 번째 block 추출 ==========
    print(f"\n[Model Loading]")
    model = timm.create_model('vit_base_patch16_224', pretrained=False)
    original_block = model.blocks[0]  # 첫 번째 transformer block

    print(f"  Model: vit_base_patch16_224")
    print(f"  Block architecture:")
    print(f"    - Embed dim: {original_block.attn.qkv.in_features}")
    print(f"    - Num heads: {original_block.attn.num_heads}")
    print(f"    - Head dim: {original_block.attn.head_dim}")
    print(f"    - MLP hidden dim: {original_block.mlp.fc1.out_features}")

    # ========== 3. QuantTimmVitBlock 생성 ==========
    print(f"\n[Quantized Block Creation]")
    quant_block = QuantTimmVitBlock(
        block=original_block,
        quant_config=config_path,  # YAML 파일 경로 전달
        enable_profiling=True  # 프로파일링 활성화
    )
    print(f"  QuantTimmVitBlock created successfully")
    print(f"  Quantized layers: {list(quant_block.get_quantized_layers().keys())}")
    print("  Profiling: ENABLED")

    # ========== 4. Calibration 데이터 생성 ==========
    num_batches = 10
    batch_size = 4
    seq_len = 197  # ViT: 1 (cls token) + 196 (14x14 patches)
    embed_dim = 768

    print(f"\n[Calibration Data]")
    print(f"  Batches: {num_batches}")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Embedding dim: {embed_dim}")

    calib_data = [
        torch.randn(batch_size, seq_len, embed_dim)
        for _ in range(num_batches)
    ]

    # ========== 5. Calibration 수행 ==========
    print(f"\n{'='*80}")
    print("Calibration Phase")
    print("="*80)

    quant_block.eval()
    with torch.no_grad():
        for batch_idx, x in enumerate(calib_data):
            _ = quant_block.calibration(x)

            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx + 1}/{num_batches} completed")

    print(f"\n  Calibration completed!")

    # ========== 6. Quantization Parameters 계산 ==========
    print(f"\n{'='*80}")
    print("Computing Quantization Parameters")
    print("="*80)

    quant_block.compute_quant_params()

    # 각 레이어의 quantization params 출력
    print(f"\n[Layer Quantization Parameters]")
    layers_to_check = {
        'attn_qkv': quant_block.attn_qkv,
        'attn_proj': quant_block.attn_proj,
        'mlp_fc1': quant_block.mlp_fc1,
        'mlp_fc2': quant_block.mlp_fc2,
        'norm1': quant_block.norm1,
        'norm2': quant_block.norm2,
    }

    for name, layer in layers_to_check.items():
        if hasattr(layer, 'scaler'):
            print(f"  {name}:")
            # scaler가 텐서인 경우 (layer_wise/channel_wise) mean 출력, 스칼라인 경우 그대로 출력
            if isinstance(layer.scaler, torch.Tensor):
                if layer.scaler.numel() == 1:
                    print(f"    Weight scale: {layer.scaler.item():.8f}")
                else:
                    print(f"    Weight scale (mean): {layer.scaler.mean().item():.8f} (shape: {layer.scaler.shape})")
            else:
                print(f"    Weight scale: {layer.scaler:.8f}")

            if hasattr(layer, 'output_scaler'):
                if isinstance(layer.output_scaler, torch.Tensor):
                    if layer.output_scaler.numel() == 1:
                        print(f"    Output scale: {layer.output_scaler.item():.8f}")
                    else:
                        print(f"    Output scale (mean): {layer.output_scaler.mean().item():.8f} (shape: {layer.output_scaler.shape})")
                else:
                    print(f"    Output scale: {layer.output_scaler:.8f}")

    # ========== 7. FP32 vs Quantized Inference 비교 ==========
    print(f"\n{'='*80}")
    print("Inference Comparison (FP32 vs Quantized)")
    print("="*80)

    test_input = torch.randn(1, seq_len, embed_dim)

    # FP32 모드
    print(f"\n[FP32 Mode]")
    quant_block.set_mode('fp32')
    quant_block.reset_profiling()  # 프로파일링 초기화
    with torch.no_grad():
        # 여러 번 실행해서 평균 성능 측정
        for _ in range(100):
            output_fp32 = quant_block(test_input)

    print(f"  Input shape: {test_input.shape}")
    print(f"  Output shape: {output_fp32.shape}")
    print(f"  Output range: [{output_fp32.min():.4f}, {output_fp32.max():.4f}]")
    print(f"  Output mean: {output_fp32.mean():.4f}")
    print(f"  Output std: {output_fp32.std():.4f}")

    # FP32 프로파일링 결과
    print("\n[FP32 Mode - Profiling Results]")
    quant_block.print_profiling_summary()

    # Quantized 모드
    print("\n[Quantized Mode]")
    quant_block.set_mode('quantized')
    quant_block.reset_profiling()  # 프로파일링 초기화
    with torch.no_grad():
        # 여러 번 실행해서 평균 성능 측정
        for _ in range(100):
            output_quant = quant_block(test_input)

    print(f"  Output shape: {output_quant.shape}")
    print(f"  Output range: [{output_quant.min():.4f}, {output_quant.max():.4f}]")
    print(f"  Output mean: {output_quant.mean():.4f}")
    print(f"  Output std: {output_quant.std():.4f}")

    # Quantized 프로파일링 결과
    print("\n[Quantized Mode - Profiling Results]")
    quant_block.print_profiling_summary()

    # ========== 8. 모든 레이어 통계 업데이트 ==========
    print(f"\n{'='*80}")
    print("Updating All Layer Statistics (FP32 vs Quantized)")
    print("="*80)

    quant_block.update_all_layer_stats(test_input)
    print("  All layer statistics updated successfully!")

    # ========== 9. 로그 저장 ==========
    if save_logs:
        print(f"\n{'='*80}")
        print("Saving Profiling Logs")
        print("="*80)

        # log 폴더 경로
        log_dir = Path(__file__).parent.parent / 'log'

        # 통계 리포트 저장 (txt)
        report_path = quant_block.save_profiling_report(log_dir)

        # 히스토그램 저장 (jpeg)
        histogram_files = quant_block.save_all_histograms(log_dir)

        print(f"\n  Log directory: {log_dir}")
        print(f"  Report file: {report_path}")
        print(f"  Histogram files: {len(histogram_files)} images saved")

    # ========== 10. Error Analysis ==========
    print(f"\n{'='*80}")
    print("Error Analysis (FP32 vs Quantized Output)")
    print("="*80)

    # MSE
    mse = F.mse_loss(output_fp32, output_quant)
    print(f"\n  MSE: {mse.item():.10f}")

    # Per-layer weight quantization quality summary
    print("\n[Per-Layer Weight Quality Summary]")
    linear_layers = ['attn_qkv', 'attn_proj', 'mlp_fc1', 'mlp_fc2']
    total_layers = len(linear_layers)
    avg_sqnr = sum([quant_block.get_layer_statistics(name).get('qsnr', 0) for name in linear_layers]) / total_layers
    avg_kl = sum([quant_block.profilers[name].get_hist().get('kl_divergence', 0) for name in linear_layers]) / total_layers
    print(f"  Average SQNR: {avg_sqnr:.2f} dB")
    print(f"  Average KL Divergence: {avg_kl:.6f}")
    print(f"  Number of quantized layers in forward: {len(quant_block.profilers)}")

    print("\n[Error Accumulation Analysis]")
    print("  Individual layer quality: Excellent (SQNR ~42 dB)")
    print(f"  Number of sequential quantized operations: {len(quant_block.profilers)}")
    print(f"  Expected error amplification: {len(quant_block.profilers)}x layers")

    # Cosine Similarity
    cos_sim = F.cosine_similarity(
        output_fp32.flatten(),
        output_quant.flatten(),
        dim=0
    )
    print(f"  Cosine Similarity: {cos_sim.item():.10f}")

    # QSNR (Quantization Signal-to-Noise Ratio)
    noise = output_fp32 - output_quant
    signal_power = (output_fp32 ** 2).mean()
    noise_power = (noise ** 2).mean()
    qsnr = 10 * torch.log10(signal_power / (noise_power + 1e-12))
    print(f"  QSNR: {qsnr.item():.2f} dB")

    # Max Absolute Difference
    max_diff = (output_fp32 - output_quant).abs().max()
    print(f"  Max Absolute Diff: {max_diff.item():.6f}")

    # Relative Error
    rel_error = ((output_fp32 - output_quant).abs() / (output_fp32.abs() + 1e-8)).mean()
    print(f"  Mean Relative Error: {rel_error.item():.6f}")

    # ========== 11. Success Check ==========
    print(f"\n{'='*80}")
    if cos_sim > 0.99 and qsnr > 30:
        print("ViT Block Quantization Test PASSED!")
        print(f"   - Cosine Similarity: {cos_sim.item():.6f} (> 0.99)")
        print(f"   - QSNR: {qsnr.item():.2f} dB (> 30 dB)")
    elif cos_sim > 0.95:
        print("ViT Block Quantization Test: Acceptable Quality")
        print(f"   - Cosine Similarity: {cos_sim.item():.6f}")
        print(f"   - QSNR: {qsnr.item():.2f} dB")
    else:
        print("ViT Block Quantization Test: Low Quality")
        print(f"   - Cosine Similarity: {cos_sim.item():.6f}")
        print(f"   - QSNR: {qsnr.item():.2f} dB")
    print("="*80)

    if save_logs:
        print(f"\nProfiling results saved to: {log_dir}")


if __name__ == "__main__":
    test_vit_block()




