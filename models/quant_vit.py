"""
QVit: Quantized Vision Transformer
- QEmbedding (Patch + CLS + PosEmbed)
- QVitBlock x N (Norm + Attn + MLP + Residual)
- Head (Final Norm + Classifier)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union
from pathlib import Path

from .quant_embedding import QEmbedding
from .quant_vit_block import QVitBlock
from .ptq.quant_layernorm import QLayerNorm
from .ptq.quant_linear import QuantLinear
from .ptq.quant_act import QAct
from .ptq.layer_profiler.profiler import profiler as ProfilerClass
from quant_config import QuantConfig, BitTypeConfig
from utils.config_loader import load_config_from_yaml


class QVit(nn.Module):
    """
    Quantized Vision Transformer for timm models.

    Structure:
        Input [B, 3, 224, 224]
        → QEmbedding [B, 197, 768]
        → QVitBlock x N [B, 197, 768]
        → Final Norm [B, 197, 768]
        → Head (CLS token → Linear) [B, num_classes]

    Supported models:
        - vit_base_patch16_224
        - vit_small_patch16_224
        - vit_tiny_patch16_224
        - etc. (timm ViT variants)
    """

    def __init__(self,
                 model,  # timm ViT model
                 config_dir: Union[str, Path],
                 num_blocks: int = None):
        """
        Args:
            model: timm ViT model (e.g., timm.create_model('vit_base_patch16_224'))
            config_dir: Directory containing config files:
                       - embed_config.yaml
                       - attn_config.yaml
                       - mlp_config.yaml
                       - block_config.yaml
                       - head_config.yaml
            num_blocks: Number of blocks to quantize (None = all blocks)
        """
        super().__init__()


        config_dir = Path(config_dir)

        # Model info
        self.num_classes = model.num_classes
        self.embed_dim = model.embed_dim
        self.num_heads = model.blocks[0].attn.num_heads
        total_blocks = len(model.blocks)
        self.num_blocks = num_blocks if num_blocks is not None else total_blocks

        print(f"[QVit] Initializing quantized ViT model")
        print(f"  - Embed dim: {self.embed_dim}")
        print(f"  - Num heads: {self.num_heads}")
        print(f"  - Num classes: {self.num_classes}")
        print(f"  - Total blocks: {total_blocks}, Quantized: {self.num_blocks}")

        # === Config paths ===
        embed_config = config_dir / 'embed_config.yaml'
        attn_config = config_dir / 'attn_config.yaml'
        mlp_config = config_dir / 'mlp_config.yaml'
        block_config = config_dir / 'block_config.yaml'
        head_config = config_dir / 'head_config.yaml'

        # === 1. QEmbedding ===
        if embed_config.exists():
            self.embedding = QEmbedding(model, embed_config)
            print(f"  - QEmbedding: loaded from {embed_config.name}")
        else:
            # Fallback: use original embedding without quantization
            print(f"  - QEmbedding: config not found, using FP32 embedding")
            self.embedding = None
            self.patch_embed = model.patch_embed
            self.cls_token = model.cls_token
            self.pos_embed = model.pos_embed

        # === 2. QVitBlocks ===
        self.blocks = nn.ModuleList()
        for i in range(self.num_blocks):
            block = model.blocks[i]
            qblock = QVitBlock(
                block=block,
                attn_config_path=attn_config,
                mlp_config_path=mlp_config,
                block_config_path=block_config if block_config.exists() else None
            )
            self.blocks.append(qblock)
        print(f"  - QVitBlocks: {len(self.blocks)} blocks created")

        # Keep remaining FP32 blocks if not all quantized
        self.fp32_blocks = nn.ModuleList()
        if self.num_blocks < total_blocks:
            for i in range(self.num_blocks, total_blocks):
                self.fp32_blocks.append(model.blocks[i])
            print(f"  - FP32 blocks: {len(self.fp32_blocks)} blocks (not quantized)")

        # === 3. Final Norm ===
        if head_config.exists():
            head_configs = load_config_from_yaml(head_config)
            if 'final_norm' in head_configs:
                self.norm = QLayerNorm(
                    input_module=model.norm,
                    quant_config=head_configs['final_norm'],
                    layer_name='final_norm'
                )
            else:
                self.norm = model.norm
        else:
            self.norm = model.norm
        print(f"  - Final Norm: {type(self.norm).__name__}")

        # === 4. Head (Classifier) ===
        if head_config.exists() and 'head' in head_configs:
            head_cfg = head_configs['head']
            self.head = QuantLinear(
                input_module=model.head,
                weight_config=head_cfg.get('weight'),
                out_config=head_cfg.get('output'),
                layer_name='head'
            )
        else:
            self.head = model.head
        print(f"  - Head: {type(self.head).__name__}")

        # === Profiling ===
        self.enable_profiling = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through quantized ViT.

        Args:
            x: Input images [B, 3, H, W]

        Returns:
            Class logits [B, num_classes]
        """
        # 1. Embedding
        if self.embedding is not None:
            x = self.embedding.forward(x)
        else:
            # FP32 embedding fallback
            x = self.patch_embed(x)
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
            x = x + self.pos_embed

        # 2. Quantized Blocks
        for block in self.blocks:
            x = block.forward(x)

        # 3. FP32 Blocks (if any)
        for block in self.fp32_blocks:
            x = block(x)

        # 4. Final Norm
        if isinstance(self.norm, QLayerNorm):
            x = self.norm.forward(x)
        else:
            x = self.norm(x)

        # 5. Head (CLS token only)
        x = x[:, 0]  # [B, embed_dim]

        if isinstance(self.head, QuantLinear):
            x = self.head.forward(x)
        else:
            x = self.head(x)

        return x

    def calibration(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calibration pass to collect statistics.

        Args:
            x: Input images [B, 3, H, W]

        Returns:
            Class logits [B, num_classes]
        """
        # 1. Embedding
        if self.embedding is not None:
            x = self.embedding.calibration(x)
        else:
            x = self.patch_embed(x)
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
            x = x + self.pos_embed

        # 2. Quantized Blocks
        for block in self.blocks:
            x = block.calibration(x)

        # 3. FP32 Blocks (if any)
        for block in self.fp32_blocks:
            x = block(x)

        # 4. Final Norm
        if isinstance(self.norm, QLayerNorm):
            x = self.norm.calibration(x)
        else:
            x = self.norm(x)

        # 5. Head
        x = x[:, 0]

        if isinstance(self.head, QuantLinear):
            x = self.head.calibration(x)
        else:
            x = self.head(x)

        return x

    def compute_quant_params(self, calib_loader=None):
        """Compute quantization parameters for all layers."""
        print("[QVit] Computing quantization parameters...")

        # Embedding
        if self.embedding is not None:
            self.embedding.compute_quant_params(calib_loader)
            print("  - Embedding: done")

        # Blocks
        for i, block in enumerate(self.blocks):
            block.compute_quant_params(calib_loader)
            if (i + 1) % 4 == 0 or i == len(self.blocks) - 1:
                print(f"  - Blocks: {i + 1}/{len(self.blocks)} done")

        # Final Norm
        if isinstance(self.norm, QLayerNorm):
            self.norm.compute_quant_params(calib_loader)
            print("  - Final Norm: done")

        # Head
        if isinstance(self.head, QuantLinear):
            self.head.compute_output_quant_params()
            print("  - Head: done")

        print("[QVit] Quantization parameters computed!")

    def set_mode(self, mode: str):
        """Set quantization mode ('fp32' or 'quantized')."""
        # Embedding
        if self.embedding is not None:
            self.embedding.set_mode(mode)

        # Blocks
        for block in self.blocks:
            block.set_mode(mode)

        # Final Norm
        if isinstance(self.norm, QLayerNorm):
            self.norm.mode = mode

        # Head
        if isinstance(self.head, QuantLinear):
            self.head.mode = mode

    def set_profiling(self, enable: bool):
        """Enable/disable profiling for all layers."""
        self.enable_profiling = enable

        # Embedding
        if self.embedding is not None:
            self.embedding.set_profiling(enable)

        # Blocks
        for block in self.blocks:
            block.set_profiling(enable)

        # Final Norm
        if isinstance(self.norm, QLayerNorm):
            self.norm.enable_profiling = enable

        # Head
        if isinstance(self.head, QuantLinear):
            self.head.enable_profiling = enable

    def get_quantized_layers(self) -> Dict[str, nn.Module]:
        """Returns all quantized layers."""
        layers = {}

        # Embedding
        if self.embedding is not None:
            for name, layer in self.embedding.get_quantized_layers().items():
                layers[f'embed_{name}'] = layer

        # Blocks
        for i, block in enumerate(self.blocks):
            for name, layer in block.get_quantized_layers().items():
                layers[f'block{i}_{name}'] = layer

        # Final Norm
        if isinstance(self.norm, QLayerNorm):
            layers['final_norm'] = self.norm

        # Head
        if isinstance(self.head, QuantLinear):
            layers['head'] = self.head

        return layers

    def get_profiler_list(self) -> List[tuple]:
        """
        Get profiler objects as a list with layer names.

        Returns:
            List of tuples: [(layer_name, profiler_object), ...]
        """
        def _extract_profilers(name_prefix: str, prof_data):
            """Extract profiler objects from various formats"""
            result = []
            if prof_data is None:
                return result

            if isinstance(prof_data, ProfilerClass):
                result.append((name_prefix, prof_data))
            elif isinstance(prof_data, dict):
                if 'weight' in prof_data and prof_data['weight'] is not None:
                    result.append((f'{name_prefix}_weight', prof_data['weight']))
                if 'output' in prof_data and prof_data['output'] is not None:
                    result.append((f'{name_prefix}_output', prof_data['output']))
            return result

        profiler_list = []

        # Embedding
        if self.embedding is not None:
            embed_profs = self.embedding.get_profiler()
            for name, prof in embed_profs.items():
                profiler_list.extend(_extract_profilers(f'embed_{name}', prof))

        # Blocks
        for i, block in enumerate(self.blocks):
            block_profs = block.get_profiler_list()
            for name, prof in block_profs:
                profiler_list.append((f'block{i}_{name}', prof))

        # Final Norm
        if isinstance(self.norm, QLayerNorm):
            norm_prof = self.norm.get_profiler()
            profiler_list.extend(_extract_profilers('final_norm', norm_prof))

        # Head
        if isinstance(self.head, QuantLinear):
            head_prof = self.head.get_profiler()
            profiler_list.extend(_extract_profilers('head', head_prof))

        return profiler_list

    def get_profiler_names(self) -> List[str]:
        """Get list of layer names that have profilers enabled."""
        return [name for name, _ in self.get_profiler_list()]

    def print_model_summary(self):
        """Print summary of the quantized model."""
        print("\n" + "=" * 70)
        print("QVit Model Summary")
        print("=" * 70)

        print(f"\n[Model Info]")
        print(f"  Embed dim: {self.embed_dim}")
        print(f"  Num heads: {self.num_heads}")
        print(f"  Num classes: {self.num_classes}")
        print(f"  Quantized blocks: {len(self.blocks)}")
        print(f"  FP32 blocks: {len(self.fp32_blocks)}")

        layers = self.get_quantized_layers()
        print(f"\n[Quantized Layers]")
        print(f"  Total: {len(layers)} layers")

        # Count by type
        type_counts = {}
        for name, layer in layers.items():
            layer_type = type(layer).__name__
            type_counts[layer_type] = type_counts.get(layer_type, 0) + 1

        for layer_type, count in type_counts.items():
            print(f"  - {layer_type}: {count}")

        print("=" * 70)

    def get_block_sqnr_summary(self, x_test: torch.Tensor, cumulative: bool = True) -> Dict[int, float]:
        """
        Get per-block SQNR summary.

        Args:
            x_test: Test input [B, 3, H, W]
            cumulative: If True, measure cumulative error (realistic E2E performance)
                       If False, measure each block independently (block's own quality)

        Returns:
            Dict mapping block index to SQNR value
        """
        sqnr_dict = {}

        # Get embeddings first (FP32)
        if self.embedding is not None:
            self.embedding.set_mode('fp32')
            x_embed = self.embedding.forward(x_test.clone())
        else:
            x_embed = self.patch_embed(x_test)
            cls_token = self.cls_token.expand(x_embed.shape[0], -1, -1)
            x_embed = torch.cat((cls_token, x_embed), dim=1)
            x_embed = x_embed + self.pos_embed

        if cumulative:
            # Cumulative mode: error accumulates through blocks
            x_fp32 = x_embed.clone()
            x_quant = x_embed.clone()

            with torch.no_grad():
                for i, block in enumerate(self.blocks):
                    # FP32 path
                    block.set_mode('fp32')
                    out_fp32 = block.forward(x_fp32)

                    # Quantized path (with accumulated error)
                    block.set_mode('quantized')
                    out_quant = block.forward(x_quant)

                    # SQNR: compare FP32 output vs Quantized output
                    signal_power = (out_fp32 ** 2).mean()
                    noise_power = ((out_fp32 - out_quant) ** 2).mean()
                    sqnr = 10 * torch.log10(signal_power / (noise_power + 1e-10))
                    sqnr_dict[i] = sqnr.item()

                    # Update for next block
                    x_fp32 = out_fp32
                    x_quant = out_quant
        else:
            # Independent mode: each block tested with same FP32 input
            # This measures each block's intrinsic quantization quality
            x_input = x_embed.clone()

            with torch.no_grad():
                for i, block in enumerate(self.blocks):
                    # FP32 output
                    block.set_mode('fp32')
                    out_fp32 = block.forward(x_input)

                    # Quantized output (same input!)
                    block.set_mode('quantized')
                    out_quant = block.forward(x_input)

                    # SQNR
                    signal_power = (out_fp32 ** 2).mean()
                    noise_power = ((out_fp32 - out_quant) ** 2).mean()
                    sqnr = 10 * torch.log10(signal_power / (noise_power + 1e-10))
                    sqnr_dict[i] = sqnr.item()

                    # Move to next block's input (FP32 path)
                    x_input = out_fp32

        return sqnr_dict

    def get_block_fisher_summary(self,
                                  data_loader,
                                  num_samples: int = 100,
                                  use_labels: bool = False) -> Dict[int, float]:
        """
        블록별 Fisher Information 계산.

        Fisher Information = E[||∂L/∂h||²]
        - 각 블록 출력에 대한 loss의 gradient 크기
        - 높을수록 해당 블록이 양자화에 민감함

        Args:
            data_loader: Calibration data loader
            num_samples: Number of samples for estimation
            use_labels: Use real labels (True) or model predictions (False)

        Returns:
            Dict mapping block index to Fisher trace value
        """
        from utils.hessian_analyzer import HessianAnalyzer

        analyzer = HessianAnalyzer(self)
        return analyzer.compute_block_fisher(data_loader, num_samples, use_labels)

    def get_sensitivity_analysis(self,
                                  data_loader,
                                  num_samples: int = 100) -> Dict[str, Dict]:
        """
        블록별 종합 민감도 분석.

        Returns:
            Dict with 'fisher' and 'perturbation' results
        """
        from utils.hessian_analyzer import analyze_block_sensitivity

        return analyze_block_sensitivity(self, data_loader, num_samples)

    def strip_for_deploy(self):
        """
        Remove unnecessary components for deployment.

        Removes:
            - Observer: Calibration 전용, inference에 불필요
            - Profiler: 개발/디버깅 전용

        Keeps:
            - Quantizer: scale, zero_point 보유, forward에서 사용
            - quant_weight: INT8 weight 저장
        """
        removed_count = {'observer': 0, 'profiler': 0}

        for name, module in self.named_modules():
            # Observer 제거
            for attr in ['observer', 'output_observer', 'weight_observer']:
                if hasattr(module, attr) and getattr(module, attr) is not None:
                    setattr(module, attr, None)
                    removed_count['observer'] += 1

            # Profiler 제거
            for attr in ['profiler', 'output_profiler', 'weight_profiler']:
                if hasattr(module, attr) and getattr(module, attr) is not None:
                    setattr(module, attr, None)
                    removed_count['profiler'] += 1

            # enable_profiling 플래그 비활성화
            if hasattr(module, 'enable_profiling'):
                module.enable_profiling = False

        print(f"[QVit] Stripped for deployment:")
        print(f"  - Observers removed: {removed_count['observer']}")
        print(f"  - Profilers removed: {removed_count['profiler']}")

        return self

    def export_onnx(self,
                    save_path: str,
                    input_shape: tuple = (1, 3, 224, 224),
                    opset_version: int = 18,
                    simplify: bool = True):
        """
        Export model to ONNX format for deployment.

        Args:
            save_path: Path to save ONNX file
            input_shape: Input tensor shape (B, C, H, W)
            opset_version: ONNX opset version (18+ recommended)
            simplify: Run onnx-simplifier after export

        Usage:
            qvit.strip_for_deploy()
            qvit.export_onnx('qvit_int8.onnx')
        """
        import torch.onnx

        # 1. Set to eval & quantized mode
        self.eval()
        self.set_mode('quantized')
        self.set_profiling(False)

        # 2. Dummy input
        device = next(self.parameters()).device
        dummy_input = torch.randn(*input_shape).to(device)

        # 3. Export
        print(f"[QVit] Exporting to ONNX: {save_path}")
        torch.onnx.export(
            self,
            dummy_input,
            save_path,
            opset_version=opset_version,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            },
            do_constant_folding=True
        )
        print(f"  - ONNX export complete")

        # 4. Simplify (optional)
        if simplify:
            try:
                import onnx
                from onnxsim import simplify as onnx_simplify

                print(f"  - Running onnx-simplifier...")
                model = onnx.load(save_path)
                model_simplified, check = onnx_simplify(model)

                if check:
                    onnx.save(model_simplified, save_path)
                    print(f"  - Simplified successfully")
                else:
                    print(f"  - Simplification check failed, keeping original")
            except ImportError:
                print(f"  - onnx-simplifier not installed, skipping")

        print(f"[QVit] Export complete: {save_path}")
        return save_path

    def save_quantized_weights(self, save_path: str):
        """
        Save quantized weights and parameters for deployment.

        Saves:
            - quant_weight (INT8)
            - scale, zero_point
            - bias (FP32)

        Args:
            save_path: Path to save (.pt file)
        """
        state = {
            'model_info': {
                'embed_dim': self.embed_dim,
                'num_heads': self.num_heads,
                'num_classes': self.num_classes,
                'num_blocks': len(self.blocks)
            },
            'quantized_params': {}
        }

        for name, module in self.named_modules():
            params = {}

            # INT8 weight
            if hasattr(module, 'quant_weight') and module.quant_weight is not None:
                params['quant_weight'] = module.quant_weight.to(torch.int8)

            # Scale & Zero point
            if hasattr(module, 'weight_scaler'):
                params['weight_scale'] = module.weight_scaler
                params['weight_zero'] = module.weight_zero
            if hasattr(module, 'scaler'):
                params['scale'] = module.scaler
                params['zero'] = module.zero
            if hasattr(module, 'output_scaler') and module.output_scaler is not None:
                params['output_scale'] = module.output_scaler
                params['output_zero'] = module.output_zero

            # Bias
            if hasattr(module, 'bias') and module.bias is not None:
                params['bias'] = module.bias

            if params:
                state['quantized_params'][name] = params

        torch.save(state, save_path)
        print(f"[QVit] Quantized weights saved: {save_path}")
        print(f"  - Layers: {len(state['quantized_params'])}")

        return save_path
