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
            quant_config=quant_config,
            act_module=None
        )
        self.attn_proj = QuantLinear(
            input_module=block.attn.proj,
            quant_config=quant_config
        )

        # kv_act: IntSoftmax에 스칼라 scale을 전달해야 하므로 강제로 layer_wise
        kv_act_config = QuantConfig(
            calibration_mode='layer_wise',  # 강제로 layer_wise
            bit_type=quant_config.bit_type,
            observer_type=quant_config.observer_type,
            percentile_alpha=quant_config.percentile_alpha,
            percentile_sigma=quant_config.percentile_sigma
        )
        self.kv_act = QAct(
            quant_config=kv_act_config,
            act_module=None
        )
        self.sv_attn= QAct(
            quant_config=quant_config,
            act_module=None
        )
        self.residual1= QAct(
            quant_config=quant_config,
            act_module=None
        )
        self.residual2= QAct(
            quant_config=quant_config,
            act_module=None
        )

        # === MLP 부분 ===
        self.mlp_fc1 = QuantLinear(
            input_module=block.mlp.fc1,
            quant_config=quant_config
        )
        self.mlp_act = QAct(
            quant_config=quant_config,
            act_module=block.mlp.act
        )
        self.mlp_fc2 = QuantLinear(
            input_module=block.mlp.fc2,
            quant_config=quant_config
        )
        self.mlp_act2 = QAct(
            quant_config=quant_config,
            act_module=None
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

        self.drop_path1 = block.drop_path1
        self.drop_path2 = block.drop_path2

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
        attn = self.kv_act.forward(attn)

        # IntSoftmax에 scale 전달 (quantized mode에서는 kv_act.scaler 사용, fp32에서는 scale=None이어도 됨)
        scale_param = self.kv_act.scaler if hasattr(self.kv_act, 'scaler') else None
        attn = self.intSoft.forward(attn, scale=scale_param)

        attn = self.attn_drop(attn)

        # Combine heads
        x_attn = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x_attn = self.attn_proj.forward(x_attn)
        x_attn = self.sv_attn.forward(x_attn)
        x_attn = self.proj_drop(x_attn)

        # Residual 1
        x = x + self.drop_path1(x_attn)
        x = self.residual1.forward(x)

        # === MLP block ===
        x_norm = self.norm2.forward(x)
        x_mlp = self.mlp_fc1.forward(x_norm)
        x_mlp = self.mlp_act.forward(x_mlp)
        x_mlp = self.mlp_fc2.forward(x_mlp)
        x_mlp = self.mlp_act2.forward(x_mlp)

        # Residual 2
        x = x + self.drop_path2(x_mlp)
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


def test_vit_block():
    """
    ViT Block 양자화 테스트
    - Observer: PercentileObserver
    - Bit: 8-bit signed (symmetric)
    """
    print("=" * 80)
    print("ViT Block Quantization Test")
    print("=" * 80)

    try:
        import timm
    except ImportError:
        print("Error: timm library not installed. Install with: pip install timm")
        return

    # ========== 1. Config 설정 (8-bit symmetric, PercentileObserver) ==========
    bit_config = BitTypeConfig(bits=8, signed=True, name='int8')
    quant_config = QuantConfig(
        calibration_mode='channel_wise',
        bit_type=bit_config,
        observer_type='PercentileObserver'
    )

    print(f"\n[Config]")
    print(f"  Bit Type: {bit_config.bits}-bit {'signed' if bit_config.signed else 'unsigned'} (symmetric)")
    print(f"  Observer: {quant_config.observer_type}")
    print(f"  Calibration Mode: {quant_config.calibration_mode}")

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
        quant_config=quant_config
    )
    print(f"  QuantTimmVitBlock created successfully")
    print(f"  Quantized layers: {list(quant_block.get_quantized_layers().keys())}")

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
    with torch.no_grad():
        output_fp32 = quant_block(test_input)

    print(f"  Input shape: {test_input.shape}")
    print(f"  Output shape: {output_fp32.shape}")
    print(f"  Output range: [{output_fp32.min():.4f}, {output_fp32.max():.4f}]")
    print(f"  Output mean: {output_fp32.mean():.4f}")
    print(f"  Output std: {output_fp32.std():.4f}")

    # Quantized 모드
    print(f"\n[Quantized Mode]")
    quant_block.set_mode('quantized')
    with torch.no_grad():
        output_quant = quant_block(test_input)

    print(f"  Output shape: {output_quant.shape}")
    print(f"  Output range: [{output_quant.min():.4f}, {output_quant.max():.4f}]")
    print(f"  Output mean: {output_quant.mean():.4f}")
    print(f"  Output std: {output_quant.std():.4f}")

    # ========== 8. Error Analysis ==========
    print(f"\n{'='*80}")
    print("Error Analysis")
    print("="*80)

    # MSE
    mse = F.mse_loss(output_fp32, output_quant)
    print(f"\n  MSE: {mse.item():.10f}")

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

    # ========== 9. Success Check ==========
    print(f"\n{'='*80}")
    if cos_sim > 0.99 and qsnr > 30:
        print("✅ ViT Block Quantization Test PASSED!")
        print(f"   - Cosine Similarity: {cos_sim.item():.6f} (> 0.99)")
        print(f"   - QSNR: {qsnr.item():.2f} dB (> 30 dB)")
    elif cos_sim > 0.95:
        print("⚠️  ViT Block Quantization Test: Acceptable Quality")
        print(f"   - Cosine Similarity: {cos_sim.item():.6f}")
        print(f"   - QSNR: {qsnr.item():.2f} dB")
    else:
        print("❌ ViT Block Quantization Test: Low Quality")
        print(f"   - Cosine Similarity: {cos_sim.item():.6f}")
        print(f"   - QSNR: {qsnr.item():.2f} dB")
    print("="*80)


if __name__ == "__main__":
    test_vit_block()




