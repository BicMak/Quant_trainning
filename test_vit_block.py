"""
ViT Block 양자화 테스트 스크립트
- Observer: PercentileObserver
- Bit: 8-bit signed (symmetric)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.vit_block import QuantTimmVitBlock
from quant_config import QuantConfig, BitTypeConfig


def test_vit_block():
    """ViT Block 양자화 테스트"""
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
        calibration_mode='layer_wise',
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
    num_batches = 8  # 빠른 테스트를 위해 감소
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
            print(f"    Weight scale: {layer.scaler:.8f}")
            if hasattr(layer, 'output_scaler'):
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
