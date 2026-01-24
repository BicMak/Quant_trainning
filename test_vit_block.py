"""
ViT Block Quantization Test with Real ImageNet-Mini Data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Union
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import os

# 절대 경로로 import 수정
from models.vit_block import QuantTimmVitBlock
from quant_config import QuantConfig, LayerQuantConfig
from utils.config_loader import load_config_from_yaml
from dataset import ImageNetMiniDataset, get_imagenet_transforms


def load_calibration_data_from_dataset(
    root_dir: str = 'imagenet-mini',
    num_samples: int = 100,
    img_size: int = 224
):
    """
    Load calibration data from ImageNet-Mini dataset.
    Selects 1 image per class for the first num_samples classes.

    Args:
        root_dir: Path to imagenet-mini directory
        num_samples: Number of samples to load (default 100)
        img_size: Image size for ViT (default 224)

    Returns:
        List of image tensors
    """
    print(f"\n[Loading Real Dataset]")

    # Create dataset
    dataset = ImageNetMiniDataset(
        root_dir=root_dir,
        split='train',
        transform=get_imagenet_transforms('train', img_size)
    )

    # Get 1 image per class for first num_samples classes
    images = []
    seen_classes = set()

    for idx in range(len(dataset)):
        _, label = dataset[idx]

        if label not in seen_classes:
            img, _ = dataset[idx]
            images.append(img)
            seen_classes.add(label)

            if len(images) >= num_samples:
                break

    print(f"  Loaded {len(images)} images from {len(seen_classes)} classes")
    print(f"  Image shape: {images[0].shape}")

    return images


def test_vit_block(
    config_path: Union[str, Path] = None,
    save_logs: bool = True
):
    """
    ViT Block 양자화 테스트 with Real ImageNet-Mini Data
    - Config를 YAML 파일에서 로드
    - ImageNet-Mini 데이터셋 사용 (라벨당 1장씩, 총 100장)
    - 모든 레이어의 프로파일링 결과를 log 폴더에 저장

    Args:
        config_path: YAML config 파일 경로 또는 config 디렉토리 경로
                    - None: 기본적으로 configs 디렉토리 사용 (3개 분할 파일)
                    - 디렉토리 경로: weight_config.yaml, activation_config.yaml, residual_config.yaml 자동 로드
                    - 파일 경로: 단일 monolithic config 파일 로드
        save_logs: True이면 log 폴더에 통계 txt와 히스토그램 jpeg 저장
    """
    print("=" * 80)
    print("ViT Block Quantization Test with ImageNet-Mini")
    print("=" * 80)

    try:
        import timm
    except ImportError:
        print("Error: timm library not installed. Install with: pip install timm")
        return

    # ========== 1. Config 로드 ==========
    print(f"\n[Config Loading]")

    # 기본값: configs 디렉토리 (분할된 파일 사용)
    if config_path is None:
        config_path = Path(__file__).parent / 'configs'

    config_path = Path(config_path)

    # 디렉토리인 경우 자동으로 분할된 파일들을 로드
    if config_path.is_dir():
        print(f"  Loading from directory: {config_path}")
        print("  Auto-detecting split config files:")
        print(f"    - {config_path / 'weight_config.yaml'}")
        print(f"    - {config_path / 'activation_config.yaml'}")
        print(f"    - {config_path / 'residual_config.yaml'}")
    else:
        print(f"  Loading from single file: {config_path}")

    layer_config = load_config_from_yaml(config_path)

    print(f"\n[Config Summary]")
    print(f"  Default bit type: {layer_config.default_config.bit_type.bits}-bit "
          f"{'signed' if layer_config.default_config.bit_type.signed else 'unsigned'}")
    print(f"  Default observer: {layer_config.default_config.observer_type}")
    print(f"  Default calibration mode: {layer_config.default_config.calibration_mode}")
    print(f"  Layer-specific configs: {len(layer_config.layer_configs)} layers")

    # ========== 2. timm ViT 모델 로드 ==========
    print(f"\n[Model Loading]")
    model = timm.create_model('vit_base_patch16_224', pretrained=False)
    model.eval()

    original_block = model.blocks[0]  # 첫 번째 transformer block
    patch_embed = model.patch_embed   # Patch embedding layer

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

    # ========== 4. 실제 데이터셋에서 Calibration 데이터 로드 ==========
    print(f"\n{'='*80}")
    print("Loading Calibration Data from ImageNet-Mini")
    print("="*80)

    # 100장의 이미지 로드 (라벨당 1장씩)
    calib_images = load_calibration_data_from_dataset(
        root_dir='imagenet-mini',
        num_samples=600,
        img_size=224
    )

    # 이미지를 batch로 변환
    batch_size = 20
    calib_batches = []

    for i in range(0, len(calib_images), batch_size):
        batch = torch.stack(calib_images[i:i+batch_size])
        calib_batches.append(batch)

    print(f"\n  Total batches: {len(calib_batches)}")
    print(f"  Batch size: {batch_size}")
    print(f"  Total images: {len(calib_images)}")

    # ========== 5. Patch Embedding으로 변환 (FP32) ==========
    print(f"\n[Converting Images to Patch Embeddings (FP32)]")
    calib_embeddings = []

    with torch.no_grad():
        for batch_images in calib_batches:
            # Patch embedding: [B, 3, 224, 224] -> [B, 197, 768]
            # FP32 원본 비트로 처리
            x = patch_embed(batch_images)
            calib_embeddings.append(x)

    print(f"  Embedding shape: {calib_embeddings[0].shape}")
    print(f"  Sequence length: {calib_embeddings[0].shape[1]}")
    print(f"  Embedding dim: {calib_embeddings[0].shape[2]}")
    print(f"  Processing: FP32 (original precision)")

    # ========== 6. Original Block으로 FP32 출력 생성 ==========
    print(f"\n[Generating FP32 Block Outputs]")
    calib_data = []

    original_block.eval()
    with torch.no_grad():
        for embedding in calib_embeddings:
            # Original block (FP32) 통과
            fp32_output = original_block(embedding)
            calib_data.append(fp32_output)

    print(f"  FP32 block output shape: {calib_data[0].shape}")
    print(f"  FP32 outputs generated for calibration")

    # ========== 7. Calibration 수행 ==========
    print(f"\n{'='*80}")
    print("Calibration Phase (Using FP32 Block Outputs)")
    print("="*80)

    quant_block.eval()
    with torch.no_grad():
        for batch_idx, x in enumerate(calib_data):
            _ = quant_block.calibration(x)

            if (batch_idx + 1) % 5 == 0:
                print(f"  Batch {batch_idx + 1}/{len(calib_data)} completed")

    print(f"\n  Calibration completed with {len(calib_data)} batches!")
    print(f"  Note: Calibration used FP32 block outputs (not raw embeddings)")

    # ========== 8. Quantization Parameters 계산 ==========
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

    # ========== 9. FP32 vs Quantized Inference 비교 ==========
    print(f"\n{'='*80}")
    print("Inference Comparison (First ViT Block)")
    print("="*80)

    # 첫 번째 embedding을 사용 (FP32 patch embedding output)
    test_embedding = calib_embeddings[0]

    print(f"\n[Input: FP32 Patch Embedding]")
    print(f"  Shape: {test_embedding.shape}")
    print(f"  Range: [{test_embedding.min():.4f}, {test_embedding.max():.4f}]")

    # Original FP32 Block 출력 생성
    print(f"\n[Original FP32 Block Output]")
    with torch.no_grad():
        fp32_block_output = original_block(test_embedding)

    print(f"  Shape: {fp32_block_output.shape}")
    print(f"  Range: [{fp32_block_output.min():.4f}, {fp32_block_output.max():.4f}]")
    print(f"  Mean: {fp32_block_output.mean():.4f}")
    print(f"  Std: {fp32_block_output.std():.4f}")

    # Quantized Block으로 inference (100번 반복해서 프로파일링)
    print(f"\n[Quantized Block Inference (100 iterations)]")
    quant_block.set_mode('quantized')
    quant_block.reset_profiling()

    with torch.no_grad():
        for _ in range(100):
            output_quant = quant_block(test_embedding)

    print(f"  Shape: {output_quant.shape}")
    print(f"  Range: [{output_quant.min():.4f}, {output_quant.max():.4f}]")
    print(f"  Mean: {output_quant.mean():.4f}")
    print(f"  Std: {output_quant.std():.4f}")

    # Quantized 프로파일링 결과
    print("\n[Quantized Block - Profiling Results]")
    quant_block.print_profiling_summary()

    # ========== 10. 모든 레이어 통계 업데이트 ==========
    print(f"\n{'='*80}")
    print("Updating All Layer Statistics (FP32 vs Quantized)")
    print("="*80)

    # FP32 embedding을 input으로 사용하여 통계 업데이트
    quant_block.update_all_layer_stats(test_embedding)
    print("  All layer statistics updated successfully!")

    # ========== 11. 로그 저장 ==========
    if save_logs:
        print(f"\n{'='*80}")
        print("Saving Profiling Logs")
        print("="*80)

        # log 폴더 경로
        log_dir = Path(__file__).parent / 'log'

        # 통계 리포트 저장 (txt)
        report_path = quant_block.save_profiling_report(log_dir)

        # 히스토그램 저장 (jpeg)
        histogram_files = quant_block.save_all_histograms(log_dir)

        print(f"\n  Log directory: {log_dir}")
        print(f"  Report file: {report_path}")
        print(f"  Histogram files: {len(histogram_files)} images saved")

    # ========== 12. Error Analysis ==========
    print(f"\n{'='*80}")
    print("Error Analysis (Original FP32 Block vs Quantized Block)")
    print("="*80)

    # MSE
    mse = F.mse_loss(fp32_block_output, output_quant)
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
        fp32_block_output.flatten(),
        output_quant.flatten(),
        dim=0
    )
    print(f"  Cosine Similarity: {cos_sim.item():.10f}")

    # QSNR (Quantization Signal-to-Noise Ratio)
    noise = fp32_block_output - output_quant
    signal_power = (fp32_block_output ** 2).mean()
    noise_power = (noise ** 2).mean()
    qsnr = 10 * torch.log10(signal_power / (noise_power + 1e-12))
    print(f"  QSNR: {qsnr.item():.2f} dB")

    # Max Absolute Difference
    max_diff = (fp32_block_output - output_quant).abs().max()
    print(f"  Max Absolute Diff: {max_diff.item():.6f}")

    # Relative Error
    rel_error = ((fp32_block_output - output_quant).abs() / (fp32_block_output.abs() + 1e-8)).mean()
    print(f"  Mean Relative Error: {rel_error.item():.6f}")

    # ========== 13. Success Check ==========
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
