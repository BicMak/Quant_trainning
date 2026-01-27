"""
QVitBlock 테스트 코드 with Profiler 저장 기능
- timm ViT 모델에서 Block 추출
- QVitBlock 모듈 테스트 (calibration, forward, quantized)
- 레이어별 SQNR 분석
- Profiler 결과 저장 (통계, 히스토그램)
"""

import torch
import torch.nn as nn
import timm
from pathlib import Path
from datetime import datetime
import json
import os

from models.quant_vit_block import QVitBlock


def compute_sqnr(fp32_out, quant_out):
    """Compute SQNR between FP32 and quantized outputs"""
    signal_power = (fp32_out ** 2).mean()
    noise_power = ((fp32_out - quant_out) ** 2).mean()
    sqnr = 10 * torch.log10(signal_power / (noise_power + 1e-10))
    return sqnr.item()


def save_profiler_results(qblock, save_dir: Path, prefix: str = "qvit_block"):
    """
    QVitBlock의 모든 profiler 결과를 저장

    Args:
        qblock: QVitBlock 인스턴스
        save_dir: 저장 디렉토리
        prefix: 파일 prefix

    Returns:
        dict: 저장된 파일 경로들
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_files = {
        'statistics': [],
        'histograms': [],
        'histogram_data': [],
        'summary': None
    }

    # 레이어별 profiler 가져오기
    profiler_list = qblock.get_profiler_list()

    print(f"\n[Saving Profiler Results]")
    print(f"  Save directory: {save_dir}")
    print(f"  Total profilers: {len(profiler_list)}")

    # 전체 요약 데이터
    summary_data = {
        'timestamp': timestamp,
        'total_layers': len(profiler_list),
        'layers': {}
    }

    for layer_name, prof in profiler_list:
        if prof is None:
            continue

        print(f"\n  Processing: {layer_name}")

        # 통계 데이터 가져오기
        stat_data = prof.get_statistic()
        hist_data = prof.get_hist()

        layer_summary = {'name': layer_name}

        # 통계 저장
        if stat_data is not None:
            stat_file = save_dir / f"{prefix}_{layer_name}_stats_{timestamp}.json"

            # tensor를 float로 변환
            stat_serializable = {}
            for k, v in stat_data.items():
                if isinstance(v, torch.Tensor):
                    stat_serializable[k] = v.item() if v.numel() == 1 else v.tolist()
                else:
                    stat_serializable[k] = v

            with open(stat_file, 'w') as f:
                json.dump(stat_serializable, f, indent=2)

            saved_files['statistics'].append(str(stat_file))
            layer_summary['statistics'] = stat_serializable
            print(f"    - Stats saved: {stat_file.name}")

            # SQNR 출력
            if 'qsnr' in stat_data:
                qsnr_val = stat_data['qsnr']
                if isinstance(qsnr_val, torch.Tensor):
                    qsnr_val = qsnr_val.item()
                print(f"    - QSNR: {qsnr_val:.2f} dB")

        # 히스토그램 데이터 저장 (JSON)
        if hist_data is not None:
            # 히스토그램 raw 데이터 저장
            hist_data_file = save_dir / f"{prefix}_{layer_name}_hist_data_{timestamp}.json"

            hist_serializable = {}
            for k, v in hist_data.items():
                if isinstance(v, torch.Tensor):
                    hist_serializable[k] = v.tolist()
                elif isinstance(v, tuple):
                    hist_serializable[k] = list(v)
                else:
                    hist_serializable[k] = v

            with open(hist_data_file, 'w') as f:
                json.dump(hist_serializable, f, indent=2)

            saved_files['histogram_data'].append(str(hist_data_file))
            layer_summary['histogram'] = {
                'kl_divergence': hist_data.get('kl_divergence'),
                'js_divergence': hist_data.get('js_divergence'),
                'original_range': hist_data.get('original_range'),
                'quantized_range': hist_data.get('quantized_range'),
            }
            print(f"    - Histogram data saved: {hist_data_file.name}")
            print(f"    - KL Divergence: {hist_data.get('kl_divergence', 'N/A'):.6f}")

            # 히스토그램 시각화 (PNG)
            try:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                import numpy as np

                # 데이터 추출
                orig_hist = hist_data['original_hist'].numpy()
                quant_hist = hist_data['quantized_hist'].numpy()
                orig_range = hist_data['original_range']
                quant_range = hist_data['quantized_range']

                # 공통 범위로 bins 생성
                min_val = min(orig_range[0], quant_range[0])
                max_val = max(orig_range[1], quant_range[1])
                bins = np.linspace(min_val, max_val, len(orig_hist) + 1)
                bin_width = (max_val - min_val) / len(orig_hist)
                bin_centers = (bins[:-1] + bins[1:]) / 2

                # Figure 생성 - 3개 subplot
                fig, axes = plt.subplots(1, 3, figsize=(18, 5))

                # 1. Original Distribution
                axes[0].bar(bin_centers, orig_hist, width=bin_width * 0.9,
                           alpha=0.8, color='blue', edgecolor='darkblue')
                axes[0].set_title(f'{layer_name}\nOriginal Distribution', fontsize=12)
                axes[0].set_xlabel('Value')
                axes[0].set_ylabel('Count')
                axes[0].axvline(x=orig_range[0], color='red', linestyle='--', alpha=0.7, label=f'min={orig_range[0]:.4f}')
                axes[0].axvline(x=orig_range[1], color='red', linestyle='--', alpha=0.7, label=f'max={orig_range[1]:.4f}')
                axes[0].legend(fontsize=8)
                axes[0].grid(True, alpha=0.3)

                # 2. Quantized Distribution
                axes[1].bar(bin_centers, quant_hist, width=bin_width * 0.9,
                           alpha=0.8, color='orange', edgecolor='darkorange')
                axes[1].set_title(f'{layer_name}\nQuantized Distribution', fontsize=12)
                axes[1].set_xlabel('Value')
                axes[1].set_ylabel('Count')
                axes[1].axvline(x=quant_range[0], color='red', linestyle='--', alpha=0.7, label=f'min={quant_range[0]:.4f}')
                axes[1].axvline(x=quant_range[1], color='red', linestyle='--', alpha=0.7, label=f'max={quant_range[1]:.4f}')
                axes[1].legend(fontsize=8)
                axes[1].grid(True, alpha=0.3)

                # 3. Overlay Comparison
                axes[2].bar(bin_centers, orig_hist, width=bin_width * 0.9,
                           alpha=0.5, color='blue', edgecolor='darkblue', label='Original')
                axes[2].bar(bin_centers, quant_hist, width=bin_width * 0.9,
                           alpha=0.5, color='orange', edgecolor='darkorange', label='Quantized')
                axes[2].set_title(f'{layer_name}\nOverlay Comparison\nKL={hist_data.get("kl_divergence", 0):.4f}, JS={hist_data.get("js_divergence", 0):.4f}', fontsize=12)
                axes[2].set_xlabel('Value')
                axes[2].set_ylabel('Count')
                axes[2].legend(fontsize=10)
                axes[2].grid(True, alpha=0.3)

                # SQNR 정보 추가
                if stat_data and 'qsnr' in stat_data:
                    qsnr_val = stat_data['qsnr']
                    if isinstance(qsnr_val, torch.Tensor):
                        qsnr_val = qsnr_val.item()
                    fig.suptitle(f'SQNR: {qsnr_val:.2f} dB', fontsize=14, fontweight='bold', y=1.02)

                plt.tight_layout()

                hist_file = save_dir / f"{prefix}_{layer_name}_hist_{timestamp}.png"
                plt.savefig(hist_file, dpi=150, bbox_inches='tight')
                plt.close()

                saved_files['histograms'].append(str(hist_file))
                print(f"    - Histogram image saved: {hist_file.name}")

            except ImportError:
                print(f"    - Histogram image skipped (matplotlib not available)")
            except Exception as e:
                print(f"    - Histogram image error: {e}")

        summary_data['layers'][layer_name] = layer_summary

    # 요약 파일 저장
    summary_file = save_dir / f"{prefix}_summary_{timestamp}.json"
    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2, default=str)

    saved_files['summary'] = str(summary_file)
    print(f"\n  Summary saved: {summary_file.name}")

    # 전체 SQNR 요약 테이블 출력
    print(f"\n" + "=" * 60)
    print("Layer-wise SQNR Summary")
    print("=" * 60)
    print(f"{'Layer':<30} {'SQNR (dB)':>12} {'KL Div':>12}")
    print("-" * 60)
    for layer_name, layer_info in summary_data['layers'].items():
        sqnr = layer_info.get('statistics', {}).get('qsnr', 'N/A')
        kl = layer_info.get('histogram', {}).get('kl_divergence', 'N/A')
        if isinstance(sqnr, (int, float)):
            sqnr_str = f"{sqnr:.2f}"
        else:
            sqnr_str = str(sqnr)
        if isinstance(kl, (int, float)):
            kl_str = f"{kl:.6f}"
        else:
            kl_str = str(kl)
        print(f"{layer_name:<30} {sqnr_str:>12} {kl_str:>12}")
    print("=" * 60)

    return saved_files


def test_qvit_block(save_profiler: bool = True):
    print("=" * 70)
    print("QVitBlock Test with Profiler")
    print("=" * 70)

    # Device 설정
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # 1. ViT 모델 로드
    print("\n[1] Loading ViT model...")
    model = timm.create_model('vit_base_patch16_224', pretrained=True)
    model.eval()
    model.to(device)

    # Block 0의 구조 확인
    block = model.blocks[0]
    print(f"\n[2] Block 0 structure:")
    print(f"  - norm1: {block.norm1}")
    print(f"  - attn.qkv: {block.attn.qkv}")
    print(f"  - attn.proj: {block.attn.proj}")
    print(f"  - norm2: {block.norm2}")
    print(f"  - mlp.fc1: {block.mlp.fc1}")
    print(f"  - mlp.fc2: {block.mlp.fc2}")

    # 2. QVitBlock 생성
    print("\n[3] Creating QVitBlock...")
    config_dir = Path(__file__).parent / 'configs'
    attn_config = config_dir / 'attn_config.yaml'
    mlp_config = config_dir / 'mlp_config.yaml'
    block_config = config_dir / 'block_config.yaml'

    # Config 파일 존재 확인
    if not attn_config.exists():
        print(f"  ERROR: {attn_config} not found!")
        return
    if not mlp_config.exists():
        print(f"  ERROR: {mlp_config} not found!")
        return

    # block_config가 없으면 None으로 설정 (기본값 사용)
    if not block_config.exists():
        print(f"  Warning: {block_config} not found, using defaults")
        block_config = None

    qblock = QVitBlock(
        block=block,
        attn_config_path=attn_config,
        mlp_config_path=mlp_config,
        block_config_path=block_config
    )
    qblock.to(device)
    print(f"  QVitBlock created successfully!")

    # 레이어 요약 출력
    qblock.print_layer_summary()

    # Profiler 목록 확인
    print("\n[4] Available profilers:")
    profiler_names = qblock.get_profiler_names()
    for name in profiler_names:
        print(f"  - {name}")

    # 3. 테스트 입력 생성
    batch_size = 4
    num_tokens = 197
    embed_dim = 768

    print(f"\n[5] Test input shape: ({batch_size}, {num_tokens}, {embed_dim})")

    # 4. Calibration 테스트
    print("\n[6] Calibration test...")
    num_calib_batches = 10

    for i in range(num_calib_batches):
        x = torch.randn(batch_size, num_tokens, embed_dim).to(device)
        with torch.no_grad():
            out = qblock.calibration(x)
        if (i + 1) % 5 == 0:
            print(f"  Batch {i+1}/{num_calib_batches}: input {x.shape} → output {out.shape}")

    # 5. Quantization parameters 계산
    print("\n[7] Computing quantization parameters...")
    qblock.compute_quant_params()
    print("  Quantization parameters computed!")

    # 6. FP32 vs Quantized 비교 (profiler에 데이터 축적)
    print("\n[8] Comparing FP32 vs Quantized outputs (accumulating profiler data)...")

    num_test_batches = 20
    total_sqnr_list = []

    for i in range(num_test_batches):
        x_test = torch.randn(batch_size, num_tokens, embed_dim).to(device)

        # FP32 모드
        qblock.set_mode('fp32')
        with torch.no_grad():
            out_fp32 = qblock.forward(x_test)

        # Quantized 모드
        qblock.set_mode('quantized')
        with torch.no_grad():
            out_quant = qblock.forward(x_test)

        sqnr = compute_sqnr(out_fp32, out_quant)
        total_sqnr_list.append(sqnr)

        if (i + 1) % 10 == 0:
            print(f"  Batch {i+1}/{num_test_batches}: SQNR = {sqnr:.2f} dB")

    avg_sqnr = sum(total_sqnr_list) / len(total_sqnr_list)
    print(f"\n  Average SQNR over {num_test_batches} batches: {avg_sqnr:.2f} dB")

    # 7. Original Block과 비교
    print("\n[9] Comparing with original Block...")
    x_test = torch.randn(batch_size, num_tokens, embed_dim).to(device)

    with torch.no_grad():
        orig_out = block(x_test)

    qblock.set_mode('fp32')
    with torch.no_grad():
        qblock_fp32_out = qblock.forward(x_test)

    diff_orig = (orig_out - qblock_fp32_out).abs()
    print(f"\n  Original vs QVitBlock(FP32):")
    print(f"    Max diff: {diff_orig.max():.10f}")
    print(f"    Mean diff: {diff_orig.mean():.10f}")

    if diff_orig.max() < 1e-4:
        print("    ✓ FP32 mode matches original Block!")
    else:
        print("    ✗ Warning: FP32 mode differs from original Block")

    # 8. Profiler 결과 저장
    if save_profiler:
        print("\n" + "=" * 70)
        print("Saving Profiler Results")
        print("=" * 70)

        log_dir = Path(__file__).parent / 'log' / 'qvit_block'
        saved_files = save_profiler_results(qblock, log_dir, prefix="block0")

        print(f"\n[Saved Files Summary]")
        print(f"  Statistics files: {len(saved_files['statistics'])}")
        print(f"  Histogram files: {len(saved_files['histograms'])}")
        print(f"  Summary file: {saved_files['summary']}")

    # 9. 품질 평가
    print("\n" + "=" * 70)
    print("Quality Assessment")
    print("=" * 70)

    if avg_sqnr > 30:
        print(f"  ✓ EXCELLENT: SQNR {avg_sqnr:.2f} dB (> 30 dB)")
    elif avg_sqnr > 20:
        print(f"  ○ GOOD: SQNR {avg_sqnr:.2f} dB (20-30 dB)")
    elif avg_sqnr > 10:
        print(f"  △ ACCEPTABLE: SQNR {avg_sqnr:.2f} dB (10-20 dB)")
    else:
        print(f"  ✗ POOR: SQNR {avg_sqnr:.2f} dB (< 10 dB)")

    print("\n" + "=" * 70)
    print("QVitBlock Test Completed!")
    print("=" * 70)

    return qblock, avg_sqnr


if __name__ == '__main__':
    qblock, sqnr = test_qvit_block(save_profiler=True)
