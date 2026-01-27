"""
Hessian/Fisher Information 기반 블록별 민감도 분석 테스트

- 각 블록의 양자화 민감도 측정
- Fisher Information: gradient 기반 민감도
- Perturbation Sensitivity: 출력 변화 민감도
- Mixed-precision 결정에 활용
"""

import torch
import torch.nn as nn
import timm
from pathlib import Path
from datetime import datetime
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from models.quant_vit import QVit
from utils.imagenet_dataset import get_imagenet_loader
from utils.hessian_analyzer import HessianAnalyzer, analyze_block_sensitivity


def test_block_sensitivity(num_samples: int = 100, save_results: bool = True):
    """
    블록별 민감도 분석 테스트.

    Args:
        num_samples: Number of samples for analysis
        save_results: Whether to save results
    """
    print("=" * 70)
    print("Block Sensitivity Analysis (Fisher Information)")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # 1. Load ViT model
    print("\n[1] Loading ViT model...")
    model = timm.create_model('vit_base_patch16_224', pretrained=True)
    model.eval()
    model.to(device)

    # 2. Create QVit
    print("\n[2] Creating QVit...")
    config_dir = Path(__file__).parent / 'configs'

    qvit = QVit(
        model=model,
        config_dir=config_dir,
        num_blocks=None  # All 12 blocks
    )
    qvit.to(device)
    qvit.eval()

    # 3. Load calibration data
    print("\n[3] Loading calibration data...")
    imagenet_path = Path(__file__).parent / 'imagenet-mini' / 'train'

    if imagenet_path.exists():
        calib_loader = get_imagenet_loader(
            data_path=str(imagenet_path),
            batch_size=8,
            num_samples=num_samples,
            shuffle=True,
            num_workers=4,
            img_size=224
        )
        print(f"  Loaded {num_samples} samples from ImageNet")
    else:
        print("  ImageNet not found, using random data")
        # Create fake loader
        class FakeLoader:
            def __init__(self, num_batches, batch_size, device):
                self.num_batches = num_batches
                self.batch_size = batch_size
                self.device = device

            def __iter__(self):
                for _ in range(self.num_batches):
                    x = torch.randn(self.batch_size, 3, 224, 224)
                    y = torch.randint(0, 1000, (self.batch_size,))
                    yield x, y

            def __len__(self):
                return self.num_batches

        calib_loader = FakeLoader(num_samples // 8, 8, device)

    # 4. Run calibration first (필요)
    print("\n[4] Running calibration...")
    sample_count = 0
    for i, (images, _) in enumerate(calib_loader):
        images = images.to(device)
        with torch.no_grad():
            _ = qvit.calibration(images)
        sample_count += images.size(0)
        if (i + 1) % 10 == 0:
            print(f"  Calibrated {sample_count} samples")
        if sample_count >= num_samples:
            break

    qvit.compute_quant_params()

    # Reload data for sensitivity analysis
    if imagenet_path.exists():
        analysis_loader = get_imagenet_loader(
            data_path=str(imagenet_path),
            batch_size=8,
            num_samples=num_samples,
            shuffle=True,
            num_workers=4,
            img_size=224
        )
    else:
        analysis_loader = FakeLoader(num_samples // 8, 8, device)

    # 5. Fisher Information Analysis
    print("\n[5] Computing Fisher Information...")
    analyzer = HessianAnalyzer(qvit)

    fisher_results = analyzer.compute_block_fisher(
        analysis_loader,
        num_samples=min(num_samples, 50),  # Fisher는 적은 샘플로도 OK
        use_labels=False  # 모델 예측값 사용
    )

    print("\n[Fisher Information per Block]")
    print("-" * 50)
    for block_idx, fisher_val in fisher_results.items():
        bar = "█" * int(fisher_val * 100 / max(fisher_results.values()))
        print(f"  Block {block_idx:2d}: {fisher_val:10.6f} {bar}")
    print("-" * 50)

    # 6. Perturbation Sensitivity
    print("\n[6] Computing Perturbation Sensitivity...")

    # Reload data
    if imagenet_path.exists():
        pert_loader = get_imagenet_loader(
            data_path=str(imagenet_path),
            batch_size=8,
            num_samples=min(num_samples, 50),
            shuffle=True,
            num_workers=4,
            img_size=224
        )
    else:
        pert_loader = FakeLoader(min(num_samples, 50) // 8, 8, device)

    pert_results = analyzer.compute_output_sensitivity(
        pert_loader,
        num_samples=min(num_samples, 50)
    )

    print("\n[Perturbation Sensitivity per Block]")
    print("-" * 60)
    print(f"{'Block':<8} {'L2 Sensitivity':>18} {'Cosine Sensitivity':>20}")
    print("-" * 60)
    for block_idx in range(len(qvit.blocks)):
        l2_sens = pert_results[block_idx]['l2']
        cos_sens = pert_results[block_idx]['cosine']
        print(f"Block {block_idx:<3} {l2_sens:>18.6f} {cos_sens:>20.8f}")
    print("-" * 60)

    # 7. Compare with SQNR
    print("\n[7] Comparing with SQNR...")

    # Get test data
    if imagenet_path.exists():
        test_loader = get_imagenet_loader(
            data_path=str(imagenet_path),
            batch_size=8,
            num_samples=8,
            shuffle=False,
            num_workers=0,
            img_size=224
        )
        x_test, _ = next(iter(test_loader))
        x_test = x_test.to(device)
    else:
        x_test = torch.randn(8, 3, 224, 224).to(device)

    sqnr_indep = qvit.get_block_sqnr_summary(x_test, cumulative=False)

    # 8. Comprehensive Summary
    print("\n" + "=" * 80)
    print("Comprehensive Block Sensitivity Summary")
    print("=" * 80)
    print(f"{'Block':<8} {'Fisher':>12} {'L2 Sens':>12} {'SQNR (dB)':>12} {'Rank':>8}")
    print("-" * 80)

    # Normalize Fisher for ranking
    fisher_max = max(fisher_results.values())
    l2_max = max(v['l2'] for v in pert_results.values())

    # Compute composite sensitivity score
    composite_scores = {}
    for block_idx in range(len(qvit.blocks)):
        fisher_norm = fisher_results[block_idx] / fisher_max
        l2_norm = pert_results[block_idx]['l2'] / l2_max
        sqnr_inv = 1.0 / (sqnr_indep[block_idx] + 1)  # 낮은 SQNR = 높은 민감도

        # Weighted composite score
        composite = 0.4 * fisher_norm + 0.3 * l2_norm + 0.3 * sqnr_inv
        composite_scores[block_idx] = composite

    # Sort by composite score
    sorted_blocks = sorted(composite_scores.items(), key=lambda x: x[1], reverse=True)
    rank_map = {block_idx: rank + 1 for rank, (block_idx, _) in enumerate(sorted_blocks)}

    for block_idx in range(len(qvit.blocks)):
        fisher = fisher_results[block_idx]
        l2_sens = pert_results[block_idx]['l2']
        sqnr = sqnr_indep[block_idx]
        rank = rank_map[block_idx]

        # Sensitivity indicator
        if rank <= 3:
            indicator = "🔴 HIGH"
        elif rank <= 6:
            indicator = "🟡 MED"
        else:
            indicator = "🟢 LOW"

        print(f"Block {block_idx:<3} {fisher:>12.6f} {l2_sens:>12.6f} {sqnr:>12.2f} {rank:>5} {indicator}")

    print("=" * 80)

    # 9. Recommendations
    print("\n[Recommendations for Mixed-Precision]")
    print("-" * 50)

    high_sensitivity_blocks = [b for b, _ in sorted_blocks[:3]]
    med_sensitivity_blocks = [b for b, _ in sorted_blocks[3:6]]
    low_sensitivity_blocks = [b for b, _ in sorted_blocks[6:]]

    print(f"  High sensitivity (keep FP32 or 16-bit):")
    print(f"    Blocks: {high_sensitivity_blocks}")

    print(f"\n  Medium sensitivity (8-bit with careful calibration):")
    print(f"    Blocks: {med_sensitivity_blocks}")

    print(f"\n  Low sensitivity (8-bit or lower):")
    print(f"    Blocks: {low_sensitivity_blocks}")

    # 10. Save results
    if save_results:
        print("\n[10] Saving results...")
        log_dir = Path(__file__).parent / 'log' / 'hessian'
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save JSON summary
        summary = {
            'timestamp': timestamp,
            'num_samples': num_samples,
            'fisher': fisher_results,
            'perturbation': {str(k): v for k, v in pert_results.items()},
            'sqnr': sqnr_indep,
            'composite_scores': composite_scores,
            'rankings': rank_map,
            'recommendations': {
                'high_sensitivity': high_sensitivity_blocks,
                'medium_sensitivity': med_sensitivity_blocks,
                'low_sensitivity': low_sensitivity_blocks
            }
        }

        summary_file = log_dir / f"sensitivity_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"  Summary saved: {summary_file.name}")

        # Save visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        blocks = list(range(len(qvit.blocks)))

        # 1. Fisher Information
        ax1 = axes[0, 0]
        fisher_vals = [fisher_results[i] for i in blocks]
        colors = ['red' if rank_map[i] <= 3 else 'orange' if rank_map[i] <= 6 else 'green' for i in blocks]
        ax1.bar(blocks, fisher_vals, color=colors, edgecolor='black', alpha=0.8)
        ax1.set_xlabel('Block Index')
        ax1.set_ylabel('Fisher Information')
        ax1.set_title('Fisher Information per Block\n(Higher = More Sensitive)')
        ax1.set_xticks(blocks)
        ax1.grid(True, alpha=0.3)

        # 2. L2 Perturbation Sensitivity
        ax2 = axes[0, 1]
        l2_vals = [pert_results[i]['l2'] for i in blocks]
        ax2.bar(blocks, l2_vals, color=colors, edgecolor='black', alpha=0.8)
        ax2.set_xlabel('Block Index')
        ax2.set_ylabel('L2 Sensitivity')
        ax2.set_title('Perturbation Sensitivity per Block\n(Higher = More Sensitive)')
        ax2.set_xticks(blocks)
        ax2.grid(True, alpha=0.3)

        # 3. SQNR (Independent)
        ax3 = axes[1, 0]
        sqnr_vals = [sqnr_indep[i] for i in blocks]
        ax3.bar(blocks, sqnr_vals, color=colors, edgecolor='black', alpha=0.8)
        ax3.set_xlabel('Block Index')
        ax3.set_ylabel('SQNR (dB)')
        ax3.set_title('Independent SQNR per Block\n(Lower = More Sensitive)')
        ax3.set_xticks(blocks)
        ax3.axhline(y=20, color='red', linestyle='--', label='Good threshold (20 dB)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Composite Sensitivity Score
        ax4 = axes[1, 1]
        composite_vals = [composite_scores[i] for i in blocks]
        ax4.bar(blocks, composite_vals, color=colors, edgecolor='black', alpha=0.8)
        ax4.set_xlabel('Block Index')
        ax4.set_ylabel('Composite Score')
        ax4.set_title('Composite Sensitivity Score\n(Higher = More Sensitive)')
        ax4.set_xticks(blocks)
        ax4.grid(True, alpha=0.3)

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', edgecolor='black', label='High Sensitivity'),
            Patch(facecolor='orange', edgecolor='black', label='Medium Sensitivity'),
            Patch(facecolor='green', edgecolor='black', label='Low Sensitivity')
        ]
        fig.legend(handles=legend_elements, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.02))

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        plot_file = log_dir / f"sensitivity_plot_{timestamp}.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Plot saved: {plot_file.name}")

    print("\n" + "=" * 70)
    print("Sensitivity Analysis Completed!")
    print("=" * 70)

    return {
        'fisher': fisher_results,
        'perturbation': pert_results,
        'sqnr': sqnr_indep,
        'rankings': rank_map
    }


if __name__ == '__main__':
    results = test_block_sensitivity(num_samples=100, save_results=True)
