"""
QVit 테스트 코드
- timm ViT 모델 전체 양자화
- Calibration → Quantization → Inference 비교
- 블록별 SQNR 분석
- ImageNet 데이터셋 사용
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from pathlib import Path
import json

from models.quant_vit import QVit
from utils.imagenet_dataset import get_imagenet_loader, CustomImageNetDataset, get_imagenet_transforms
from utils.log_editor import save_qvit_profiler_results
from utils.evaluation import ModelEvaluator, evaluate_model, compare_models


def compute_sqnr(fp32_out, quant_out):
    """Compute SQNR between FP32 and quantized outputs"""
    signal_power = (fp32_out ** 2).mean()
    noise_power = ((fp32_out - quant_out) ** 2).mean()
    sqnr = 10 * torch.log10(signal_power / (noise_power + 1e-10))
    return sqnr.item()


def test_qvit(num_blocks: int = None, save_results: bool = True, use_real_data: bool = True):
    """
    Test QVit full model quantization.

    Args:
        num_blocks: Number of blocks to quantize (None = all 12 blocks)
        save_results: Whether to save profiler results
        use_real_data: Whether to use real ImageNet data for calibration
    """
    print("=" * 70)
    print("QVit Full Model Test")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # 1. Load ViT model
    print("\n[1] Loading ViT model...")
    model = timm.create_model('vit_base_patch16_224', pretrained=True)
    model.eval()
    model.to(device)

    print(f"  Model: vit_base_patch16_224")
    print(f"  Embed dim: {model.embed_dim}")
    print(f"  Num heads: {model.blocks[0].attn.num_heads}")
    print(f"  Num blocks: {len(model.blocks)}")
    print(f"  Num classes: {model.num_classes}")

    # 2. Create QVit
    print("\n[2] Creating QVit...")
    config_dir = Path(__file__).parent / 'configs'

    qvit = QVit(
        model=model,
        config_dir=config_dir,
        num_blocks=num_blocks
    )
    qvit.to(device)
    qvit.eval()

    # Print model summary
    qvit.print_model_summary()

    # 3. Calibration with Real ImageNet Data
    print("\n[3] Calibration...")

    if use_real_data:
        # Use real ImageNet-mini data
        imagenet_path = Path(__file__).parent / 'imagenet-mini' / 'train'
        if imagenet_path.exists():
            print(f"  Using real ImageNet data from: {imagenet_path}")
            calib_loader = get_imagenet_loader(
                data_path=str(imagenet_path),
                batch_size=16,
                num_samples=1000,  # 500 samples for calibration
                shuffle=True,
                num_workers=4,
                img_size=224
            )

            num_calib_batches = len(calib_loader)
            print(f"  Calibration batches: {num_calib_batches}")

            for i, (images, _) in enumerate(calib_loader):
                images = images.to(device)
                with torch.no_grad():
                    _ = qvit.calibration(images)
                if (i + 1) % 10 == 0 or i == num_calib_batches - 1:
                    print(f"  Batch {i+1}/{num_calib_batches}")
        else:
            print(f"  Warning: ImageNet path not found: {imagenet_path}")
            print(f"  Falling back to random data...")
            use_real_data = False

    if not use_real_data:
        # Fallback: use random data
        print("  Using random data for calibration")
        num_calib_batches = 50
        batch_size = 16

        for i in range(num_calib_batches):
            x = torch.randn(batch_size, 3, 224, 224).to(device)
            with torch.no_grad():
                _ = qvit.calibration(x)
            if (i + 1) % 5 == 0:
                print(f"  Batch {i+1}/{num_calib_batches}")

    # 4. Compute quantization parameters
    print("\n[4] Computing quantization parameters...")
    qvit.compute_quant_params()

    # 5. Compare FP32 vs Quantized
    print("\n[5] Comparing FP32 vs Quantized outputs...")

    # Test input - use real data if available
    if use_real_data and imagenet_path.exists():
        # Get a batch from validation set if exists, else from train
        val_path = Path(__file__).parent / 'imagenet-mini' / 'val'
        test_path = val_path if val_path.exists() else imagenet_path
        test_loader = get_imagenet_loader(
            data_path=str(test_path),
            batch_size=8,
            num_samples=8,
            shuffle=False,
            num_workers=0,
            img_size=224
        )
        x_test, labels = next(iter(test_loader))
        x_test = x_test.to(device)
        print(f"  Test input: real ImageNet images ({x_test.shape})")
    else:
        x_test = torch.randn(8, 3, 224, 224).to(device)
        print(f"  Test input: random data ({x_test.shape})")

    # FP32 original model
    with torch.no_grad():
        out_orig = model(x_test)

    # QVit FP32 mode
    qvit.set_mode('fp32')
    with torch.no_grad():
        out_qvit_fp32 = qvit.forward(x_test)

    # QVit Quantized mode
    qvit.set_mode('quantized')
    with torch.no_grad():
        out_qvit_quant = qvit.forward(x_test)

    # Compare
    print("\n  [Original vs QVit(FP32)]")
    diff_fp32 = (out_orig - out_qvit_fp32).abs()
    print(f"    Max diff: {diff_fp32.max():.8f}")
    print(f"    Mean diff: {diff_fp32.mean():.8f}")

    if diff_fp32.max() < 1e-4:
        print("    ✓ FP32 mode matches original model!")
    else:
        print("    ✗ Warning: FP32 mode differs from original")

    print("\n  [Original vs QVit(Quantized)]")
    diff_quant = (out_orig - out_qvit_quant).abs()
    print(f"    Max diff: {diff_quant.max():.4f}")
    print(f"    Mean diff: {diff_quant.mean():.4f}")

    # SQNR
    sqnr = compute_sqnr(out_orig, out_qvit_quant)
    print(f"    SQNR: {sqnr:.2f} dB")

    # Cosine similarity
    cos_sim = F.cosine_similarity(
        out_orig.flatten(),
        out_qvit_quant.flatten(),
        dim=0
    ).item()
    print(f"    Cosine Similarity: {cos_sim:.6f}")

    # Top-1 accuracy comparison (on test batch)
    _, pred_orig = out_orig.topk(1, dim=1)
    _, pred_quant = out_qvit_quant.topk(1, dim=1)
    match_rate = (pred_orig == pred_quant).float().mean().item()
    print(f"    Prediction Match Rate: {match_rate * 100:.1f}%")

    # 6. Classification Evaluation (Precision, Recall, F1)
    print("\n[6] Classification Evaluation...")

    # Evaluation 전 profiler 비활성화 (메모리 절약)
    # Forward 시 profiler가 텐서를 누적하여 메모리 폭발 방지
    original_profiling_state = qvit.enable_profiling
    qvit.set_profiling(False)
    print("  Profiling disabled for evaluation (memory optimization)")

    # Evaluation 데이터로더 준비
    if use_real_data and imagenet_path.exists():
        val_path = Path(__file__).parent / 'imagenet-mini' / 'val'
        eval_path = val_path if val_path.exists() else imagenet_path

        # 메모리 효율을 위해 작은 배치, 적은 샘플
        eval_loader = get_imagenet_loader(
            data_path=str(eval_path),
            batch_size=20,  # 줄임
            num_samples=10000,  # 줄임
            shuffle=False,
            num_workers=2,
            img_size=224
        )
        num_classes = model.num_classes

        # FP32 Original Model 평가
        print("\n  [Original FP32 Model Evaluation]")
        metrics_orig = evaluate_model(
            model=model,
            dataloader=eval_loader,
            device=device,
            num_classes=num_classes,
            verbose=True
        )

        # 메모리 확보: 캐시 클리어
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # QVit Quantized Model 평가
        print("\n  [QVit Quantized Model Evaluation]")
        qvit.set_mode('quantized')
        metrics_quant = evaluate_model(
            model=qvit,
            dataloader=eval_loader,
            device=device,
            num_classes=num_classes,
            verbose=True
        )

        # 모델 비교
        metrics_diff = compare_models(metrics_orig, metrics_quant, ('Original', 'Quantized'))
    else:
        print("  Skipping classification evaluation (no real data available)")
        metrics_orig = {}
        metrics_quant = {}
        metrics_diff = {}

    # Evaluation 후 profiler 상태 복원
    qvit.set_profiling(original_profiling_state)
    print("  Profiling state restored")

    # 7. Per-block SQNR analysis
    print("\n[7] Per-block SQNR analysis...")

    # Independent SQNR: each block's intrinsic quality
    print("\n  [Independent SQNR] - Each block's own quantization quality")
    print("-" * 50)
    block_sqnr_indep = qvit.get_block_sqnr_summary(x_test, cumulative=False)
    for block_idx, sqnr_val in block_sqnr_indep.items():
        status = "✓" if sqnr_val > 20 else "△" if sqnr_val > 10 else "✗"
        print(f"  Block {block_idx:2d}: {sqnr_val:6.2f} dB {status}")
    avg_indep = sum(block_sqnr_indep.values()) / len(block_sqnr_indep)
    print("-" * 50)
    print(f"  Average: {avg_indep:.2f} dB")

    # Cumulative SQNR: realistic E2E performance
    print("\n  [Cumulative SQNR] - Error accumulation through blocks")
    print("-" * 50)
    block_sqnr = qvit.get_block_sqnr_summary(x_test, cumulative=True)
    for block_idx, sqnr_val in block_sqnr.items():
        status = "✓" if sqnr_val > 20 else "△" if sqnr_val > 10 else "✗"
        print(f"  Block {block_idx:2d}: {sqnr_val:6.2f} dB {status}")
    avg_block_sqnr = sum(block_sqnr.values()) / len(block_sqnr)
    print("-" * 50)
    print(f"  Average: {avg_block_sqnr:.2f} dB")

    # 8. Save results
    if save_results:
        print("\n[8] Saving results...")

        # Save profiler results (CSV, histograms, summary)
        log_dir = Path(__file__).parent / 'log' / 'qvit'
        saved_files = save_qvit_profiler_results(
            qvit,
            base_log_dir=str(log_dir),
            prefix="qvit"
        )
        print(f"  Results saved to: {saved_files['directory']}")
        print(f"  - CSV: {Path(saved_files['csv']).name if saved_files['csv'] else 'N/A'}")
        print(f"  - Histograms: {len(saved_files['histograms'])} files")

        # Also save test summary to the same directory
        summary = {
            'model': 'vit_base_patch16_224',
            'num_blocks': qvit.num_blocks,
            'total_sqnr': sqnr,
            'cosine_similarity': cos_sim,
            'prediction_match_rate': match_rate,
            'block_sqnr_independent': block_sqnr_indep,
            'block_sqnr_cumulative': block_sqnr,
            'avg_block_sqnr_independent': avg_indep,
            'avg_block_sqnr_cumulative': avg_block_sqnr,
            'classification_metrics': {
                'original': metrics_orig,
                'quantized': metrics_quant,
                'diff': metrics_diff
            }
        }

        test_summary_file = Path(saved_files['directory']) / "test_summary.json"
        with open(test_summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"  - Test summary: {test_summary_file.name}")

    # 9. Quality assessment
    print("\n" + "=" * 70)
    print("Quality Assessment")
    print("=" * 70)

    if sqnr > 30:
        print(f"  ✓ EXCELLENT: SQNR {sqnr:.2f} dB (> 30 dB)")
    elif sqnr > 20:
        print(f"  ○ GOOD: SQNR {sqnr:.2f} dB (20-30 dB)")
    elif sqnr > 10:
        print(f"  △ ACCEPTABLE: SQNR {sqnr:.2f} dB (10-20 dB)")
    else:
        print(f"  ✗ POOR: SQNR {sqnr:.2f} dB (< 10 dB)")

    if cos_sim > 0.99:
        print(f"  ✓ Cosine Similarity: {cos_sim:.6f} (> 0.99)")
    elif cos_sim > 0.95:
        print(f"  ○ Cosine Similarity: {cos_sim:.6f} (0.95-0.99)")
    else:
        print(f"  △ Cosine Similarity: {cos_sim:.6f} (< 0.95)")

    # Classification metrics summary
    if metrics_quant:
        print(f"\n  Classification Metrics (Quantized):")
        print(f"    Accuracy:  {metrics_quant.get('accuracy', 0):.2f}%")
        print(f"    Precision: {metrics_quant.get('precision', 0):.4f}")
        print(f"    Recall:    {metrics_quant.get('recall', 0):.4f}")
        print(f"    F1 Score:  {metrics_quant.get('f1', 0):.4f}")

        if metrics_diff:
            acc_drop = metrics_diff.get('accuracy_diff', 0)
            if abs(acc_drop) < 1.0:
                print(f"  ✓ Accuracy drop: {acc_drop:+.2f}% (< 1%)")
            elif abs(acc_drop) < 3.0:
                print(f"  ○ Accuracy drop: {acc_drop:+.2f}% (1-3%)")
            else:
                print(f"  △ Accuracy drop: {acc_drop:+.2f}% (> 3%)")

    print("=" * 70)
    print("QVit Test Completed!")
    print("=" * 70)

    return qvit, sqnr


def test_qvit_inference_speed(qvit, device='cuda', num_iterations=100):
    """Test inference speed comparison."""
    print("\n" + "=" * 70)
    print("Inference Speed Test")
    print("=" * 70)

    x = torch.randn(1, 3, 224, 224).to(device)

    # Warm-up
    for _ in range(10):
        with torch.no_grad():
            _ = qvit.forward(x)

    # FP32 mode
    qvit.set_mode('fp32')
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(num_iterations):
        with torch.no_grad():
            _ = qvit.forward(x)
    end.record()
    torch.cuda.synchronize()
    fp32_time = start.elapsed_time(end) / num_iterations

    # Quantized mode
    qvit.set_mode('quantized')
    start.record()
    for _ in range(num_iterations):
        with torch.no_grad():
            _ = qvit.forward(x)
    end.record()
    torch.cuda.synchronize()
    quant_time = start.elapsed_time(end) / num_iterations

    print(f"  FP32 mode: {fp32_time:.2f} ms/iter")
    print(f"  Quantized mode: {quant_time:.2f} ms/iter")
    print(f"  Speedup: {fp32_time / quant_time:.2f}x")
    print("=" * 70)


if __name__ == '__main__':
    # Test with all 12 blocks
    qvit, sqnr = test_qvit(num_blocks=None, save_results=True)

    # Optional: speed test
    if torch.cuda.is_available():
        test_qvit_inference_speed(qvit)

    # Export for deployment
    print("\n" + "=" * 70)
    print("[Export] Preparing model for deployment...")
    print("=" * 70)

    export_dir = Path(__file__).parent / 'export'
    export_dir.mkdir(exist_ok=True)

    # 1. Strip unnecessary components (Observer, Profiler)
    qvit.strip_for_deploy()

    # 2. Save quantized weights
    weights_path = export_dir / 'qvit_weights.pt'
    qvit.save_quantized_weights(str(weights_path))

    # 3. Export to ONNX
    onnx_path = export_dir / 'qvit_int8.onnx'
    try:
        qvit.export_onnx(str(onnx_path))
    except Exception as e:
        print(f"  ONNX export failed: {e}")
        print("  (This is expected if onnx/onnxsim is not installed)")

    print("\n[Export] Complete!")
    print(f"  - Weights: {weights_path}")
    print(f"  - ONNX: {onnx_path}")
