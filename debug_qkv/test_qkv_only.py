"""
Simple debugging script to test only QKV quantization layers.

Tests 4 layers:
1. attn_qkv (QuantLinear - weight quantization)
2. attn_q_out (QAct - activation quantization)
3. attn_k_out (QAct - activation quantization)
4. attn_v_input (QAct - activation quantization)
"""

import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import timm
from models.ptq.quant_linear import QuantLinear
from models.ptq.quant_act import QAct
from utils.config_loader import load_config_from_yaml


def print_stats_table(title, stats):
    """Print comparison table for statistics"""
    if not stats:
        print(f"  ✗ {title}: No statistics available")
        return

    print(f"\n  [ {title} ]")
    print(f"  {'Metric':<15} | {'Original (FP32)':>20} | {'Quantized (INT8)':>20}")
    print(f"  {'-'*15}-+-{'-'*20}-+-{'-'*20}")

    def fmt(key, is_pct=False):
        v = stats.get(key, 'N/A')
        if not isinstance(v, (int, float)):
            return f"{'N/A':>20}"
        if is_pct:
            return f"{v*100:>19.4f}%"
        return f"{v:>20.6f}"

    # Basic statistics
    metrics = [
        ("Min", "original_min", "min", False),
        ("Max", "original_max", "max", False),
        ("Mean", "original_mean", "mean", False),
        ("Std", "original_std", "std", False),
        ("Outlier %", "original_outlier_ratio", "outlier_ratio", True)
    ]
    for label, ok, qk, pct in metrics:
        print(f"  {label:<15} | {fmt(ok, pct)} | {fmt(qk, pct)}")

    # Quality metrics
    print(f"  {'-'*59}")
    qsnr = stats.get('qsnr', 'N/A')
    cos = stats.get('cosine_sim', 'N/A')
    mse = stats.get('mse', 'N/A')

    qsnr_str = f"{qsnr:>12.2f} dB" if isinstance(qsnr, (int, float)) else "N/A"
    cos_str = f"{cos:>12.6f}" if isinstance(cos, (int, float)) else "N/A"
    mse_str = f"{mse:>12.6e}" if isinstance(mse, (int, float)) else "N/A"

    print(f"  > QSNR: {qsnr_str}  |  Cosine: {cos_str}  |  MSE: {mse_str}")


def test_qkv_quantization():
    """Test QKV quantization layers only"""
    print("="*80)
    print("QKV Quantization Debug Test")
    print("="*80)

    # Config
    config_path = project_root / 'configs' / 'attn_config.yaml'
    batch_size = 4
    seq_len = 197
    dim = 768
    num_heads = 12
    head_dim = 64
    num_batches = 10

    print(f"\n[1] Loading configuration")
    configs = load_config_from_yaml(str(config_path))
    print(f"  ✓ Config loaded from {config_path}")

    # Load pretrained ViT model
    print(f"\n[2] Loading pretrained ViT model")
    model = timm.create_model('vit_base_patch16_224', pretrained=True)
    model.eval()
    block = model.blocks[0]
    print(f"  ✓ Model loaded: vit_base_patch16_224")
    print(f"  ✓ Using first attention block")

    # Create QuantLinear for QKV
    print(f"\n[3] Creating QuantLinear layer (attn_qkv)")
    qkv_config = configs['attn_qkv']
    qkv_layer = QuantLinear(
        input_module=block.attn.qkv,
        out_config=qkv_config['output'],
        weight_config=qkv_config['weight'],
        layer_name='attn_qkv'
    )
    print(f"  ✓ QuantLinear created")
    print(f"  Weight config: {qkv_config['weight'].observer_type}, {qkv_config['weight'].calibration_mode}")
    print(f"  Output config: output_quant_enable={qkv_config['output'].output_quant_enable}")

    # Create QAct layers for Q, K, V
    print(f"\n[4] Creating QAct layers (Q, K, V)")
    q_layer = QAct(
        quant_config=configs['attn_q_out'],
        act_module=None,
        layer_name='attn_q_out'
    )
    k_layer = QAct(
        quant_config=configs['attn_k_out'],
        act_module=None,
        layer_name='attn_k_out'
    )
    v_layer = QAct(
        quant_config=configs['attn_v_input'],
        act_module=None,
        layer_name='attn_v_input'
    )
    print(f"  ✓ Q layer: {configs['attn_q_out'].observer_type}, {configs['attn_q_out'].calibration_mode}")
    print(f"  ✓ K layer: {configs['attn_k_out'].observer_type}, {configs['attn_k_out'].calibration_mode}")
    print(f"  ✓ V layer: {configs['attn_v_input'].observer_type}, {configs['attn_v_input'].calibration_mode}")

    # Generate calibration data
    print(f"\n[5] Generating calibration data")
    calib_data = [torch.randn(batch_size, seq_len, dim) for _ in range(num_batches)]
    print(f"  ✓ Generated {num_batches} batches of shape [{batch_size}, {seq_len}, {dim}]")

    # Calibration phase
    print(f"\n[6] Running calibration")
    with torch.no_grad():
        for idx, x in enumerate(calib_data):
            # QKV linear forward
            qkv = qkv_layer.calibration(x)  # [B, N, 3*dim]

            # Split Q, K, V
            B, N, _ = qkv.shape
            qkv_reshaped = qkv.reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv_reshaped.unbind(0)  # Each: [B, num_heads, N, head_dim]

            # Calibrate Q, K, V
            q = q_layer.calibration(q)
            k = k_layer.calibration(k)
            v = v_layer.calibration(v)

            if (idx + 1) % 5 == 0:
                print(f"  Batch {idx + 1}/{num_batches} calibrated")

    print(f"  ✓ Calibration completed")

    # Compute quantization parameters
    print(f"\n[7] Computing quantization parameters")
    qkv_layer.compute_output_quant_params()
    q_layer.compute_quant_params()
    k_layer.compute_quant_params()
    v_layer.compute_quant_params()
    print(f"  ✓ Quantization parameters computed")

    # Print scale information
    print(f"\n[8] Quantization scale information")
    print(f"  Q scale shape: {q_layer.scaler.shape if hasattr(q_layer, 'scaler') else 'None'}")
    print(f"  K scale shape: {k_layer.scaler.shape if hasattr(k_layer, 'scaler') else 'None'}")
    print(f"  V scale shape: {v_layer.scaler.shape if hasattr(v_layer, 'scaler') else 'None'}")

    if hasattr(q_layer, 'scaler') and q_layer.scaler is not None:
        print(f"  Q scale range: [{q_layer.scaler.min():.6f}, {q_layer.scaler.max():.6f}]")
        print(f"  K scale range: [{k_layer.scaler.min():.6f}, {k_layer.scaler.max():.6f}]")
        print(f"  V scale range: [{v_layer.scaler.min():.6f}, {v_layer.scaler.max():.6f}]")

    # Quantized forward pass
    print(f"\n[9] Running quantized forward passes")
    qkv_layer.mode = 'quantized'
    q_layer.mode = 'quantized'
    k_layer.mode = 'quantized'
    v_layer.mode = 'quantized'

    with torch.no_grad():
        for idx, x in enumerate(calib_data):
            # QKV linear forward
            qkv = qkv_layer(x)

            # Split Q, K, V
            B, N, _ = qkv.shape
            qkv_reshaped = qkv.reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv_reshaped.unbind(0)

            # Quantize Q, K, V
            q = q_layer(q)
            k = k_layer(k)
            v = v_layer(v)

            if (idx + 1) % 5 == 0:
                print(f"  Batch {idx + 1}/{num_batches} processed")

    print(f"  ✓ Quantized forward completed")

    # Get profiling results
    print(f"\n[10] Profiling Results")
    print("="*80)

    # Weight profiling
    qkv_results = qkv_layer.get_profiling_results()
    if qkv_results and 'weight' in qkv_results:
        print_stats_table("attn_qkv (Weight)", qkv_results['weight']['statistics'])

    # Activation profiling
    q_results = q_layer.get_profiling_results()
    if q_results and q_results.get('statistics'):
        print_stats_table("attn_q_out (Activation)", q_results['statistics'])

    k_results = k_layer.get_profiling_results()
    if k_results and k_results.get('statistics'):
        print_stats_table("attn_k_out (Activation)", k_results['statistics'])

    v_results = v_layer.get_profiling_results()
    if v_results and v_results.get('statistics'):
        print_stats_table("attn_v_input (Activation)", v_results['statistics'])

    # Summary
    print(f"\n" + "="*80)
    print("Test Summary")
    print("="*80)

    # Collect QSNR values
    qsnr_values = {}
    if qkv_results and 'weight' in qkv_results and qkv_results['weight']['statistics']:
        qsnr_values['qkv_weight'] = qkv_results['weight']['statistics'].get('qsnr', 'N/A')
    if q_results and q_results.get('statistics'):
        qsnr_values['q_out'] = q_results['statistics'].get('qsnr', 'N/A')
    if k_results and k_results.get('statistics'):
        qsnr_values['k_out'] = k_results['statistics'].get('qsnr', 'N/A')
    if v_results and v_results.get('statistics'):
        qsnr_values['v_input'] = v_results['statistics'].get('qsnr', 'N/A')

    print(f"\nQSNR Summary:")
    for name, qsnr in qsnr_values.items():
        if isinstance(qsnr, (int, float)):
            status = "✓" if qsnr >= 20 else "✗"
            print(f"  {status} {name:15s}: {qsnr:6.2f} dB")
        else:
            print(f"  ? {name:15s}: {qsnr}")

    print(f"\n" + "="*80)
    print("Debug Test COMPLETED")
    print("="*80)


if __name__ == "__main__":
    test_qkv_quantization()