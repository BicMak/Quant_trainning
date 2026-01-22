# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Post-Training Quantization (PTQ)** framework for Vision Transformers (ViT) from the `timm` library. The goal is to create fully quantized ViT blocks capable of integer-only inference on specialized hardware.

**Key techniques:**
- **PTF (Power-of-Two Factor) LayerNorm**: Channel-wise 2^α scaling enables bit-shift operations instead of division
- **LIS (Log-Int-Softmax)**: Log-domain quantization for softmax handles extreme value distributions
- **LUT-based activations**: GELU and other non-linear functions via lookup tables

## Commands

```bash
# Run tests
python test_config_loader.py      # Config YAML loading
python test_vit_block.py          # ViT block quantization end-to-end
python -m models.vit_block        # ViT block module test
python -m models.ptq.quant_linear # QuantLinear layer test

# Run main quantization examples
python quant_cnn.py               # CNN quantization example
```

## Architecture

### Data Flow Through Quantized ViT Block

```
Input (INT8) → norm1 (QLayerNorm/PTF) → INT8
  → attn_qkv (QuantLinear) → INT32
  → attn_qkv_output (QAct) → INT8
  → Q @ K^T → INT32 → kv_act (QAct) → INT8
  → QuantIntSoft (LIS) → Log-INT8
  → Attn @ V → INT32 → residual1 (QAct) → INT8
  → + residual → norm2 → mlp_fc1 → mlp_act (GELU LUT)
  → mlp_fc2 → mlp_act2 → residual2 → Output (INT8)
```

### Key Components

| Directory | Purpose |
|-----------|---------|
| `models/vit_block.py` | Main `QuantTimmVitBlock` - orchestrates full block quantization |
| `models/ptq/` | Quantized layer implementations |
| `models/ptq/layer_observer/` | Calibration observers (MinMax, Percentile, OMSE, KL) |
| `models/ptq/layer_quantizer/` | Quantization schemes (Uniform, Log2) |
| `models/ptq/layer_profiler/` | Profiling tools (stats, histograms, timing, memory) |
| `configs/` | YAML configuration files |
| `utils/config_loader.py` | YAML → LayerQuantConfig parser |

### Configuration System

**Priority**: Layer-specific config > Default config

```yaml
# configs/quant_config_int8.yaml structure:
default:           # Fallback for all layers
layers:            # Weight quantization (attn_qkv, attn_proj, mlp_fc1, mlp_fc2, norm1, norm2)
attn_layer:        # Activation quantization (attn_qkv_output, kv_act, sv_attn, mlp_act, mlp_act2, intSoft)
residual_layer:    # Residual connections (residual1, residual2)
```

### Observer Types

- **MinmaxObserver**: Simple min/max range, sensitive to outliers
- **PercentileObserver**: Clips outliers via percentile threshold (default, use `percentile_alpha`)
- **OmseObserver**: Grid search minimizing MSE (accurate but slower)
- **KLObserver**: Minimizes KL divergence between original and quantized distributions

### Usage Pattern

```python
from models.vit_block import QuantTimmVitBlock

# 1. Create quantized block (YAML config recommended)
quant_block = QuantTimmVitBlock(
    block=original_block,
    quant_config='configs/quant_config_int8.yaml',
    enable_profiling=True
)

# 2. Calibration phase
quant_block.calibration(calibration_data)

# 3. Compute quantization parameters
quant_block.compute_quant_params()

# 4. Inference
quant_block.mode = 'quantized'  # or 'fp32'
output = quant_block(input_tensor)
```

### Two-Phase Quantization

1. **Calibration**: `layer.calibration(x)` collects activation statistics
2. **Parameter computation**: `layer.compute_quant_params()` finalizes scale/zero_point

### Mode Switching

All quantized layers support:
- `layer.mode = 'fp32'`: Full precision (training/debugging)
- `layer.mode = 'quantized'`: Fake quantization (simulates integer inference)

## Important Constraints

- **YAML layer names must match** `QuantTimmVitBlock` attribute names exactly
- **QuantIntSoft requires input_scale** propagated from the previous MatMul layer
- **PTF LayerNorm**: Computation stays FP32, only output is quantized
- **Re-quantization points**: Required after each Linear and residual add to prevent bit-width explosion
- This implements **fake quantization** (FP32 compute simulating INT8), not actual integer kernels