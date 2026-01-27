# Per-Layer Quantization Configuration Update

## Summary

Updated the quantization configuration system to support per-layer configurations for the attention block, allowing fine-grained control over each layer's quantization settings.

## Changes Made

### 1. [configs/attn_config.yaml](configs/attn_config.yaml)

**Before (Global format):**
```yaml
outputs:
  calibration_mode: channel_wise
  bit_type: ...

weight:
  calibration_mode: channel_wise
  ...

activation:
  ...
```

**After (Per-layer format):**
```yaml
# Linear layers (have both weight and output configs)
attn_qkv:
  weight:
    calibration_mode: channel_wise
    bit_type:
      bits: 8
      signed: True
    observer_type: MinmaxObserver
    enable_profiler: True
  output:
    calibration_mode: channel_wise
    bit_type:
      bits: 8
      signed: False
    observer_type: MinmaxObserver
    enable_profiler: True
    output_quant_enable: False

attn_proj:
  weight: ...
  output: ...

# QAct layers (single config each)
attn_q_out:
  calibration_mode: channel_wise
  bit_type:
    bits: 8
    signed: True  # Changed from False to handle negative values
  observer_type: MinmaxObserver
  enable_profiler: True

attn_k_out:
  ...

attn_v_input:
  ...

attn_kv_out:
  ...

attn_v_out:
  ...

# IntSoftmax
intSoft:
  ...
```

### 2. [utils/config_loader.py](utils/config_loader.py)

**Key Changes:**
- Added automatic format detection (global vs per-layer)
- Supports both formats for backward compatibility
- Per-layer configs are returned as:
  - `dict` with `'weight'` and `'output'` keys for QuantLinear layers
  - `QuantConfig` object for QAct/IntSoft layers

**Example Usage:**
```python
configs = load_config_from_yaml('configs/attn_config.yaml')

# Per-layer format:
attn_qkv_weight_config = configs['attn_qkv']['weight']
attn_qkv_output_config = configs['attn_qkv']['output']
attn_q_out_config = configs['attn_q_out']

# Global format (legacy, still supported):
weight_config = configs['weight']
output_config = configs['output']
```

### 3. [models/quant_attn.py](models/quant_attn.py)

**Key Changes:**
- Added `get_layer_config()` helper function to handle both formats
- Updated all layer initializations to use per-layer configs
- Backward compatible with global format

**Before:**
```python
self.attn_qkv = QuantLinear(
    input_module=block.attn.qkv,
    out_config=configs['output'],
    weight_config=configs['weight'],
    layer_name='attn_qkv'
)

self.attn_q_out = QAct(
    quant_config=configs.get('activation', configs['output']),
    act_module=None,
    layer_name='attn_q_out'
)
```

**After:**
```python
def get_layer_config(layer_name):
    if layer_name in configs:
        return configs[layer_name]
    # Fallback to global configs
    if layer_name in ['attn_qkv', 'attn_proj']:
        return {'weight': configs.get('weight'), 'output': configs.get('output')}
    else:
        return configs.get('activation', configs.get('output'))

qkv_config = get_layer_config('attn_qkv')
self.attn_qkv = QuantLinear(
    input_module=block.attn.qkv,
    out_config=qkv_config['output'] if isinstance(qkv_config, dict) else configs['output'],
    weight_config=qkv_config['weight'] if isinstance(qkv_config, dict) else configs['weight'],
    layer_name='attn_qkv'
)

self.attn_q_out = QAct(
    quant_config=get_layer_config('attn_q_out'),
    act_module=None,
    layer_name='attn_q_out'
)
```

## Layer Configurations

The attention block now has 8 separately configurable layers:

### QuantLinear Layers (Weight + Output configs):
1. **attn_qkv**: QKV projection layer
   - Weight: 8-bit signed (int8), channel_wise, MinmaxObserver
   - Output: 8-bit unsigned (uint8), channel_wise, MinmaxObserver, **disabled** (output_quant_enable: False)

2. **attn_proj**: Output projection layer
   - Weight: 8-bit signed (int8), channel_wise, MinmaxObserver
   - Output: 8-bit unsigned (uint8), channel_wise, MinmaxObserver, **disabled** (output_quant_enable: False)

### QAct Layers (Single config):
3. **attn_q_out**: Q output quantization after QKV split
   - 8-bit **signed** (int8), channel_wise, MinmaxObserver

4. **attn_k_out**: K output quantization after QKV split
   - 8-bit **signed** (int8), channel_wise, MinmaxObserver

5. **attn_v_input**: V input quantization after QKV split
   - 8-bit **signed** (int8), channel_wise, MinmaxObserver

6. **attn_kv_out**: Attention score quantization (Q @ K^T output)
   - 8-bit **signed** (int8), channel_wise, MinmaxObserver

7. **attn_v_out**: Attention @ V output quantization
   - 8-bit **signed** (int8), channel_wise, MinmaxObserver

### IntSoftmax Layer:
8. **intSoft**: Integer softmax approximation
   - 8-bit unsigned (uint8), channel_wise, MinmaxObserver

## Key Improvements

1. **Individual Layer Control**: Each layer can now have different:
   - Bit width (bits)
   - Signed vs unsigned quantization
   - Observer type (MinmaxObserver, PercentileObserver, KLObserver)
   - Calibration mode (channel_wise, layer_wise)
   - Percentile parameters

2. **Fixed Signed/Unsigned Issues**:
   - Q, K, V now use **signed** quantization (was unsigned)
   - This properly handles negative values in attention computation

3. **Backward Compatibility**:
   - Code still supports global format configs
   - Automatic format detection

4. **Better Debugging**:
   - Each layer's config is explicit and visible in the YAML
   - Easier to experiment with different quantization strategies per layer

## Testing

To test the new configuration:

```bash
python test_quant_attn.py
# or
python test_embed_to_attn.py
```

The config loader will automatically detect the per-layer format and use it.

## Migration Guide

**If you have existing global format configs:**

No changes needed! The config loader automatically detects the format and handles both.

**If you want to use per-layer configs:**

1. Create a new YAML file with per-layer structure (see [configs/attn_config.yaml](configs/attn_config.yaml))
2. Define configs for each layer: `attn_qkv`, `attn_proj`, `attn_q_out`, `attn_k_out`, `attn_v_input`, `attn_kv_out`, `attn_v_out`, `intSoft`
3. Use the same `load_config_from_yaml()` function - it automatically detects the format

## Next Steps

You can now experiment with different quantization settings for each layer to improve SQNR:

- Try different observers (MinmaxObserver, PercentileObserver, KLObserver) for attn_kv_out
- Adjust bit widths per layer if needed
- Fine-tune percentile parameters for each layer independently
