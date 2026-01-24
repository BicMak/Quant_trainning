# Quantization Configuration Usage Guide

## Quick Start

### 1. Default Usage (Recommended)
The simplest way to use the quantization framework with split configuration files:

```python
from models.vit_block import QuantTimmVitBlock
import timm

# Load model
model = timm.create_model('vit_base_patch16_224', pretrained=False)
original_block = model.blocks[0]

# Create quantized block - automatically uses configs/ directory
quant_block = QuantTimmVitBlock(
    block=original_block,
    quant_config='configs',  # Uses split configs by default
    enable_profiling=True
)
```

### 2. Run Test Script
```bash
# Default: Uses split configs from configs/ directory
python test_vit_block.py

# Or specify a single file
python test_vit_block.py --config configs/quant_config_int8.yaml
```

## Configuration File Organization

### Split Configuration (Recommended)
```
configs/
├── weight_config.yaml        # Linear & LayerNorm weights
├── activation_config.yaml    # Activation quantization
└── residual_config.yaml      # Residual connections
```

**Benefits:**
- Clear separation of concerns
- Easy to modify specific layer types
- Better for version control
- Modular and maintainable

### Single Configuration (Legacy)
```
configs/
└── quant_config_int8.yaml   # All configurations in one file
```

**Use when:**
- Quick prototyping
- Simple configurations
- Backward compatibility needed

## Configuration Loading Methods

### Method 1: Automatic Detection
The `load_config_from_yaml()` function automatically detects file vs directory:

```python
from utils.config_loader import load_config_from_yaml

# Pass directory → loads 3 split files
config = load_config_from_yaml('configs')

# Pass file → loads single file
config = load_config_from_yaml('configs/quant_config_int8.yaml')
```

### Method 2: Explicit Multi-File Loading
```python
from utils.config_loader import load_multi_config_from_yaml

# Option A: Use directory
config = load_multi_config_from_yaml(config_dir='configs')

# Option B: Specify individual files
config = load_multi_config_from_yaml(
    weight_config_path='configs/weight_config.yaml',
    activation_config_path='configs/activation_config.yaml',
    residual_config_path='configs/residual_config.yaml'
)
```

## Layer Configuration Details

### Weight Layers (weight_config.yaml)
- `attn_qkv`: Attention QKV projection (8-bit, PercentileObserver)
- `attn_proj`: Attention output projection (8-bit, PercentileObserver)
- `mlp_fc1`: MLP first layer (8-bit, PercentileObserver)
- `mlp_fc2`: MLP second layer (8-bit, PercentileObserver)
- `norm1`: LayerNorm 1 (8-bit, PercentileObserver)
- `norm2`: LayerNorm 2 (8-bit, PercentileObserver)

### Activation Layers (activation_config.yaml)
- `attn_qkv_output`: QKV output (8-bit, OmseObserver)
- `kv_act`: Key-Value activation (16-bit, OmseObserver) ⚠️
- `sv_attn`: Softmax-Value attention (16-bit, OmseObserver) ⚠️
- `mlp_act`: MLP first activation (16-bit, OmseObserver) ⚠️
- `mlp_act2`: MLP second activation (16-bit, OmseObserver) ⚠️
- `intSoft`: IntSoftmax (16-bit, PercentileObserver) ⚠️

⚠️ **Critical 16-bit layers**: These layers use higher precision due to their sensitivity to quantization error accumulation.

### Residual Layers (residual_config.yaml)
- `residual1`: First residual connection (8-bit, PercentileObserver, layer_wise)
- `residual2`: Second residual connection (8-bit, PercentileObserver, layer_wise)

## Modifying Configurations

### Example: Change a single layer's bit-width
Edit `configs/activation_config.yaml`:

```yaml
attn_layer:
  kv_act:
    calibration_mode: channel_wise
    bit_type:
      bits: 8  # Changed from 16 to 8
      signed: true
      name: int8
    observer_type: OmseObserver
    quantization_method: Uniform
    percentile_alpha: 0.9999
    percentile_sigma: 0.1
```

### Example: Change observer type for all weights
Edit `configs/weight_config.yaml` default section:

```yaml
default:
  calibration_mode: channel_wise
  bit_type:
    bits: 8
    signed: true
    name: int8
  observer_type: MinmaxObserver  # Changed from PercentileObserver
  quantization_method: Uniform
  percentile_alpha: 0.999
  percentile_sigma: 0.01
  kl_bins: 2048
```

## Test and Validation

### Run configuration tests
```bash
# Test split config system
python test_config_split.py

# Test config loader
python utils/config_loader.py
```

### Run full quantization test
```bash
# Test with real ImageNet-Mini data
python test_vit_block.py
```

Expected output:
- ✓ All 14 layers configured
- ✓ Critical 16-bit layers verified
- ✓ Profiling results saved to `log/`

## Troubleshooting

### Issue: "Configuration file not found"
**Solution:** Ensure you're in the project root directory or use absolute paths.

### Issue: Different results between split and single config
**Solution:** Run `python test_config_split.py` to verify configurations match.

### Issue: Low quantization quality
**Solution:** Check critical layers are using 16-bit precision:
```python
config = load_config_from_yaml('configs')
for layer in ['kv_act', 'sv_attn', 'mlp_act', 'mlp_act2', 'intSoft']:
    c = config.get_config(layer)
    print(f"{layer}: {c.bit_type.bits}-bit")  # Should be 16
```

## Best Practices

1. **Use split configs for production**: Better organization and maintainability
2. **Keep critical layers at 16-bit**: kv_act, sv_attn, mlp_act, mlp_act2, intSoft
3. **Test after changes**: Always run `test_config_split.py` after modifying configs
4. **Use profiling**: Enable profiling to monitor quantization quality
5. **Version control**: Commit config changes separately for easier tracking

## Migration from Single Config

If you have an existing single config file, you can continue using it:

```python
# Old code - still works!
quant_block = QuantTimmVitBlock(
    block=original_block,
    quant_config='configs/quant_config_int8.yaml',
    enable_profiling=True
)
```

To migrate to split configs:
1. Your split configs already exist in `configs/`
2. Change the path from file to directory:
   ```python
   quant_config='configs/quant_config_int8.yaml'  # Old
   quant_config='configs'                          # New
   ```
3. Test: `python test_config_split.py`

## Additional Resources

- **CLAUDE.md**: Project architecture and technical details
- **CONFIG_SPLIT_SUMMARY.md**: Implementation details of split config system
- **test_config_split.py**: Comprehensive configuration validation
- **test_vit_block.py**: End-to-end quantization testing
