"""Test script for config loader with YAML"""
from pathlib import Path

# Test 1: Check YAML structure
print("=" * 80)
print("Testing YAML Config Loader Integration")
print("=" * 80)

config_path = Path(__file__).parent / 'configs' / 'quant_config_int8.yaml'

try:
    import yaml
    print("\n[YAML Module Check]")
    print("  ✓ PyYAML is installed")

    # Load and display YAML structure
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    print(f"\n[YAML File Structure]")
    print(f"  Config file: {config_path}")
    print(f"  Top-level keys: {list(config_dict.keys())}")

    # Display layers section
    if 'layers' in config_dict:
        print(f"\n[Weight Layers Section]")
        print(f"  Layers defined: {list(config_dict['layers'].keys())}")

    # Display attn_layer section
    if 'attn_layer' in config_dict:
        print(f"\n[Activation Layers Section]")
        print(f"  Layers defined: {list(config_dict['attn_layer'].keys())}")

    # Display residual_layer section
    if 'residual_layer' in config_dict:
        print(f"\n[Residual Layers Section]")
        print(f"  Layers defined: {list(config_dict['residual_layer'].keys())}")

    # Test loading with config_loader
    print(f"\n[Config Loader Test]")
    from utils.config_loader import load_config_from_yaml

    layer_config = load_config_from_yaml(config_path)

    print(f"  ✓ Config loaded successfully")
    print(f"  Default config bits: {layer_config.default_config.bit_type.bits}")
    print(f"  Default observer: {layer_config.default_config.observer_type}")
    print(f"  Total layer configs: {len(layer_config.layer_configs)}")

    # Test specific layer configs
    test_layers = ['attn_qkv', 'attn_qkv_output', 'kv_act', 'residual1', 'norm1', 'intSoft']
    print(f"\n[Layer-Specific Configs]")
    for layer_name in test_layers:
        config = layer_config.get_config(layer_name)
        print(f"  {layer_name}:")
        print(f"    - bits: {config.bit_type.bits}")
        print(f"    - calibration_mode: {config.calibration_mode}")
        print(f"    - observer: {config.observer_type}")

    print(f"\n{'=' * 80}")
    print("✓ All tests passed!")
    print("=" * 80)

except ImportError as e:
    print(f"\n[Error]")
    print(f"  PyYAML is not installed")
    print(f"  Please install it with: pip install pyyaml")
    print(f"\n  Error details: {e}")
except Exception as e:
    print(f"\n[Error]")
    print(f"  {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print("\n[Usage Example]")
print("""
# Example 1: Load from YAML file (recommended)
from models.vit_block import QuantTimmVitBlock
quant_block = QuantTimmVitBlock(
    block=original_block,
    quant_config='configs/quant_config_int8.yaml',  # YAML 파일 경로
    enable_profiling=True
)

# Example 2: Load config first, then use
from utils.config_loader import load_config_from_yaml
layer_config = load_config_from_yaml('configs/quant_config_int8.yaml')
quant_block = QuantTimmVitBlock(
    block=original_block,
    quant_config=layer_config,  # LayerQuantConfig 객체
    enable_profiling=True
)

# Example 3: Use old method (single QuantConfig for all layers)
from quant_config import QuantConfig, BitTypeConfig
bit_config = BitTypeConfig(bits=8, signed=True, name='int8')
quant_config = QuantConfig(
    calibration_mode='channel_wise',
    bit_type=bit_config,
    observer_type='PercentileObserver'
)
quant_block = QuantTimmVitBlock(
    block=original_block,
    quant_config=quant_config,  # QuantConfig 객체 (기존 방식)
    enable_profiling=True
)
""")
