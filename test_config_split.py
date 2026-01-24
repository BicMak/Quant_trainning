"""
Test script to verify the split configuration system
"""
from pathlib import Path
from utils.config_loader import load_config_from_yaml, load_multi_config_from_yaml

def test_config_split():
    """Test both single-file and multi-file configuration loading"""
    configs_dir = Path(__file__).parent / 'configs'

    print("=" * 80)
    print("Configuration System Test")
    print("=" * 80)

    # Test 1: Single-file configuration
    print("\n[Test 1: Single-File Configuration]")
    print(f"Loading from: {configs_dir / 'quant_config_int8.yaml'}")

    single_config = load_config_from_yaml(configs_dir / 'quant_config_int8.yaml')

    print(f"  Default: {single_config.default_config.bit_type.bits}-bit, "
          f"{single_config.default_config.observer_type}")
    print(f"  Total layers: {len(single_config.layer_configs)}")
    print(f"  Layers: {list(single_config.layer_configs.keys())}")

    # Test 2: Multi-file configuration
    print("\n[Test 2: Multi-File Configuration (3 separate files)]")
    print("  Loading from:")
    print(f"    - {configs_dir / 'weight_config.yaml'}")
    print(f"    - {configs_dir / 'activation_config.yaml'}")
    print(f"    - {configs_dir / 'residual_config.yaml'}")

    multi_config = load_multi_config_from_yaml(config_dir=configs_dir)

    print(f"  Default: {multi_config.default_config.bit_type.bits}-bit, "
          f"{multi_config.default_config.observer_type}")
    print(f"  Total layers: {len(multi_config.layer_configs)}")
    print(f"  Layers: {list(multi_config.layer_configs.keys())}")

    # Test 3: Verify configurations are identical
    print("\n[Test 3: Configuration Comparison]")

    # Compare layer names
    single_layers = set(single_config.layer_configs.keys())
    multi_layers = set(multi_config.layer_configs.keys())

    print(f"  Single-file layers: {len(single_layers)}")
    print(f"  Multi-file layers: {len(multi_layers)}")
    print(f"  Match: {single_layers == multi_layers}")

    if single_layers != multi_layers:
        print(f"  Missing in multi: {single_layers - multi_layers}")
        print(f"  Extra in multi: {multi_layers - single_layers}")

    # Test 4: Detailed layer comparison
    print("\n[Test 4: Layer-by-Layer Comparison]")

    mismatches = []
    for layer_name in single_layers & multi_layers:
        single_layer = single_config.get_config(layer_name)
        multi_layer = multi_config.get_config(layer_name)

        if (single_layer.bit_type.bits != multi_layer.bit_type.bits or
            single_layer.observer_type != multi_layer.observer_type):
            mismatches.append(layer_name)
            print(f"  {layer_name}: MISMATCH")
            print(f"    Single: {single_layer.bit_type.bits}-bit, {single_layer.observer_type}")
            print(f"    Multi:  {multi_layer.bit_type.bits}-bit, {multi_layer.observer_type}")

    if not mismatches:
        print("  All layers match! ✓")

    # Test 5: Critical layers check (16-bit layers)
    print("\n[Test 5: Critical 16-bit Layers Check]")
    critical_layers = ['kv_act', 'sv_attn', 'mlp_act', 'mlp_act2', 'intSoft']

    for layer_name in critical_layers:
        config = multi_config.get_config(layer_name)
        status = "✓" if config.bit_type.bits == 16 else "✗"
        print(f"  {layer_name}: {config.bit_type.bits}-bit, {config.observer_type} {status}")

    print("\n" + "=" * 80)
    print("Configuration System Test Completed!")
    print("=" * 80)


if __name__ == '__main__':
    test_config_split()
