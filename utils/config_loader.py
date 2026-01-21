"""YAML configuration loader for quantization settings"""
import yaml
from pathlib import Path
from typing import Union
import sys
sys.path.append(str(Path(__file__).parent.parent))

from quant_config import QuantConfig, BitTypeConfig, LayerQuantConfig


def load_config_from_yaml(yaml_path: Union[str, Path]) -> LayerQuantConfig:
    """
    Load quantization configuration from YAML file.

    Args:
        yaml_path: Path to YAML configuration file

    Returns:
        LayerQuantConfig object with default and layer-specific configurations

    Example YAML structure:
        default:
          calibration_mode: channel_wise
          bit_type:
            bits: 8
            signed: true
            name: int8
          observer_type: PercentileObserver
          quantization_method: Uniform
          percentile_alpha: 0.95
          percentile_sigma: 0.01
          kv_bins: 2048

        layers:  # Weight quantization
          attn_qkv:
            bit_type:
              bits: 16
              signed: true
              name: int16
          attn_proj:
            # Uses default config

        attn_layer:  # Activation quantization
          attn_qkv_output:
            # config here
          kv_act:
            # config here

        residual_layer:  # Residual quantization
          residual1:
            # config here
    """
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

    with open(yaml_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    # Parse default configuration
    default_dict = config_dict.get('default', {})
    default_config = _parse_quant_config(default_dict)

    # Parse layer-specific configurations from all sections
    layer_configs = {}

    # Section 1: Weight quantization layers
    layers_dict = config_dict.get('layers', {})
    for layer_name, layer_dict in layers_dict.items():
        if layer_dict is None or not layer_dict:
            # Empty entry means use default config
            continue

        # Merge with default config (layer config overrides default)
        merged_dict = {**default_dict, **layer_dict}

        # Handle bit_type merging separately (nested dict)
        if 'bit_type' in layer_dict and 'bit_type' in default_dict:
            merged_dict['bit_type'] = {**default_dict['bit_type'], **layer_dict['bit_type']}

        layer_configs[layer_name] = _parse_quant_config(merged_dict)

    # Section 2: Activation quantization layers (attn_layer)
    attn_layer_dict = config_dict.get('attn_layer', {})
    for layer_name, layer_dict in attn_layer_dict.items():
        if layer_dict is None or not layer_dict:
            continue

        merged_dict = {**default_dict, **layer_dict}
        if 'bit_type' in layer_dict and 'bit_type' in default_dict:
            merged_dict['bit_type'] = {**default_dict['bit_type'], **layer_dict['bit_type']}

        layer_configs[layer_name] = _parse_quant_config(merged_dict)

    # Section 3: Residual quantization layers (residual_layer)
    residual_layer_dict = config_dict.get('residual_layer', {})
    for layer_name, layer_dict in residual_layer_dict.items():
        if layer_dict is None or not layer_dict:
            continue

        merged_dict = {**default_dict, **layer_dict}
        if 'bit_type' in layer_dict and 'bit_type' in default_dict:
            merged_dict['bit_type'] = {**default_dict['bit_type'], **layer_dict['bit_type']}

        layer_configs[layer_name] = _parse_quant_config(merged_dict)

    return LayerQuantConfig(
        default_config=default_config,
        layer_configs=layer_configs
    )


def _parse_quant_config(config_dict: dict) -> QuantConfig:
    """Parse dictionary into QuantConfig object"""
    # Parse bit_type
    bit_type_dict = config_dict.get('bit_type', {})
    bit_type = BitTypeConfig(
        bits=bit_type_dict.get('bits', 8),
        signed=bit_type_dict.get('signed', True),
        name=bit_type_dict.get('name', 'int8')
    )

    # Parse QuantConfig
    return QuantConfig(
        calibration_mode=config_dict.get('calibration_mode', 'layer_wise'),
        bit_type=bit_type,
        observer_type=config_dict.get('observer_type', 'PercentileObserver'),
        quantization_method=config_dict.get('quantization_method', 'Uniform'),
        percentile_alpha=config_dict.get('percentile_alpha', 0.95),
        percentile_sigma=config_dict.get('percentile_sigma', 0.01),
        kv_bins=config_dict.get('kv_bins', 2048)
    )


def save_config_to_yaml(layer_config: LayerQuantConfig, yaml_path: Union[str, Path]):
    """
    Save LayerQuantConfig to YAML file.

    Args:
        layer_config: LayerQuantConfig object to save
        yaml_path: Output path for YAML file
    """
    yaml_path = Path(yaml_path)

    # Convert to dictionary
    config_dict = {
        'default': _quant_config_to_dict(layer_config.default_config),
        'layers': {}
    }

    for layer_name, layer_quant_config in layer_config.layer_configs.items():
        config_dict['layers'][layer_name] = _quant_config_to_dict(layer_quant_config)

    # Save to YAML
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)


def _quant_config_to_dict(config: QuantConfig) -> dict:
    """Convert QuantConfig to dictionary"""
    return {
        'calibration_mode': config.calibration_mode,
        'bit_type': {
            'bits': config.bit_type.bits,
            'signed': config.bit_type.signed,
            'name': config.bit_type.name
        },
        'observer_type': config.observer_type,
        'quantization_method': config.quantization_method,
        'percentile_alpha': config.percentile_alpha,
        'percentile_sigma': config.percentile_sigma,
        'kv_bins': config.kv_bins
    }


if __name__ == '__main__':
    # Test loading configurations
    configs_dir = Path(__file__).parent.parent / 'configs'

    print("Testing INT8 configuration:")
    int8_config = load_config_from_yaml(configs_dir / 'quant_config_int8.yaml')
    print(f"  Default bits: {int8_config.default_config.bit_type.bits}")
    print(f"  Default observer: {int8_config.default_config.observer_type}")
    print(f"  Layer configs: {list(int8_config.layer_configs.keys())}")

    print("\nTesting Mixed Precision configuration:")
    mixed_config = load_config_from_yaml(configs_dir / 'quant_config_mixed_precision.yaml')
    print(f"  Default bits: {mixed_config.default_config.bit_type.bits}")
    print(f"  attn_qkv bits: {mixed_config.get_config('attn_qkv').bit_type.bits}")
    print(f"  kv_act bits: {mixed_config.get_config('kv_act').bit_type.bits}")
    print(f"  mlp_fc1 bits: {mixed_config.get_config('mlp_fc1').bit_type.bits}")

    print("\nTesting Aggressive configuration:")
    aggressive_config = load_config_from_yaml(configs_dir / 'quant_config_aggressive.yaml')
    print(f"  Default bits: {aggressive_config.default_config.bit_type.bits}")
    print(f"  attn_qkv bits: {aggressive_config.get_config('attn_qkv').bit_type.bits}")
    print(f"  kv_act bits: {aggressive_config.get_config('kv_act').bit_type.bits}")
    print(f"  mlp_fc1 bits: {aggressive_config.get_config('mlp_fc1').bit_type.bits}")
