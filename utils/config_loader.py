"""YAML configuration loader for quantization settings"""
import yaml
from pathlib import Path
from typing import Union
import sys
sys.path.append(str(Path(__file__).parent.parent))

from quant_config import QuantConfig, BitTypeConfig, LayerQuantConfig


def load_config_from_yaml(yaml_path: Union[str, Path]) -> LayerQuantConfig:
    """
    Load quantization configuration from YAML file(s).

    Args:
        yaml_path: Path to YAML configuration file OR directory containing split configs
                  - If file: Load single monolithic config
                  - If directory: Automatically load weight_config.yaml,
                    activation_config.yaml, and residual_config.yaml

    Returns:
        LayerQuantConfig object with default and layer-specific configurations

    Example YAML structure (single file):
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
          kl_bins: 2048

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

    Example usage:
        # Load from single file
        config = load_config_from_yaml('configs/quant_config_int8.yaml')

        # Load from directory (automatically finds 3 split files)
        config = load_config_from_yaml('configs')
    """
    yaml_path = Path(yaml_path)

    # Check if path is a directory
    if yaml_path.is_dir():
        # Automatically load from 3 split config files
        return load_multi_config_from_yaml(config_dir=yaml_path)

    # Otherwise, load as single file
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
        kl_bins=config_dict.get('kl_bins', 2048)
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
        'kl_bins': config.kl_bins
    }


def load_multi_config_from_yaml(
    weight_config_path: Union[str, Path] = None,
    activation_config_path: Union[str, Path] = None,
    residual_config_path: Union[str, Path] = None,
    config_dir: Union[str, Path] = None
) -> LayerQuantConfig:
    """
    Load quantization configuration from 3 separate YAML files and merge them.

    Args:
        weight_config_path: Path to weight layer configuration file
        activation_config_path: Path to activation layer configuration file
        residual_config_path: Path to residual layer configuration file
        config_dir: If provided, automatically loads from
                   config_dir/weight_config.yaml,
                   config_dir/activation_config.yaml,
                   config_dir/residual_config.yaml

    Returns:
        LayerQuantConfig object with merged configurations

    Example usage:
        # Option 1: Specify directory
        config = load_multi_config_from_yaml(config_dir='configs')

        # Option 2: Specify individual files
        config = load_multi_config_from_yaml(
            weight_config_path='configs/weight_config.yaml',
            activation_config_path='configs/activation_config.yaml',
            residual_config_path='configs/residual_config.yaml'
        )
    """
    # If config_dir is provided, construct paths
    if config_dir is not None:
        config_dir = Path(config_dir)
        weight_config_path = config_dir / 'weight_config.yaml'
        activation_config_path = config_dir / 'activation_config.yaml'
        residual_config_path = config_dir / 'residual_config.yaml'

    # Convert paths to Path objects
    weight_config_path = Path(weight_config_path) if weight_config_path else None
    activation_config_path = Path(activation_config_path) if activation_config_path else None
    residual_config_path = Path(residual_config_path) if residual_config_path else None

    # Load configurations
    all_config_dicts = []
    config_paths = [
        ('weight', weight_config_path),
        ('activation', activation_config_path),
        ('residual', residual_config_path)
    ]

    for config_name, config_path in config_paths:
        if config_path and config_path.exists():
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
                all_config_dicts.append((config_name, config_dict))
        elif config_path:
            print(f"Warning: {config_name} config file not found: {config_path}")

    if not all_config_dicts:
        raise ValueError("No valid configuration files found")

    # Use the first config's default as the global default
    # (typically weight_config.yaml should have the most general default)
    default_dict = all_config_dicts[0][1].get('default', {})
    default_config = _parse_quant_config(default_dict)

    # Merge all layer configurations
    layer_configs = {}

    for config_name, config_dict in all_config_dicts:
        # Get default from this config file (for merging)
        file_default_dict = config_dict.get('default', default_dict)

        # Process 'layers' section (weight quantization)
        layers_dict = config_dict.get('layers', {})
        for layer_name, layer_dict in layers_dict.items():
            if layer_dict is None or not layer_dict:
                continue

            merged_dict = {**file_default_dict, **layer_dict}
            if 'bit_type' in layer_dict and 'bit_type' in file_default_dict:
                merged_dict['bit_type'] = {**file_default_dict['bit_type'], **layer_dict['bit_type']}

            layer_configs[layer_name] = _parse_quant_config(merged_dict)

        # Process 'attn_layer' section (activation quantization)
        attn_layer_dict = config_dict.get('attn_layer', {})
        for layer_name, layer_dict in attn_layer_dict.items():
            if layer_dict is None or not layer_dict:
                continue

            merged_dict = {**file_default_dict, **layer_dict}
            if 'bit_type' in layer_dict and 'bit_type' in file_default_dict:
                merged_dict['bit_type'] = {**file_default_dict['bit_type'], **layer_dict['bit_type']}

            layer_configs[layer_name] = _parse_quant_config(merged_dict)

        # Process 'residual_layer' section (residual quantization)
        residual_layer_dict = config_dict.get('residual_layer', {})
        for layer_name, layer_dict in residual_layer_dict.items():
            if layer_dict is None or not layer_dict:
                continue

            merged_dict = {**file_default_dict, **layer_dict}
            if 'bit_type' in layer_dict and 'bit_type' in file_default_dict:
                merged_dict['bit_type'] = {**file_default_dict['bit_type'], **layer_dict['bit_type']}

            layer_configs[layer_name] = _parse_quant_config(merged_dict)

    return LayerQuantConfig(
        default_config=default_config,
        layer_configs=layer_configs
    )


if __name__ == '__main__':
    # Test loading configurations
    configs_dir = Path(__file__).parent.parent / 'configs'

    print("="*80)
    print("Testing Single-File Configuration Loading")
    print("="*80)

    print("\nTesting INT8 configuration:")
    int8_config = load_config_from_yaml(configs_dir / 'quant_config_int8.yaml')
    print(f"  Default bits: {int8_config.default_config.bit_type.bits}")
    print(f"  Default observer: {int8_config.default_config.observer_type}")
    print(f"  Layer configs: {list(int8_config.layer_configs.keys())}")

    print("\n" + "="*80)
    print("Testing Multi-File Configuration Loading")
    print("="*80)

    print("\nLoading from 3 separate YAML files:")
    multi_config = load_multi_config_from_yaml(config_dir=configs_dir)
    print(f"  Default bits: {multi_config.default_config.bit_type.bits}")
    print(f"  Default observer: {multi_config.default_config.observer_type}")
    print(f"  Total layer configs: {len(multi_config.layer_configs)}")
    print(f"  Layer configs: {list(multi_config.layer_configs.keys())}")

    print("\n[Layer-specific Configuration Check]")
    # Check weight layers
    print("  Weight layers:")
    for layer_name in ['attn_qkv', 'attn_proj', 'mlp_fc1', 'mlp_fc2', 'norm1', 'norm2']:
        config = multi_config.get_config(layer_name)
        print(f"    {layer_name}: {config.bit_type.bits}-bit, {config.observer_type}")

    # Check activation layers
    print("\n  Activation layers:")
    for layer_name in ['attn_qkv_output', 'kv_act', 'sv_attn', 'mlp_act', 'mlp_act2', 'intSoft']:
        config = multi_config.get_config(layer_name)
        print(f"    {layer_name}: {config.bit_type.bits}-bit, {config.observer_type}")

    # Check residual layers
    print("\n  Residual layers:")
    for layer_name in ['residual1', 'residual2']:
        config = multi_config.get_config(layer_name)
        print(f"    {layer_name}: {config.bit_type.bits}-bit, {config.observer_type}")

    print("\n" + "="*80)
    print("Configuration Loading Tests Completed Successfully!")
    print("="*80)
