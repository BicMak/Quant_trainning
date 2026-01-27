"""YAML configuration loader for quantization settings"""
import yaml
from pathlib import Path
from typing import Union
import sys
sys.path.append(str(Path(__file__).parent.parent))

from quant_config import QuantConfig, BitTypeConfig


def load_config_from_yaml(yaml_path: Union[str, Path]) -> dict:
    """
    Load quantization configuration from YAML file(s).

    Args:
        yaml_path: Path to YAML configuration file containing separate output/weight/activation configs
                   OR per-layer configs (attn_qkv, attn_proj, etc.)

    Returns:
        dict with config keys. Can be:
        - Global configs: 'output', 'weight', 'activation', 'norm', 'intsoft'
        - Per-layer configs: 'attn_qkv', 'attn_proj', 'attn_q_out', 'attn_k_out', etc.

        Each value is either:
        - A QuantConfig object (for activation/norm/intsoft)
        - A dict with 'weight' and 'output' keys (for QuantLinear layers like attn_qkv, attn_proj)

    Example YAML structure (global format):
        outputs:
          calibration_mode: channel_wise
          bit_type:
            bits: 8
            symmetric: false
            name: int8
          observer_type: PercentileObserver
          ...

        weight:
          calibration_mode: channel_wise
          ...

    Example YAML structure (per-layer format):
        attn_qkv:
          weight:
            calibration_mode: channel_wise
            ...
          output:
            calibration_mode: channel_wise
            ...

        attn_q_out:
          calibration_mode: channel_wise
          ...

    Example usage:
        # Load config
        configs = load_config_from_yaml('configs/attn_config.yaml')

        # Global format:
        output_config = configs['output']
        weight_config = configs['weight']

        # Per-layer format:
        attn_qkv_config = configs['attn_qkv']
        weight_config = attn_qkv_config['weight']
        output_config = attn_qkv_config['output']
    """
    yaml_path = Path(yaml_path)

    # Load YAML file
    if not yaml_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

    with open(yaml_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    result = {}

    # Detect config format (global vs per-layer)
    # If 'outputs' or 'weight' exists at top level, it's global format
    # Otherwise, it's per-layer format
    is_global_format = ('outputs' in config_dict or 'weight' in config_dict)

    if is_global_format:
        # Parse global format (legacy)
        # Parse outputs config
        outputs_dict = config_dict.get('outputs', {})
        if outputs_dict:
            result['output'] = _parse_quant_config(outputs_dict)
        else:
            raise ValueError("'outputs' section not found in config file")

        # Parse weight config
        weight_dict = config_dict.get('weight', {})
        if weight_dict:
            result['weight'] = _parse_quant_config(weight_dict)
        else:
            raise ValueError("'weight' section not found in config file")

        # Parse activation config (optional)
        activation_dict = config_dict.get('activation', {})
        if activation_dict:
            result['activation'] = _parse_quant_config(activation_dict)

        # Parse norm config (optional)
        norm_dict = config_dict.get('norm', {})
        if norm_dict:
            result['norm'] = _parse_quant_config(norm_dict)

        # Parse intsoft config (optional)
        intsoft_dict = config_dict.get('intsoft', {})
        if intsoft_dict:
            result['intsoft'] = _parse_quant_config(intsoft_dict)

    else:
        # Parse per-layer format
        for layer_name, layer_config in config_dict.items():
            if not isinstance(layer_config, dict):
                continue

            # Check if this layer has separate weight/output configs
            # (QuantLinear layers like attn_qkv, attn_proj)
            if 'weight' in layer_config and 'output' in layer_config:
                result[layer_name] = {
                    'weight': _parse_quant_config(layer_config['weight']),
                    'output': _parse_quant_config(layer_config['output'])
                }
            # Otherwise it's a single config (QAct layers, IntSoft)
            else:
                result[layer_name] = _parse_quant_config(layer_config)

    return result


def _parse_quant_config(config_dict: dict) -> QuantConfig:
    """Parse dictionary into QuantConfig object"""
    # Parse bit_type
    bit_type_dict = config_dict.get('bit_type', {})
    bit_type = BitTypeConfig(
        bits=bit_type_dict.get('bits', 8),
        symmetric=bit_type_dict.get('symmetric', True),
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
        kl_bins=config_dict.get('kl_bins', 2048),
        enable_profiler=config_dict.get('enable_profiler', False),
        output_quant_enable=config_dict.get('output_quant_enable', True)
    )




if __name__ == '__main__':
    # Test loading configurations
    configs_dir = Path(__file__).parent.parent / 'configs'

    print("="*80)
    print("Testing Configuration Loading")
    print("="*80)

    print("\nTesting attn_config.yaml:")
    config = load_config_from_yaml(configs_dir / 'attn_config.yaml')
    print(f"  Loaded sections: {list(config.keys())}")

    # Check if it's per-layer format
    if 'attn_qkv' in config:
        print("\n  Per-layer format detected!")

        # attn_qkv (QuantLinear layer with weight and output configs)
        print("\n  attn_qkv config:")
        if isinstance(config['attn_qkv'], dict):
            print(f"    Weight bits: {config['attn_qkv']['weight'].bit_type.bits}")
            print(f"    Weight symmetric: {config['attn_qkv']['weight'].bit_type.symmetric}")
            print(f"    Weight observer: {config['attn_qkv']['weight'].observer_type}")
            print(f"    Output bits: {config['attn_qkv']['output'].bit_type.bits}")
            print(f"    Output enable: {config['attn_qkv']['output'].output_quant_enable}")

        # attn_q_out (QAct layer with single config)
        print("\n  attn_q_out config:")
        print(f"    Bits: {config['attn_q_out'].bit_type.bits}")
        print(f"    Signed: {config['attn_q_out'].bit_type.symmetric}")
        print(f"    Observer: {config['attn_q_out'].observer_type}")
        print(f"    Enable profiler: {config['attn_q_out'].enable_profiler}")

        # attn_kv_out (Attention score quantization)
        print("\n  attn_kv_out config:")
        print(f"    Bits: {config['attn_kv_out'].bit_type.bits}")
        print(f"    Signed: {config['attn_kv_out'].bit_type.symmetric}")
        print(f"    Observer: {config['attn_kv_out'].observer_type}")

        # intSoft
        print("\n  intSoft config:")
        print(f"    Bits: {config['intSoft'].bit_type.bits}")
        print(f"    Signed: {config['intSoft'].bit_type.symmetric}")
        print(f"    Observer: {config['intSoft'].observer_type}")

    else:
        # Global format (legacy)
        print("\n  Global format detected!")

        print("\n  Output config:")
        print(f"    Bits: {config['output'].bit_type.bits}")
        print(f"    Signed: {config['output'].bit_type.symmetric}")
        print(f"    Observer: {config['output'].observer_type}")
        print(f"    Enable profiler: {config['output'].enable_profiler}")
        print(f"    Output quant enable: {config['output'].output_quant_enable}")

        print("\n  Weight config:")
        print(f"    Bits: {config['weight'].bit_type.bits}")
        print(f"    Signed: {config['weight'].bit_type.symmetric}")
        print(f"    Observer: {config['weight'].observer_type}")
        print(f"    Enable profiler: {config['weight'].enable_profiler}")

        if 'activation' in config:
            print("\n  Activation config:")
            print(f"    Bits: {config['activation'].bit_type.bits}")
            print(f"    Signed: {config['activation'].bit_type.symmetric}")
            print(f"    Observer: {config['activation'].observer_type}")

        if 'intsoft' in config:
            print("\n  IntSoft config:")
            print(f"    Bits: {config['intsoft'].bit_type.bits}")
            print(f"    Signed: {config['intsoft'].bit_type.symmetric}")
            print(f"    Observer: {config['intsoft'].observer_type}")

    print("\n" + "="*80)
    print("Configuration Loading Test Completed Successfully!")
    print("="*80)
