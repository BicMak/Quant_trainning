from dataclasses import dataclass, field
from typing import Literal, Dict


@dataclass
class BitTypeConfig:
    bits: int = 8
    signed: bool = True
    name: str = 'int8'

@dataclass
class QuantConfig:
    calibration_mode: Literal['layer_wise', 'channel_wise'] = 'layer_wise'
    bit_type: BitTypeConfig = field(default_factory=lambda: BitTypeConfig())
    observer_type: Literal['MinmaxObserver', 'PercentileObserver', 'OmseObserver', 'KLObserver'] = 'PercentileObserver'

    # set quantization_method
    quantization_method: Literal['Uniform', 'Log2'] = 'Uniform'

    # Percentile observer parameters
    percentile_alpha: float = 0.95
    percentile_sigma: float = 0.01

    # KL observer parameters
    kl_bins: int = 2048


@dataclass
class LayerQuantConfig:
    """
    Layer-specific quantization configuration.

    Manages default quantization settings and per-layer overrides.
    When a layer is requested via get_config(), it returns the layer-specific
    config if available, otherwise falls back to the default config.
    """
    default_config: QuantConfig
    layer_configs: Dict[str, QuantConfig] = field(default_factory=dict)

    def get_config(self, layer_name: str) -> QuantConfig:
        """
        Get quantization config for a specific layer.

        Args:
            layer_name: Name of the layer (e.g., 'attn_qkv', 'mlp_fc1')

        Returns:
            QuantConfig for the layer (layer-specific if available, else default)
        """
        return self.layer_configs.get(layer_name, self.default_config)
    