from dataclasses import dataclass, field
from typing import Literal


@dataclass
class BitTypeConfig:
    bits: int = 8
    symmetric: bool = True  # True: symmetric quantization (zero_point=0), False: asymmetric
    name: str = 'int8'

@dataclass
class QuantConfig:
    calibration_mode: Literal['layer_wise', 'channel_wise', 'head_wise'] = 'layer_wise'
    bit_type: BitTypeConfig = field(default_factory=lambda: BitTypeConfig())
    observer_type: Literal['MinmaxObserver', 'PercentileObserver', 'OmseObserver', 'KLObserver'] = 'PercentileObserver'
    quantization_method: Literal['Uniform', 'Affine'] = 'Uniform'

    # Percentile observer parameters
    percentile_alpha: float = 0.9999
    percentile_sigma: float = 0.01

    # KL observer parameters
    kl_bins: int = 2048

    # Profiler
    enable_profiler: bool = False

    # Output quantization enable/disable
    output_quant_enable: bool = True