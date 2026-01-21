from dataclasses import dataclass, field
from typing import Literal


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
    
    # Percentile observer parameters
    percentile_alpha: float = 0.95
    percentile_sigma: float = 0.01
    
    # Kv observer parameters
    kv_bins: int = 2048
    