"""Layer Observer Package for Post-Training Quantization (PTQ)

Provides calibration observer implementations:
- MinmaxObserver: Simple min/max based calibration
- PercentileObserver: Percentile-based with EMA
- OmseObserver: Optimal MSE through grid search
- KVObserver: KL divergence based calibration
"""

from .minmax import MinmaxObserver
from .percentile import PercentileObserver
from .omse import OmseObserver
from .kv_divergence import KVObserver

__version__ = '0.1.0'

__all__ = [
    'MinmaxObserver',
    'PercentileObserver',
    'OmseObserver',
    'KVObserver',
]