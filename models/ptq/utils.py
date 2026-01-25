import torch
import torch.nn as nn
import torch.nn.functional as F

from .layer_observer.minmax import MinmaxObserver
from .layer_observer.percentile import PercentileObserver
from .layer_observer.omse import OmseObserver
from .layer_observer.kl_divergence import KLObserver



ACTIVATION_MAP = {
    nn.ReLU: F.relu,
    nn.ReLU6: F.relu6,
    nn.GELU: F.gelu,
    nn.SiLU: F.silu,
    nn.Sigmoid: torch.sigmoid,
    nn.Tanh: torch.tanh,
    nn.LeakyReLU: lambda x, m: F.leaky_relu(x, m.negative_slope),
    nn.Hardswish: F.hardswish,
}

def init_observers(observer_type, bit_type,
                   module_type, calibration_mode,
                   quant_config):
    """Observer 초기화 함수"""
    # Normalize observer type to handle both formats
    observer_type_lower = observer_type.lower()

    if observer_type_lower in ['minmaxobserver', 'minmax']:
        observer = MinmaxObserver(
            bit_type=bit_type,
            module_type=module_type,
            calibration_mode=calibration_mode
        )
    elif observer_type_lower in ['percentileobserver', 'percentile']:
        observer = PercentileObserver(
            bit_type=bit_type,
            module_type=module_type,
            calibration_mode=calibration_mode,
            percentile_alpha=quant_config.percentile_alpha,
            percentile_sigma=quant_config.percentile_sigma
        )
    elif observer_type_lower in ['omseobserver', 'omse']:
        observer = OmseObserver(
            bit_type=bit_type,
            module_type=module_type,
            calibration_mode=calibration_mode
        )
    elif observer_type_lower in ['klobserver', 'kl']:
        observer = KLObserver(
            bit_type=bit_type,
            module_type=module_type,
            calibration_mode=calibration_mode,
            hist_bins=quant_config.kl_bins
        )
    else:
        raise ValueError(f"Unknown observer type: {observer_type}. "
                       f"Valid options: MinmaxObserver, PercentileObserver, "
                       f"OmseObserver, KLObserver (case-insensitive)")

    return observer

