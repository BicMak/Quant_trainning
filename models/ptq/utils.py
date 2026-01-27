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
                   quant_config, num_heads=None, head_dim=None):
    """Observer 초기화 함수

    Args:
        observer_type: Observer 종류 (MinmaxObserver, PercentileObserver 등)
        bit_type: 양자화 비트 타입
        module_type: 모듈 타입 (activation, linear_weight 등)
        calibration_mode: calibration 모드 (layer_wise, channel_wise, head_wise)
        quant_config: 양자화 설정
        num_heads: head 수 (head_wise 모드에서 필요)
        head_dim: head 차원 (head_wise 모드에서 필요)
    """
    # Normalize observer type to handle both formats
    observer_type_lower = observer_type.lower()

    if observer_type_lower in ['minmaxobserver', 'minmax']:
        observer = MinmaxObserver(
            bit_type=bit_type,
            module_type=module_type,
            calibration_mode=calibration_mode,
            num_heads=num_heads,
            head_dim=head_dim
        )
    elif observer_type_lower in ['percentileobserver', 'percentile']:
        observer = PercentileObserver(
            bit_type=bit_type,
            module_type=module_type,
            calibration_mode=calibration_mode,
            percentile_alpha=quant_config.percentile_alpha,
            percentile_sigma=quant_config.percentile_sigma,
            num_heads=num_heads,
            head_dim=head_dim
        )
    elif observer_type_lower in ['omseobserver', 'omse']:
        observer = OmseObserver(
            bit_type=bit_type,
            module_type=module_type,
            calibration_mode=calibration_mode,
            num_heads=num_heads,
            head_dim=head_dim
        )
    elif observer_type_lower in ['klobserver', 'kl']:
        observer = KLObserver(
            bit_type=bit_type,
            module_type=module_type,
            calibration_mode=calibration_mode,
            hist_bins=quant_config.kl_bins,
            num_heads=num_heads,
            head_dim=head_dim
        )
    else:
        raise ValueError(f"Unknown observer type: {observer_type}. "
                       f"Valid options: MinmaxObserver, PercentileObserver, "
                       f"OmseObserver, KLObserver (case-insensitive)")

    return observer

