# Copyright (c) MEGVII Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn as nn


from .base import BaseObserver
from .utils import lp_loss
from ..bit_type import BitType


class KLObserver(BaseObserver):
    """
    KL Divergence based Observer for quantization calibration.

    Supports both layer-wise and channel-wise calibration modes:
    - layer_wise: Single threshold for entire tensor (faster)
    - channel_wise: Per-channel threshold optimization (more accurate)
    """

    def __init__(self,
                 bit_type,
                 module_type,
                 calibration_mode,
                 hist_bins=2048):
        super(KLObserver, self).__init__(bit_type, module_type, calibration_mode)

        self.max_val = None
        self.min_val = None
        self.percentile_sigma = 0.01  #Fixed
        self.percentile_alpha = 0.99999     #Fixed
        self.symmetric = self.bit_type.signed

        self.start_ratio = 0.1  # start searching from 10% of bins
        self.hist_bins = hist_bins
        self.v = None
        self.device = None

    def update(self, v: torch.Tensor):
        if self.device is None:
            self.device = v.device.type

        if self.device != v.device.type:
            raise ValueError(
                "Device type mismatch in observer. "
                f"Expected device type: {self.device}, but got: {v.device.type}")

        # CUDA histogram을 위해 CPU로 이동
        v_cpu = v.cpu() if v.device.type == 'cuda' else v
        v_reshaped = self.reshape_tensor(v_cpu)
        self.v = v_cpu.detach()

        # 1. Initialize min/max bounds using percentile
        self._cal_minmax(v_reshaped)

        if self.calibration_mode == 'layer_wise':
            self._update_layer_wise(v_reshaped)
        elif self.calibration_mode == 'channel_wise':
            self._update_channel_wise(v_reshaped)
        else:
            raise ValueError(f"Unknown calibration_mode: {self.calibration_mode}")

    def _update_layer_wise(self, v_reshaped: torch.Tensor):
        """Layer-wise KL divergence: single threshold for entire tensor."""
        v_flat = v_reshaped.reshape(-1)

        # Get symmetric range
        abs_max = max(abs(self.min_val.item()), abs(self.max_val.item()))

        # Build histogram
        hist_origin, bin_edge = torch.histogram(
            v_flat,
            bins=self.hist_bins,
            range=(-abs_max, abs_max)
        )

        # Compute origin probability
        hist_origin_P = hist_origin / (hist_origin.sum() + self.eps)

        # Search for optimal threshold
        start_idx = int(self.hist_bins * self.start_ratio)
        best_threshold = self._search_optimal_threshold(
            v_flat, hist_origin_P, bin_edge, start_idx, abs_max
        )

        self.max_val = best_threshold
        self.min_val = -best_threshold

    def _update_channel_wise(self, v_reshaped: torch.Tensor):
        """Channel-wise KL divergence: per-channel threshold optimization."""
        num_channels = v_reshaped.shape[0]

        max_vals = torch.zeros(num_channels)
        min_vals = torch.zeros(num_channels)

        for ch in range(num_channels):
            v_ch = v_reshaped[ch]

            # Get symmetric range for this channel
            ch_abs_max = max(
                abs(v_ch.min().item()),
                abs(v_ch.max().item())
            )

            if ch_abs_max < self.eps:
                # Skip channels with zero variance
                max_vals[ch] = self.eps
                min_vals[ch] = -self.eps
                continue

            # Build histogram for this channel
            hist_origin, bin_edge = torch.histogram(
                v_ch,
                bins=self.hist_bins,
                range=(-ch_abs_max, ch_abs_max)
            )

            # Compute origin probability
            hist_origin_P = hist_origin / (hist_origin.sum() + self.eps)

            # Search for optimal threshold
            start_idx = int(self.hist_bins * self.start_ratio)
            best_threshold = self._search_optimal_threshold(
                v_ch, hist_origin_P, bin_edge, start_idx, ch_abs_max
            )

            max_vals[ch] = best_threshold.item()
            min_vals[ch] = -best_threshold.item()

        self.max_val = max_vals
        self.min_val = min_vals

    def _search_optimal_threshold(self, v_data, hist_origin_P, bin_edge, start_idx, abs_max, step=25):
        """Search for optimal threshold that minimizes KL divergence.

        Args:
            step: Search interval (default 25 for faster search)
        """
        min_kl_div = float('inf')
        best_threshold = torch.tensor(abs_max)

        for idx in range(start_idx, self.hist_bins, step):
            threshold = bin_edge[idx]
            if threshold <= 0:
                continue

            kl_div = self._compute_kl(v_data, hist_origin_P, threshold, abs_max)

            if kl_div < min_kl_div:
                min_kl_div = kl_div
                best_threshold = threshold

        return best_threshold

    def _compute_kl(self, v_data, hist_origin_P, threshold, abs_max):
        """Compute KL divergence between original and quantized distributions."""
        v_clipped = torch.clip(v_data, max=threshold, min=-threshold)

        # Quantize to int8, symmetric zero_point
        qmax = self.bit_type.upper_bound
        scale = torch.abs(threshold) / float(qmax)
        v_quantized = torch.round(v_clipped / scale).clamp(-qmax, qmax)
        v_dequantized = v_quantized * scale

        # Make histogram again
        hist_ranged, _ = torch.histogram(
            v_dequantized,
            bins=self.hist_bins,
            range=(-abs_max, abs_max)
        )
        hist_ranged_P = hist_ranged / (hist_ranged.sum() + self.eps)

        # KL divergence: sum(P * log(P/Q)) for P > 0
        mask = hist_origin_P > 0
        frac = (hist_origin_P[mask] / (hist_ranged_P[mask] + self.eps)).log()
        kl_div = (hist_origin_P[mask] * frac).sum()

        return kl_div

    def _cal_minmax(self, v: torch.Tensor):
        """Calculate initial min/max bounds using percentile."""
        if self.calibration_mode == 'channel_wise':
            cur_max = torch.quantile(v, self.percentile_alpha, dim=1)
            cur_min = torch.quantile(v, 1.0 - self.percentile_alpha, dim=1)
        elif self.calibration_mode == 'layer_wise':
            cur_max = torch.quantile(v.reshape(-1), self.percentile_alpha)
            cur_min = torch.quantile(v.reshape(-1), 1.0 - self.percentile_alpha)
        else:
            raise ValueError(f"Unknown calibration_mode: {self.calibration_mode}")

        if self.max_val is None:
            self.max_val = cur_max.max() if self.calibration_mode == 'layer_wise' else cur_max
            self.min_val = cur_min.min() if self.calibration_mode == 'layer_wise' else cur_min
        else:
            self.max_val = self.max_val + self.percentile_sigma * (cur_max - self.max_val)
            self.min_val = self.min_val + self.percentile_sigma * (cur_min - self.min_val)

    def get_quantization_params(self, *args, **kwargs):
        max_val = self.max_val
        min_val = self.min_val

        qmax = self.bit_type.upper_bound
        qmin = self.bit_type.lower_bound

        if self.symmetric:
            # Symmetric quantization parameters
            max_val = torch.max(-min_val, max_val)
            scale = max_val / (float(qmax - qmin) / 2)

            # Handle both scalar and tensor cases
            if isinstance(scale, torch.Tensor):
                scale = scale.clamp(min=self.eps)
                zero_point = torch.zeros_like(max_val, dtype=torch.int64)
            else:
                scale = max(scale, self.eps)
                zero_point = torch.tensor(0, dtype=torch.int64)
        else:
            raise ValueError("Only symmetric quantization is supported in KLObserver.")

        return scale, zero_point


def main():
    print("=" * 70)
    print("KLObserver Test: Layer-wise vs Channel-wise KL Divergence")
    print("=" * 70)

    # GPU 사용 가능 여부 확인
    device_list = ['cpu']
    if torch.cuda.is_available():
        device_list.append('cuda')
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA not available, testing CPU only")

    print("\nNote: Channel-wise KL Divergence computes per-channel thresholds.")
    print("      Layer-wise KL Divergence computes a single threshold for all.")
    print()

    # 테스트할 calibration 모드
    calibration_modes = ['layer_wise', 'channel_wise']

    # 테스트 케이스
    test_cases = [
        {
            'name': 'Conv2d Weight (8 channels)',
            'layer': nn.Conv2d(3, 8, kernel_size=3, padding=1),
            'input_shape': (4, 3, 32, 32),
            'module_type': 'conv_weight',
            'use_weight': True,
            'bit_type': BitType(bits=8, signed=True, name='int8_signed'),
            'hist_bins': 512  # Reduced for faster testing
        },
        {
            'name': 'Linear Weight (16 out_features)',
            'layer': nn.Linear(32, 16),
            'input_shape': (4, 32),
            'module_type': 'linear_weight',
            'use_weight': True,
            'bit_type': BitType(bits=8, signed=True, name='int8_signed'),
            'hist_bins': 512
        },
        {
            'name': 'Conv Activation',
            'layer': nn.Sequential(
                nn.Conv2d(3, 8, kernel_size=3, padding=1),
                nn.ReLU()
            ),
            'input_shape': (4, 3, 32, 32),
            'module_type': 'activation',
            'use_weight': False,
            'bit_type': BitType(bits=8, signed=True, name='int8_signed'),
            'hist_bins': 512
        },
    ]

    # Device별 테스트
    for device in device_list:
        print("=" * 70)
        print(f"Device: {device.upper()}")
        print("=" * 70)

        for test_case in test_cases:
            for calib_mode in calibration_modes:
                print(f"\n{'-'*70}")
                print(f"[{test_case['name']}] - Calibration: {calib_mode.upper()}")
                print(f"{'-'*70}")

                # 레이어를 device로 이동
                layer = test_case['layer'].to(device)

                # Observer 생성
                try:
                    observer = KLObserver(
                        bit_type=test_case['bit_type'],
                        module_type=test_case['module_type'],
                        calibration_mode=calib_mode,
                        hist_bins=test_case['hist_bins']
                    )

                    print("Observer config:")
                    print(f"  - BitType: {test_case['bit_type'].name}")
                    print(f"  - Calibration mode: {calib_mode}")
                    print(f"  - Histogram bins: {test_case['hist_bins']}")
                    print(f"  - Module type: {test_case['module_type']}")

                    if test_case['use_weight']:
                        # Weight 테스트
                        weight = layer.weight if isinstance(layer, (nn.Conv2d, nn.Linear)) else layer[0].weight
                        print(f"  - Input shape: {weight.shape}")
                        print(f"  - Input range: [{weight.min():.4f}, {weight.max():.4f}]")

                        # KL divergence 계산
                        print("\nComputing KL divergence...")
                        observer.update(weight)

                    else:
                        # Activation 테스트
                        dummy_input = torch.randn(*test_case['input_shape']).to(device)
                        with torch.no_grad():
                            output = layer(dummy_input)

                        print(f"  - Input shape: {output.shape}")
                        print(f"  - Input range: [{output.min():.4f}, {output.max():.4f}]")

                        # KL divergence 계산
                        print("\nComputing KL divergence...")
                        observer.update(output)

                    # Quantization parameters 가져오기
                    scale, zero_point = observer.get_quantization_params()

                    print("\n[Quantization Parameters]")
                    if calib_mode == 'layer_wise':
                        print(f"  Scale: {scale.item():.6f} (scalar)")
                        print(f"  Zero Point: {zero_point.item()}")
                        print(f"  Threshold: [{observer.min_val:.4f}, {observer.max_val:.4f}]")
                    else:
                        print(f"  Scale shape: {scale.shape} (per-channel)")
                        print(f"  Scale range: [{scale.min():.6f}, {scale.max():.6f}]")
                        print(f"  Zero Point shape: {zero_point.shape}")
                        print(f"  Threshold max_val shape: {observer.max_val.shape}")
                        print(f"  Threshold range: [{observer.max_val.min():.4f}, {observer.max_val.max():.4f}]")

                    # Verification
                    print("\n[Verification]")
                    if calib_mode == 'layer_wise':
                        print("  - Single threshold: OK")
                    else:
                        print(f"  - Per-channel thresholds: {observer.max_val.shape[0]} channels")

                except Exception as e:
                    print(f"\nError: {type(e).__name__}")
                    print(f"   {str(e)}")
                    import traceback
                    traceback.print_exc()

    print("\n" + "=" * 70)
    print("Test completed!")
    print("=" * 70)
    print("\n[Summary]")
    print(f"- Devices tested: {', '.join(device_list)}")
    print("- Calibration modes: layer_wise, channel_wise")
    print("- Layer types: Conv Weight, Linear Weight, Conv Activation")
    print("\n[Expected Behavior]")
    print("- Layer-wise: Single scale/zero_point for entire tensor")
    print("- Channel-wise: Per-channel scale/zero_point (more accurate, slower)")


if __name__ == "__main__":
    main()
