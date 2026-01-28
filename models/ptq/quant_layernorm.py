import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import copy
from torch import Tensor
import os

from quant_config import QuantConfig, BitTypeConfig
from .bit_type import BitType
from .layer_quantizer.build import build_quantizer
from .utils import init_observers
from .layer_profiler.profiler import profiler

class QLayerNorm(nn.Module):
    def __init__(self,
                 input_module: nn.Module,
                 quant_config: QuantConfig,
                 layer_name: str = 'qlayernorm'):
        super().__init__()

        # Config 설정 (input_module 참조 저장하지 않음 - 메모리 절약)
        self.layer_name = layer_name
        self.quant_config = quant_config
        self.enable_profiling = quant_config.enable_profiler
        self.observer_type = quant_config.observer_type
        self.bit_type = BitType(
            bits=quant_config.bit_type.bits,
            symmetric=quant_config.bit_type.symmetric,
            name=quant_config.bit_type.name
        )
        self.calibration_mode = quant_config.calibration_mode
        self.quant_weight = None
        self.mode = 'fp32'

        # 필요한 값만 복사
        self.normalized_shape = input_module.normalized_shape
        self.eps = input_module.eps
        self.num_channels = input_module.weight.size(0)

        #LayerNorm specific params
        self.scale = None
        self.zero = None
        self.s_base = None

        #1. set layer type & observer
        self.output_observer = init_observers(self.observer_type,
                                        self.bit_type,
                                        'activation',
                                        self.calibration_mode,
                                        self.quant_config)

        #2. quantizer build
        self.output_quantizer = build_quantizer(
            quantizer_str='uniform',
            bit_type=self.bit_type,
            module_type='activation')

        #3. profiler 초기화
        self.profiler = None
        if self.enable_profiling:
            self.profiler = profiler(layer_name)

        #4. layer initialization - 필요한 값만 복사하고 원본 참조는 저장하지 않음
        self.fwd_kwargs = dict()
        self.fwd_func = F.layer_norm

        self.weight = input_module.weight.clone().detach()

        self.bias = input_module.bias.clone().detach() if input_module.bias is not None else None

    
    def calibration(self, x):
        """Calibration 전용 - observer update만 수행"""
        with torch.no_grad():
            out = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
            self.output_observer.update(out)

        return out
    
    @staticmethod
    def _quantize(x, scale, bitwidth=8):
        """Fake quantization helper"""
        q_min = -(2 ** (bitwidth - 1))
        q_max = 2 ** (bitwidth - 1) - 1
        x_q = torch.round(x / scale)
        x_q = torch.clamp(x_q, q_min, q_max)
        return x_q * scale

    def _find_best_ptf(self, X, s_base, K=3, bitwidth=8):
        """
        PTF alpha 탐색: alpha_c = argmin | X_c - quant(X_c, 2^alpha * s) |_2

        Args:
            X: 입력 텐서 (..., Channel)
            s_base: layer-wise base scale
            K: 최대 시프트 값 (0 ~ K)
            bitwidth: 양자화 비트수
        """
        C = X.shape[-1]
        X_flat = X.reshape(-1, C)
        best_alphas = torch.zeros(C, dtype=torch.int32, device=X.device)

        for c in range(C):
            Xc = X_flat[:, c]
            min_error = float('inf')
            best_alpha = 0

            for alpha in range(K + 1):
                current_scale = (2 ** alpha) * s_base
                Xc_quant = self._quantize(Xc, current_scale, bitwidth)
                error = torch.norm(Xc - Xc_quant, p=2).item()

                if error < min_error:
                    min_error = error
                    best_alpha = alpha

            best_alphas[c] = best_alpha

        return best_alphas

    def _find_alpha(self, calib_loader):
        """alpha 탐색용 내부 메서드"""
        all_alphas = []
        for x in calib_loader:
            with torch.no_grad():
                out = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
                alpha = self._find_best_ptf(out, self.s_base, K=3, bitwidth=self.bit_type.bits)
                all_alphas.append(alpha)
        # 배치별 alpha 중 평균 후 반올림
        return torch.stack(all_alphas).float().mean(dim=0).round().int() 

    def compute_quant_params(self, calib_loader):
        """Calibration 끝나고 한 번 호출"""
        scale, self.zero = self.output_observer.get_quantization_params()

        # scale: layer-wise (mean 또는 max), zero : layer-wise
        self.s_base = scale.mean()
        self.scale = scale.fill_(self.s_base)  # shape 유지하면서 값만 통일
        self.output_quantizer.update_quantization_params(
            self.scale, self.zero
            )

        if calib_loader is not None:
            self.alpha = self._find_alpha(calib_loader)
            # PTF 적용: scale = 2^alpha * s_base
            self.scale = (2 ** self.alpha.float()) * self.s_base
            self.output_quantizer.update_quantization_params(self.scale, self.zero)
        else:
            # calib_loader 없으면 alpha=0 (기본값, PTF 미적용)
            self.alpha = torch.zeros(self.num_channels, dtype=torch.int32)

        return (self.scale, self.zero)

    def get_profiler(self):
        """Get profiler object"""
        if self.enable_profiling and self.profiler is not None:
            return self.profiler
        return None

    def get_profiling_results(self):
        """
        Get all profiling results after forward pass.

        Returns:
            dict: Dictionary containing statistics, histogram, time, and memory records
                  Returns None if profiling is not enabled

        Usage:
            layer.forward(x)  # Automatically updates profiler
            results = layer.get_profiling_results()
            print(results['statistics']['qsnr'])
            print(results['time'])
        """
        if not self.enable_profiling or self.profiler is None:
            return None

        try:
            stats = self.profiler.get_statistic() if len(self.profiler.weight_batch_list) > 0 else None
            hist = self.profiler.get_hist() if len(self.profiler.weight_batch_list) > 0 else None
        except (ValueError, AttributeError):
            # update_weight()가 아직 호출되지 않은 경우
            stats = None
            hist = None

        return {
            'statistics': stats,
            'histogram': hist,
            'time': self.profiler.get_time_record(),
            'memory': self.profiler.get_memory_record()
        }

    def reset_profiling(self):
        """Reset profiling data"""
        if self.enable_profiling and self.profiler is not None:
            self.profiler.reset_time_profiler()
            self.profiler.reset_memory_profiler()
            self.profiler.clear_batches()


    @staticmethod
    def get_MN(x, bit=8):
        """
        FQ-ViT bit-shift 파라미터 계산
        x / scale → (M * x) >> N 변환을 위한 M, N 계산

        Args:
            x: 입력 텐서 (일반적으로 affine transform 계수)
            bit: 비트 폭 (default: 8)

        Returns:
            M: Mantissa (고정소수점 표현)
            N: Shift 값 (2^N으로 나눌 값)
        """
        # N: 시프트 값 계산 (2^N으로 나눌 값)
        N = torch.clamp(bit - 1 - torch.floor(torch.log2(x + 1e-8)), 0, 31)
        # M: Mantissa (고정소수점 표현)
        M = torch.clamp(torch.floor(x * torch.pow(2, N)), 0, 2 ** bit - 1)
        return M, N

    def _integer_layernorm(self, x, in_quantizer, out_quantizer):
        """
        FQ-ViT Integer LayerNorm (bit-shift 기반)

        Args:
            x: 입력 텐서
            in_quantizer: 이전 레이어의 quantizer (scale 정보 포함)
            out_quantizer: 다음 레이어의 quantizer (scale 정보 포함)
        """
        in_scale = in_quantizer.scale
        out_scale = out_quantizer.scale

        # Tensor shape: (B, N, C)
        channel_nums = x.shape[-1]

        # 1. Input de-quantization (INT8 → integer domain 개념상)
        x_q = (x / in_scale).round()

        # 2. Scale 통일 (channel-wise → layer-wise)
        # in_scale이 channel-wise일 경우 layer-wise로 변환
        if in_scale.numel() > 1:
            in_scale1 = in_scale.min()
            in_scale_mask = (in_scale / in_scale1).round()
            x_q = x_q * in_scale_mask
        else:
            in_scale1 = in_scale

        # 3. Integer domain에서 mean, std 계산
        mean_x_q = x_q.mean(dim=-1, keepdim=True) * in_scale1

        # Variance 계산 (integer domain)
        var_x_q = (in_scale1 / channel_nums) * torch.sqrt(
            channel_nums * (x_q ** 2).sum(dim=-1, keepdim=True) -
            x_q.sum(dim=-1, keepdim=True) ** 2 + 1e-5
        )

        # 4. Affine transformation 계산 (bit-shift 준비)
        # A = (in_scale1 / std) * weight / out_scale
        A = (in_scale1 / (var_x_q + self.eps)) * \
            self.weight.reshape(1, 1, -1) / out_scale
        A_sign = A.sign()
        M, N = self.get_MN(A.abs(), bit=self.bit_type.bits)

        # B = (bias - mean * weight / std) / out_scale * 2^N
        if self.bias is not None:
            B = ((self.bias.reshape(1, 1, -1) -
                  (mean_x_q / (var_x_q + self.eps)) * self.weight.reshape(1, 1, -1))
                 / out_scale * torch.pow(2, N)).round()
        else:
            B = (-(mean_x_q / (var_x_q + self.eps)) * self.weight.reshape(1, 1, -1)
                 / out_scale * torch.pow(2, N)).round()

        # 5. Integer-only 연산 (bit-shift)
        x_q = ((A_sign * M * x_q + B) / torch.pow(2, N)).round()

        # 6. Re-quantization
        x = x_q * out_scale

        return x

    def forward(self, x, in_quantizer=None, out_quantizer=None):
        """
        Forward pass with optional Integer LayerNorm

        Args:
            x: 입력 텐서
            in_quantizer: 이전 레이어의 quantizer (scale propagation용)
            out_quantizer: 다음 레이어의 quantizer (scale propagation용)
        """
        # Integer LayerNorm 모드 (FQ-ViT 방식)
        if self.mode == 'int' and in_quantizer is not None and out_quantizer is not None:
            if self.enable_profiling and self.profiler is not None:
                with self.profiler.measure_time():
                    return self._integer_layernorm(x, in_quantizer, out_quantizer)
            else:
                return self._integer_layernorm(x, in_quantizer, out_quantizer)

        # FP32 또는 기존 Quantized 모드
        out = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)

        if self.mode == 'quantized':
            if self.enable_profiling and self.profiler is not None:
                # Measure quantization time
                with self.profiler.measure_time():
                    # Store FP32 output before quantization
                    fp32_output = out.clone().detach()
                    # Fake quantization: quant -> dequant
                    out = self.output_quantizer.forward(out)
                    # Update profiler with FP32 vs Quantized outputs
                    self.profiler.update_weight(fp32_output, out.detach())
            else:
                out = self.output_quantizer.forward(out)
        elif self.enable_profiling and self.profiler is not None:
            # FP32 mode - measure time without quantization
            with self.profiler.measure_time():
                pass  # No quantization, just time measurement

        return out


def test_qlayernorm_profiling():
    """
    Test QLayerNorm profiling functionality.

    Tests:
    1. Forward pass automatically updates profiler
    2. get_profiling_results() returns correct data
    3. Statistics, histogram, and timing info are collected
    4. PTF (Per-Tensor-Factorization) functionality
    """
    print("="*80)
    print("QLayerNorm Profiling Test")
    print("="*80)

    # Create test LayerNorm module
    print("\n[1] Creating LayerNorm module")
    normalized_shape = 768
    original_ln = nn.LayerNorm(normalized_shape)
    print(f"  ✓ Original LayerNorm created (normalized_shape={normalized_shape})")

    # Create test config
    config = QuantConfig(
        bit_type=BitTypeConfig(bits=8, symmetric=True, name='int8'),
        observer_type='MinmaxObserver',
        calibration_mode='channel_wise',
        enable_profiler=True
    )

    # Create QLayerNorm with profiling enabled
    print("\n[2] Creating QLayerNorm with profiling enabled")
    layer = QLayerNorm(
        input_module=original_ln,
        quant_config=config,
        layer_name='test_qlayernorm'
    )
    print("  ✓ QLayerNorm created")
    print(f"  Layer name: {layer.layer_name}")
    print(f"  Profiling enabled: {layer.enable_profiling}")
    print(f"  Profiler object: {layer.profiler is not None}")
    print(f"  Normalized shape: {layer.normalized_shape}")

    # Generate test data
    print("\n[3] Generating calibration data")
    batch_size = 8
    seq_len = 197  # ViT patch tokens (196 patches + 1 cls token)
    feature_dim = 768
    num_calib_batches = 10

    calib_data = [torch.randn(batch_size, seq_len, feature_dim) for _ in range(num_calib_batches)]
    print(f"  Calibration batches: {num_calib_batches}")
    print(f"  Batch shape: {calib_data[0].shape}")

    # Calibration
    print("\n[4] Running calibration")
    layer.eval()
    with torch.no_grad():
        for idx, x in enumerate(calib_data):
            _ = layer.calibration(x)
            if (idx + 1) % 5 == 0:
                print(f"  Batch {idx + 1}/{num_calib_batches} completed")
    print("  ✓ Calibration completed")

    # Compute quantization parameters (without PTF for simplicity)
    print("\n[5] Computing quantization parameters")
    scaler, zero = layer.compute_quant_params(calib_loader=None)
    print(f"  Scale shape: {scaler.shape if isinstance(scaler, torch.Tensor) else 'scalar'}")
    print(f"  Scale (first 5): {scaler[:5] if isinstance(scaler, torch.Tensor) and scaler.numel() > 5 else scaler}")
    if isinstance(zero, torch.Tensor):
        if zero.numel() == 1:
            print(f"  Zero point: {zero.item()}")
        else:
            print(f"  Zero point shape: {zero.shape}, first 5: {zero[:5]}")
    else:
        print(f"  Zero point: {zero}")
    print(f"  Alpha (first 5): {layer.alpha[:5]}")
    print("  ✓ Quantization parameters computed")

    # Test data
    test_input = torch.randn(1, seq_len, feature_dim)

    # FP32 inference
    print("\n[6] Testing FP32 mode (no quantization)")
    layer.mode = 'fp32'
    layer.reset_profiling()

    with torch.no_grad():
        for _ in range(50):
            _ = layer(test_input)

    results_fp32 = layer.get_profiling_results()
    print("  ✓ FP32 forward completed (50 iterations)")

    if results_fp32 and results_fp32['time']:
        time_data = results_fp32['time']
        if 'test_qlayernorm' in time_data:
            stats = time_data['test_qlayernorm']
            print(f"  Timing - Count: {stats['count']}, Mean: {stats['mean']*1000:.4f} ms")
        else:
            print("  Warning: No timing data for 'test_qlayernorm'")
    else:
        print("  Warning: No timing results in FP32 mode")

    # Quantized inference
    print("\n[7] Testing Quantized mode (with profiling)")
    layer.mode = 'quantized'
    layer.reset_profiling()

    print("  Running 100 iterations...")
    with torch.no_grad():
        for _ in range(100):
            output_quant = layer(test_input)

    print("  ✓ Quantized forward completed (100 iterations)")
    print(f"  Output shape: {output_quant.shape}")
    print(f"  Output range: [{output_quant.min():.6f}, {output_quant.max():.6f}]")

    # Get profiling results
    print("\n[8] Retrieving profiling results")
    results = layer.get_profiling_results()

    if results is None:
        print("  ✗ ERROR: get_profiling_results() returned None!")
        return

    print("  ✓ Profiling results retrieved")
    print(f"  Result keys: {list(results.keys())}")

    # Check statistics
    print("\n[9] Statistics")
    if results['statistics'] is not None:
        stats = results['statistics']
        print("  ✓ Statistics available")
        qsnr = stats.get('qsnr', 'N/A')
        mse = stats.get('mse', 'N/A')
        max_error = stats.get('max_error', 'N/A')
        mean_error = stats.get('mean_error', 'N/A')

        print(f"    QSNR: {qsnr:.2f} dB" if isinstance(qsnr, (int, float)) else f"    QSNR: {qsnr}")
        print(f"    MSE: {mse:.10f}" if isinstance(mse, (int, float)) else f"    MSE: {mse}")
        print(f"    Max Error: {max_error:.6f}" if isinstance(max_error, (int, float)) else f"    Max Error: {max_error}")
        print(f"    Mean Error: {mean_error:.6f}" if isinstance(mean_error, (int, float)) else f"    Mean Error: {mean_error}")
    else:
        print("  ✗ Statistics: None")

    # Check histogram
    print("\n[10] Histogram")
    if results['histogram'] is not None:
        hist = results['histogram']
        print("  ✓ Histogram available")
        if 'kl_divergence' in hist:
            print(f"    KL Divergence: {hist['kl_divergence']:.6f}")
        print(f"    Histogram keys: {list(hist.keys())}")
    else:
        print("  ✗ Histogram: None")

    # Check timing
    print("\n[11] Timing Information")
    time_data = results['time']
    if time_data and 'test_qlayernorm' in time_data:
        time_stats = time_data['test_qlayernorm']
        print("  ✓ Timing data available")
        print(f"    Count: {time_stats['count']}")
        print(f"    Mean: {time_stats['mean']*1000:.4f} ms")
        print(f"    Min: {time_stats['min']*1000:.4f} ms")
        print(f"    Max: {time_stats['max']*1000:.4f} ms")
        print(f"    Total: {time_stats['total']*1000:.4f} ms")
    else:
        print("  ✗ Timing: No data for 'test_qlayernorm'")
        print("    Available keys: {}".format(list(time_data.keys()) if time_data else 'None'))

    # Check memory
    print("\n[12] Memory Information")
    mem_data = results['memory']
    if mem_data:
        print(f"  Memory profiler data: {mem_data}")
    else:
        print("  Memory: No data (expected if not attached)")

    # Verify profiler object
    print("\n[13] Verifying profiler object")
    prof = layer.get_profiler()
    if prof is not None:
        print("  ✓ get_profiler() returns profiler object")
        print(f"    Profiler name: {prof.name}")
        print(f"    Has weight: {prof.weight is not None}")
        print(f"    Has quant_weight: {prof.quant_weight is not None}")
    else:
        print("  ✗ get_profiler() returned None")

    # Test reset
    print("\n[14] Testing reset_profiling()")
    layer.reset_profiling()
    print("  ✓ reset_profiling() called")

    # Verify reset
    prof_after_reset = layer.get_profiler()
    if prof_after_reset is not None:
        print(f"    Weight after reset: {prof_after_reset.weight}")
        print(f"    Quant weight after reset: {prof_after_reset.quant_weight}")

    # Test with PTF
    print("\n[15] Testing with PTF (Per-Tensor-Factorization)")
    print("  Re-calibrating with PTF enabled...")
    layer.reset_profiling()

    # Re-calibration for PTF
    with torch.no_grad():
        for x in calib_data:
            _ = layer.calibration(x)

    # Compute with PTF
    scaler_ptf, zero_ptf = layer.compute_quant_params(calib_loader=calib_data)
    print(f"  ✓ PTF quantization parameters computed")
    print(f"    Alpha (first 10): {layer.alpha[:10]}")
    print(f"    Scale with PTF (first 5): {scaler_ptf[:5] if isinstance(scaler_ptf, torch.Tensor) and scaler_ptf.numel() > 5 else scaler_ptf}")

    # Test quantized with PTF
    layer.mode = 'quantized'
    with torch.no_grad():
        output_ptf = layer(test_input)
    print(f"  Output with PTF range: [{output_ptf.min():.6f}, {output_ptf.max():.6f}]")

    # Summary
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)

    passed_tests = []
    failed_tests = []

    # Check critical functionality
    if layer.profiler is not None:
        passed_tests.append("Profiler initialization")
    else:
        failed_tests.append("Profiler initialization")

    if results is not None:
        passed_tests.append("get_profiling_results() returns data")
    else:
        failed_tests.append("get_profiling_results() returns data")

    if results and results['statistics'] is not None:
        passed_tests.append("Statistics collection (FP32 vs Quantized)")
    else:
        failed_tests.append("Statistics collection")

    if results and results['histogram'] is not None:
        passed_tests.append("Histogram generation")
    else:
        failed_tests.append("Histogram generation")

    if results and results['time'] and 'test_qlayernorm' in results['time']:
        passed_tests.append("Timing profiling")
    else:
        failed_tests.append("Timing profiling")

    if layer.alpha is not None and layer.alpha.numel() > 0:
        passed_tests.append("PTF (Per-Tensor-Factorization)")
    else:
        failed_tests.append("PTF")

    print(f"\n✓ Passed: {len(passed_tests)}")
    for test in passed_tests:
        print(f"  - {test}")

    if failed_tests:
        print(f"\n✗ Failed: {len(failed_tests)}")
        for test in failed_tests:
            print(f"  - {test}")

    print("\n" + "="*80)
    if len(failed_tests) == 0:
        print("ALL TESTS PASSED!")
    else:
        print(f"SOME TESTS FAILED ({len(failed_tests)}/{len(passed_tests) + len(failed_tests)})")
    print("="*80)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    # Add project root to path
    project_root = Path(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    test_qlayernorm_profiling()
