import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from torch import Tensor
import os

from quant_config import QuantConfig, BitTypeConfig
from .bit_type import BitType
from .layer_quantizer.build import build_quantizer
from .utils import init_observers
from .layer_profiler.profiler import profiler

class QAct(nn.Module):
    def __init__(self,
                 quant_config: QuantConfig,
                 act_module: nn.Module = None,
                 layer_name: str = 'qact',
                 num_heads: int = None,
                 head_dim: int = None):
        super().__init__()

        # Config 설정
        self.act_module = act_module
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
        self.mode = 'fp32'

        # Quantization params - register_buffer로 ONNX initializer로 인식되게 함
        self.register_buffer('scaler', None)
        self.register_buffer('zero', None)

        # Head 정보 저장 (head_wise 모드에서 사용)
        self.num_heads = num_heads
        self.head_dim = head_dim

        # Observer 초기화 (head_wise 모드면 num_heads, head_dim 전달)
        self.observer = init_observers(
            self.observer_type,
            self.bit_type,
            'activation',
            self.calibration_mode,
            self.quant_config,
            num_heads=num_heads,
            head_dim=head_dim
        )

        # Quantizer 초기화
        self.quantizer = build_quantizer(
            quantizer_str=quant_config.quantization_method.lower(),
            bit_type=self.bit_type,
            module_type='activation',
            num_heads=num_heads,
            head_dim=head_dim
        )

        # profiler 초기화
        self.profiler = None
        if self.enable_profiling:
            self.profiler = profiler(layer_name)

    def calibration(self, x):
        """Calibration 전용 - observer update만 수행"""
        with torch.no_grad():
            if self.act_module is not None:
                x = self.act_module(x)

            # Observer가 head_wise 처리를 자동으로 함
            self.observer.update(x)
            output = x

        return output

    def compute_quant_params(self):
        """Calibration 끝나고 한 번 호출"""
        self.scaler, self.zero = self.observer.get_quantization_params()

        # quantizer에 quantization param 설정
        self.quantizer.update_quantization_params(
            self.scaler, self.zero
        )

        return (self.scaler, self.zero)

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


    def forward(self, x):
        # Apply activation module if exists
        if self.act_module is not None:
            x = self.act_module(x)

        # Quantization
        if self.mode == 'quantized':
            if self.enable_profiling and self.profiler is not None:
                # Measure quantization time
                with self.profiler.measure_time():
                    # Store FP32 output before quantization
                    fp32_output = x.clone().detach()
                    # Fake quantization: quant -> dequant (quantizer가 head_wise 처리)
                    x = self.quantizer.forward(x)
                    # Update profiler with FP32 vs Quantized outputs
                    self.profiler.update_weight(fp32_output, x.detach())
            else:
                x = self.quantizer.forward(x)
        elif self.enable_profiling and self.profiler is not None:
            # FP32 mode - measure time without quantization
            with self.profiler.measure_time():
                pass  # No quantization, just time measurement

        return x


def test_qact_profiling():
    """
    Test QAct profiling functionality.

    Tests:
    1. Forward pass automatically updates profiler
    2. get_profiling_results() returns correct data
    3. Statistics, histogram, and timing info are collected
    """
    print("="*80)
    print("QAct Profiling Test")
    print("="*80)

    # Create test config (QuantConfig, BitTypeConfig already imported at top)
    config = QuantConfig(
        bit_type=BitTypeConfig(bits=8, symmetric=True, name='int8'),
        observer_type='MinmaxObserver',
        calibration_mode='layer_wise',
        enable_profiler=True
    )

    # Create QAct with profiling enabled
    print("\n[1] Creating QAct with profiling enabled")
    layer = QAct(
        quant_config=config,
        act_module=None,
        layer_name='test_qact'
    )
    print("  ✓ QAct created")
    print(f"  Layer name: {layer.layer_name}")
    print(f"  Profiling enabled: {layer.enable_profiling}")
    print(f"  Profiler object: {layer.profiler is not None}")

    # Generate test data
    print("\n[2] Generating calibration data")
    batch_size = 8
    feature_dim = 768
    num_calib_batches = 10

    calib_data = [torch.randn(batch_size, feature_dim) for _ in range(num_calib_batches)]
    print(f"  Calibration batches: {num_calib_batches}")
    print(f"  Batch shape: {calib_data[0].shape}")

    # Calibration
    print("\n[3] Running calibration")
    layer.eval()
    with torch.no_grad():
        for idx, x in enumerate(calib_data):
            _ = layer.calibration(x)
            if (idx + 1) % 5 == 0:
                print(f"  Batch {idx + 1}/{num_calib_batches} completed")
    print("  ✓ Calibration completed")

    # Compute quantization parameters
    print("\n[4] Computing quantization parameters")
    scaler, zero = layer.compute_quant_params()
    print(f"  Scale: {scaler.item() if isinstance(scaler, torch.Tensor) else scaler:.8f}")
    print(f"  Zero point: {zero.item() if isinstance(zero, torch.Tensor) else zero}")
    print("  ✓ Quantization parameters computed")

    # Test data
    test_input = torch.randn(1, feature_dim)

    # FP32 inference
    print("\n[5] Testing FP32 mode (no quantization)")
    layer.mode = 'fp32'
    layer.reset_profiling()

    with torch.no_grad():
        for _ in range(50):
            _ = layer(test_input)

    results_fp32 = layer.get_profiling_results()
    print("  ✓ FP32 forward completed (50 iterations)")

    if results_fp32 and results_fp32['time']:
        time_data = results_fp32['time']
        if 'test_qact' in time_data:
            stats = time_data['test_qact']
            print(f"  Timing - Count: {stats['count']}, Mean: {stats['mean']*1000:.4f} ms")
        else:
            print("  Warning: No timing data for 'test_qact'")
    else:
        print("  Warning: No timing results in FP32 mode")

    # Quantized inference
    print("\n[6] Testing Quantized mode (with profiling)")
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
    print("\n[7] Retrieving profiling results")
    results = layer.get_profiling_results()

    if results is None:
        print("  ✗ ERROR: get_profiling_results() returned None!")
        return

    print("  ✓ Profiling results retrieved")
    print(f"  Result keys: {list(results.keys())}")

    # Check statistics
    print("\n[8] Statistics")
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
    print("\n[9] Histogram")
    if results['histogram'] is not None:
        hist = results['histogram']
        print("  ✓ Histogram available")
        if 'kl_divergence' in hist:
            print(f"    KL Divergence: {hist['kl_divergence']:.6f}")
        print(f"    Histogram keys: {list(hist.keys())}")
    else:
        print("  ✗ Histogram: None")

    # Check timing
    print("\n[10] Timing Information")
    time_data = results['time']
    if time_data and 'test_qact' in time_data:
        time_stats = time_data['test_qact']
        print("  ✓ Timing data available")
        print(f"    Count: {time_stats['count']}")
        print(f"    Mean: {time_stats['mean']*1000:.4f} ms")
        print(f"    Min: {time_stats['min']*1000:.4f} ms")
        print(f"    Max: {time_stats['max']*1000:.4f} ms")
        print(f"    Total: {time_stats['total']*1000:.4f} ms")
    else:
        print("  ✗ Timing: No data for 'test_qact'")
        print(f"    Available keys: {list(time_data.keys()) if time_data else 'None'}")

    # Check memory
    print("\n[11] Memory Information")
    mem_data = results['memory']
    if mem_data:
        print(f"  Memory profiler data: {mem_data}")
    else:
        print("  Memory: No data (expected if not attached)")

    # Verify profiler object
    print("\n[12] Verifying profiler object")
    prof = layer.get_profiler()
    if prof is not None:
        print("  ✓ get_profiler() returns profiler object")
        print(f"    Profiler name: {prof.name}")
        print(f"    Has weight: {prof.weight is not None}")
        print(f"    Has quant_weight: {prof.quant_weight is not None}")
    else:
        print("  ✗ get_profiler() returned None")

    # Test reset
    print("\n[13] Testing reset_profiling()")
    layer.reset_profiling()
    print("  ✓ reset_profiling() called")

    # Verify reset
    prof_after_reset = layer.get_profiler()
    if prof_after_reset is not None:
        print(f"    Weight after reset: {prof_after_reset.weight}")
        print(f"    Quant weight after reset: {prof_after_reset.quant_weight}")

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

    if results and results['time'] and 'test_qact' in results['time']:
        passed_tests.append("Timing profiling")
    else:
        failed_tests.append("Timing profiling")

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

    test_qact_profiling()
