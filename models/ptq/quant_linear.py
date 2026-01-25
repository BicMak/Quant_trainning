import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch import Tensor
import os

from quant_config import QuantConfig, BitTypeConfig
from .bit_type import BitType
from .layer_quantizer.build import build_quantizer
from .utils import init_observers
from .layer_profiler.profiler import profiler

class QuantLinear(nn.Module):
    def __init__(self,
                 input_module: nn.Module,
                 quant_config: QuantConfig,
                 layer_name: str = 'qlinear',
                 enable_profiling: bool = False):
        super().__init__()

        # quant_config에서 설정 추출
        self.input_module = input_module
        self.layer_name = layer_name
        self.enable_profiling = enable_profiling
        self.quant_config = quant_config
        self.observer_type = quant_config.observer_type
        self.bit_type = BitType(
            bits=quant_config.bit_type.bits,
            signed=quant_config.bit_type.signed,
            name=quant_config.bit_type.name
        )

        # Hardcoded weight bit type (signed=True for weights)
        self.weight_bit_type = BitType(
            bits=quant_config.bit_type.bits,
            signed=True,  # Hardcoded for weights
            name=f"int{quant_config.bit_type.bits}_weight"
        )

        self.calibration_mode = quant_config.calibration_mode
        self.quant_weight = None
        self.mode = 'fp32'


        #1. set layer type & observer, fix minmax
        self.observer = init_observers("MinmaxObserver",  # Hardcoded
                                        self.weight_bit_type,  # Hardcoded signed=True
                                        'linear_weight',  # Hardcoded
                                        self.calibration_mode,
                                        self.quant_config)
        # output_observer는 별도로 초기화 (activation 타입으로)
        self.output_observer = init_observers(self.observer_type,
                                              self.bit_type,
                                              'activation',
                                              self.calibration_mode,
                                              self.quant_config)

        #2. quantizer build
        self.quantizer = build_quantizer(
            quantizer_str='uniform',
            bit_type=self.weight_bit_type,  # Use hardcoded weight bit type
            module_type='linear_weight')
        self.output_quantizer = build_quantizer(
            quantizer_str='uniform',
            bit_type=self.bit_type,
            module_type='activation')  # output은 activation type

        #3. profiler 초기화
        self.profiler = None
        self.weight_profiler = None
        if self.enable_profiling:
            self.profiler = profiler(layer_name + '_output')
            self.weight_profiler = profiler(layer_name + '_weight')

        #4. layer initialization
        self.fwd_kwargs = dict()
        self.fwd_func = F.linear

        self.weight = self.input_module.weight.clone().detach()

        if self.input_module.bias != None:
            self.bias = self.input_module.bias.clone().detach()
        else:
            self.bias = torch.zeros(self.input_module.weight.size(0)).to(
                self.input_module.weight.device
            )

    
    def calibration(self, x):
        """Calibration 전용 - observer update만 수행"""
        with torch.no_grad():
            self.observer.update(self.weight)
            output = self.fwd_func(x, self.weight, self.bias, **self.fwd_kwargs)
            self.output_observer.update(output)

        return output  # 필요하면 반환

    def compute_quant_params(self):
        """Calibration 끝나고 한 번 호출"""
        self.scaler, self.zero = self.observer.get_quantization_params()
        self.output_scaler, self.output_zero = self.output_observer.get_quantization_params()

        # Device 통일: weight의 device로 모든 파라미터 이동
        weight_device = self.weight.device
        self.scaler = self.scaler.to(weight_device)
        self.zero = self.zero.to(weight_device)
        self.output_scaler = self.output_scaler.to(weight_device)
        self.output_zero = self.output_zero.to(weight_device)

        # quantizer에 quantization param 설정
        self.quantizer.update_quantization_params(
            self.scaler, self.zero
            )
        self.output_quantizer.update_quantization_params(
            self.output_scaler, self.output_zero
            )

        # weight quantization, save quant_weight
        # scaler와 zero를 올바른 shape으로 reshape
        range_shape = self.quantizer.get_reshape_range(self.weight)
        scaler_reshaped = self.scaler.reshape(range_shape)
        zero_reshaped = self.zero.reshape(range_shape)

        self.quant_weight = torch.clamp(
            torch.round(self.weight / scaler_reshaped) + zero_reshaped,
            min=self.bit_type.lower_bound,
            max=self.bit_type.upper_bound
        )

        # profiler에 weight 업데이트 (FP32 vs Quantized weight 비교)
        if self.enable_profiling and self.weight_profiler is not None:
            # Manual dequantization: (quant_weight - zero) * scale
            # quant_weight is already integer values, so we need to dequantize properly
            dequant_weight = (self.quant_weight - zero_reshaped) * scaler_reshaped
            self.weight_profiler.update_weight(self.weight, dequant_weight.detach())

        return (self.scaler, self.zero), (self.output_scaler, self.output_zero)

    def get_profiler(self):
        """Get profiler objects (output and weight)"""
        if self.enable_profiling:
            return {
                'output': self.profiler,
                'weight': self.weight_profiler
            }
        return None

    def get_profiling_results(self):
        """
        Get all profiling results after forward pass.

        Returns:
            dict: Dictionary containing statistics, histogram, time, and memory records
                  for both output and weight quantization
                  Returns None if profiling is not enabled

        Usage:
            layer.forward(x)  # Automatically updates profiler
            results = layer.get_profiling_results()
            print(results['output']['statistics']['qsnr'])
            print(results['weight']['statistics']['qsnr'])
        """
        if not self.enable_profiling:
            return None

        results = {}

        # Output profiling results
        if self.profiler is not None:
            try:
                output_stats = self.profiler.get_statistic() if self.profiler.weight is not None else None
                output_hist = self.profiler.get_hist() if self.profiler.weight is not None else None
            except (ValueError, AttributeError):
                output_stats = None
                output_hist = None

            results['output'] = {
                'statistics': output_stats,
                'histogram': output_hist,
                'time': self.profiler.get_time_record(),
                'memory': self.profiler.get_memory_record()
            }

        # Weight profiling results
        if self.weight_profiler is not None:
            try:
                weight_stats = self.weight_profiler.get_statistic() if self.weight_profiler.weight is not None else None
                weight_hist = self.weight_profiler.get_hist() if self.weight_profiler.weight is not None else None
            except (ValueError, AttributeError):
                weight_stats = None
                weight_hist = None

            results['weight'] = {
                'statistics': weight_stats,
                'histogram': weight_hist,
                'time': self.weight_profiler.get_time_record(),
                'memory': self.weight_profiler.get_memory_record()
            }

        return results

    def reset_profiling(self):
        """Reset profiling data for both output and weight"""
        if self.enable_profiling:
            if self.profiler is not None:
                self.profiler.reset_time_profiler()
                self.profiler.reset_memory_profiler()
                self.profiler.weight = None
                self.profiler.quant_weight = None

            if self.weight_profiler is not None:
                self.weight_profiler.reset_time_profiler()
                self.weight_profiler.reset_memory_profiler()
                self.weight_profiler.weight = None
                self.weight_profiler.quant_weight = None


    def forward(self, x):
        # in inference x input is int8 tensor

        if self.mode == 'quantized':
            if self.enable_profiling and self.profiler is not None:
                # Measure quantization time
                with self.profiler.measure_time():
                    # 1. dequantize weights (int8 -> fp32)
                    dequant_weight = self.quantizer.forward(self.quant_weight)

                    # 2. Linear operation in fp32
                    out_fp32 = self.fwd_func(x, dequant_weight, self.bias, **self.fwd_kwargs)

                    # Store FP32 output before quantization
                    fp32_output = out_fp32.clone().detach()

                    # 3. Output fake quantization
                    x = self.output_quantizer.forward(out_fp32)

                    # Update profiler with FP32 vs Quantized outputs
                    self.profiler.update_weight(fp32_output, x.detach())
            else:
                # 1. dequantize weights (int8 -> fp32)
                dequant_weight = self.quantizer.forward(self.quant_weight)

                # 2. fake quantization in fp32
                x = self.fwd_func(x, dequant_weight, self.bias, **self.fwd_kwargs)

                # 3. Output fake quantization
                x = self.output_quantizer.forward(x)

            return x

        else:  # fp32
            if self.enable_profiling and self.profiler is not None:
                # FP32 mode - measure time without quantization
                with self.profiler.measure_time():
                    pass  # No quantization, just time measurement

            return self.fwd_func(x, self.weight, self.bias, **self.fwd_kwargs)


def test_quantlinear_profiling():
    """
    Test QuantLinear profiling functionality.

    Tests:
    1. Forward pass automatically updates profiler
    2. get_profiling_results() returns correct data for both weight and output
    3. Statistics, histogram, and timing info are collected
    """
    print("="*80)
    print("QuantLinear Profiling Test")
    print("="*80)

    # Create test Linear module
    print("\n[1] Creating Linear module")
    in_features = 768
    out_features = 3072  # MLP expansion
    original_linear = nn.Linear(in_features, out_features)
    print(f"  ✓ Original Linear created ({in_features} -> {out_features})")

    # Create test config
    config = QuantConfig(
        bit_type=BitTypeConfig(bits=8, signed=True, name='int8'),
        observer_type='PercentileObserver',
        calibration_mode='channel_wise'
    )

    # Create QuantLinear with profiling enabled
    print("\n[2] Creating QuantLinear with profiling enabled")
    layer = QuantLinear(
        input_module=original_linear,
        quant_config=config,
        layer_name='test_qlinear',
        enable_profiling=True
    )
    print("  ✓ QuantLinear created")
    print(f"  Layer name: {layer.layer_name}")
    print(f"  Profiling enabled: {layer.enable_profiling}")
    print(f"  Output profiler: {layer.profiler is not None}")
    print(f"  Weight profiler: {layer.weight_profiler is not None}")
    print(f"  Weight shape: {layer.weight.shape}")

    # Generate test data
    print("\n[3] Generating calibration data")
    batch_size = 8
    seq_len = 197
    num_calib_batches = 10

    calib_data = [torch.randn(batch_size, seq_len, in_features) for _ in range(num_calib_batches)]
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

    # Compute quantization parameters
    print("\n[5] Computing quantization parameters")
    (weight_scaler, weight_zero), (output_scaler, output_zero) = layer.compute_quant_params()
    print(f"  Weight scale shape: {weight_scaler.shape if isinstance(weight_scaler, torch.Tensor) else 'scalar'}")
    print(f"  Weight scale (first 5): {weight_scaler[:5] if isinstance(weight_scaler, torch.Tensor) and weight_scaler.numel() > 5 else weight_scaler}")
    print(f"  Output scale shape: {output_scaler.shape if isinstance(output_scaler, torch.Tensor) else 'scalar'}")
    print(f"  Quantized weight shape: {layer.quant_weight.shape}")
    print("  ✓ Quantization parameters computed")

    # Check if weight profiler was updated
    print("\n[5.1] Checking weight profiler after compute_quant_params")
    if layer.weight_profiler is not None:
        has_weight = layer.weight_profiler.weight is not None
        has_quant_weight = layer.weight_profiler.quant_weight is not None
        print(f"  Weight profiler has weight: {has_weight}")
        print(f"  Weight profiler has quant_weight: {has_quant_weight}")
        if has_weight:
            weight_results = layer.get_profiling_results()
            if weight_results and 'weight' in weight_results:
                w_stats = weight_results['weight']['statistics']
                if w_stats:
                    print(f"  Weight QSNR: {w_stats.get('qsnr', 'N/A'):.2f} dB")
                    print(f"  Weight MSE: {w_stats.get('mse', 'N/A'):.10f}")
    else:
        print("  ✗ Weight profiler is None")

    # Save weight profiling results before reset
    print("\n[5.2] Saving weight profiling results")
    weight_profiling_results = layer.get_profiling_results()
    print("  ✓ Weight profiling results saved")

    # Test data
    test_input = torch.randn(1, seq_len, in_features)

    # FP32 inference
    print("\n[6] Testing FP32 mode (no quantization)")
    layer.mode = 'fp32'
    # Don't reset profiling here - keep weight data
    # Only reset output profiler
    if layer.profiler is not None:
        layer.profiler.reset_time_profiler()
        layer.profiler.reset_memory_profiler()
        layer.profiler.weight = None
        layer.profiler.quant_weight = None

    with torch.no_grad():
        for _ in range(50):
            _ = layer(test_input)

    results_fp32 = layer.get_profiling_results()
    print("  ✓ FP32 forward completed (50 iterations)")

    if results_fp32 and 'output' in results_fp32 and results_fp32['output']['time']:
        time_data = results_fp32['output']['time']
        if 'test_qlinear_output' in time_data:
            stats = time_data['test_qlinear_output']
            print(f"  Timing - Count: {stats['count']}, Mean: {stats['mean']*1000:.4f} ms")
        else:
            print("  Warning: No timing data for 'test_qlinear_output'")
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

    # Check weight statistics (from saved results after compute_quant_params)
    print("\n[9] Weight Statistics (from compute_quant_params)")
    weight_qsnr_pass = False
    if weight_profiling_results and 'weight' in weight_profiling_results and weight_profiling_results['weight']['statistics'] is not None:
        stats = weight_profiling_results['weight']['statistics']
        print("  ✓ Weight statistics available")
        qsnr = stats.get('qsnr', 'N/A')
        mse = stats.get('mse', 'N/A')

        print(f"    QSNR: {qsnr:.2f} dB" if isinstance(qsnr, (int, float)) else f"    QSNR: {qsnr}")
        print(f"    MSE: {mse:.10f}" if isinstance(mse, (int, float)) else f"    MSE: {mse}")

        # Quality check
        if isinstance(qsnr, (int, float)):
            if qsnr >= 30:
                print("    ✓ Quality: EXCELLENT (QSNR >= 30 dB)")
                weight_qsnr_pass = True
            elif qsnr >= 20:
                print("    ⚠ Quality: GOOD (QSNR >= 20 dB)")
                weight_qsnr_pass = True
            elif qsnr >= 10:
                print("    ⚠ Quality: ACCEPTABLE (QSNR >= 10 dB)")
                weight_qsnr_pass = True
            else:
                print("    ✗ Quality: POOR (QSNR < 10 dB) - Quantization may be incorrect!")
    else:
        print("  ✗ Weight statistics: None")

    # Check weight histogram (from saved results after compute_quant_params)
    print("\n[10] Weight Histogram (from compute_quant_params)")
    if weight_profiling_results and 'weight' in weight_profiling_results and weight_profiling_results['weight']['histogram'] is not None:
        hist = weight_profiling_results['weight']['histogram']
        print("  ✓ Weight histogram available")
        if 'kl_divergence' in hist:
            print(f"    KL Divergence: {hist['kl_divergence']:.6f}")
    else:
        print("  ✗ Weight histogram: None")

    # Check output statistics
    print("\n[11] Output Statistics")
    if 'output' in results and results['output']['statistics'] is not None:
        stats = results['output']['statistics']
        print("  ✓ Output statistics available")
        qsnr = stats.get('qsnr', 'N/A')
        mse = stats.get('mse', 'N/A')

        print(f"    QSNR: {qsnr:.2f} dB" if isinstance(qsnr, (int, float)) else f"    QSNR: {qsnr}")
        print(f"    MSE: {mse:.10f}" if isinstance(mse, (int, float)) else f"    MSE: {mse}")
    else:
        print("  ✗ Output statistics: None")

    # Check output histogram
    print("\n[12] Output Histogram")
    if 'output' in results and results['output']['histogram'] is not None:
        hist = results['output']['histogram']
        print("  ✓ Output histogram available")
        if 'kl_divergence' in hist:
            print(f"    KL Divergence: {hist['kl_divergence']:.6f}")
    else:
        print("  ✗ Output histogram: None")

    # Check timing
    print("\n[13] Timing Information")
    if 'output' in results:
        time_data = results['output']['time']
        if time_data and 'test_qlinear_output' in time_data:
            time_stats = time_data['test_qlinear_output']
            print("  ✓ Timing data available")
            print(f"    Count: {time_stats['count']}")
            print(f"    Mean: {time_stats['mean']*1000:.4f} ms")
            print(f"    Min: {time_stats['min']*1000:.4f} ms")
            print(f"    Max: {time_stats['max']*1000:.4f} ms")
            print(f"    Total: {time_stats['total']*1000:.4f} ms")
        else:
            print("  ✗ Timing: No data for 'test_qlinear_output'")

    # Check memory
    print("\n[14] Memory Information")
    if 'output' in results:
        mem_data = results['output']['memory']
        if mem_data:
            print(f"  Memory profiler data: {mem_data}")
        else:
            print("  Memory: No data (expected if not attached)")

    # Verify profiler objects
    print("\n[15] Verifying profiler objects")
    prof_dict = layer.get_profiler()
    if prof_dict is not None:
        print("  ✓ get_profiler() returns profiler dictionary")
        print(f"    Output profiler name: {prof_dict['output'].name}")
        print(f"    Weight profiler name: {prof_dict['weight'].name}")
        print(f"    Output has data: {prof_dict['output'].weight is not None}")
        print(f"    Weight has data: {prof_dict['weight'].weight is not None}")
    else:
        print("  ✗ get_profiler() returned None")

    # Test reset
    print("\n[16] Testing reset_profiling()")
    layer.reset_profiling()
    print("  ✓ reset_profiling() called")

    # Verify reset
    prof_dict_after = layer.get_profiler()
    if prof_dict_after is not None:
        print(f"    Output weight after reset: {prof_dict_after['output'].weight}")
        print(f"    Weight weight after reset: {prof_dict_after['weight'].weight}")

    # Summary
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)

    passed_tests = []
    failed_tests = []

    # Check critical functionality
    if layer.profiler is not None and layer.weight_profiler is not None:
        passed_tests.append("Profiler initialization (output + weight)")
    else:
        failed_tests.append("Profiler initialization")

    if results is not None:
        passed_tests.append("get_profiling_results() returns data")
    else:
        failed_tests.append("get_profiling_results() returns data")

    if weight_profiling_results and 'weight' in weight_profiling_results and weight_profiling_results['weight']['statistics'] is not None:
        passed_tests.append("Weight statistics collection")
        if weight_qsnr_pass:
            passed_tests.append("Weight quantization quality (QSNR >= 10 dB)")
        else:
            failed_tests.append("Weight quantization quality (QSNR < 10 dB)")
    else:
        failed_tests.append("Weight statistics collection")

    if results and 'output' in results and results['output']['statistics'] is not None:
        passed_tests.append("Output statistics collection")
    else:
        failed_tests.append("Output statistics collection")

    if weight_profiling_results and 'weight' in weight_profiling_results and weight_profiling_results['weight']['histogram'] is not None:
        passed_tests.append("Weight histogram generation")
    else:
        failed_tests.append("Weight histogram generation")

    if results and 'output' in results and results['output']['histogram'] is not None:
        passed_tests.append("Output histogram generation")
    else:
        failed_tests.append("Output histogram generation")

    if results and 'output' in results and results['output']['time'] and 'test_qlinear_output' in results['output']['time']:
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

    test_quantlinear_profiling()
