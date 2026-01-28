import torch
import torch.nn as nn

from quant_config import QuantConfig, BitTypeConfig
from .bit_type import BitType
from .utils import init_observers
from .layer_quantizer.log2 import Log2Quantizer
from .layer_profiler.profiler import profiler

class QuantIntSoft(nn.Module):

    def __init__(self,
                 input_module: nn.Module,
                 quant_config: QuantConfig,
                 layer_name: str = 'qintsoft'):
        super(QuantIntSoft, self).__init__()

        # Config 설정
        self.layer_name = layer_name
        self.observer_config = quant_config
        self.enable_profiling = quant_config.enable_profiler
        self.observer_type = quant_config.observer_type
        self.bit_type = BitType(
            bits=quant_config.bit_type.bits,
            symmetric=quant_config.bit_type.symmetric,
            name=quant_config.bit_type.name
        )
        self.calibration_mode = quant_config.calibration_mode
        self.mode = 'fp32'

        self.PTQ = True  # PTQ or I-bert

        # Observer와 Log2Quantizer 초기화
        self.observer = init_observers(
            observer_type=self.observer_type,
            bit_type=self.bit_type,
            module_type='activation',
            calibration_mode=self.calibration_mode,
            quant_config=quant_config
        )
        self.quantizer = Log2Quantizer(
            bit_type=self.bit_type,
            module_type='activation'
        )

        #1. profiler 초기화
        self.profiler = None
        if self.enable_profiling:
            self.profiler = profiler(layer_name)

        # I-BERT Integer Softmax용 scale (propagation으로 전달받음)
        # register_buffer로 ONNX initializer로 인식되게 함
        self.register_buffer('input_scale', None)
        self.register_buffer('output_scale', None)

        # Quantization parameters
        self.register_buffer('scaler', None)
        self.register_buffer('zero', None)

    def calibration(self, x, input_scale=None):
        """
        Calibration 전용 메서드
        - I-BERT 모드: scale propagation 정보만 저장 (observer 사용 안 함)
        - PTQ 모드: observer update 수행
        """
        with torch.no_grad():
            x = x.softmax(dim=-1)

            if self.PTQ:
                # PTQ 모드: observer 사용
                self.observer.update(x)
            # I-BERT 모드: observer 사용 안 함 (input_scale만 저장)

        return x
    

    def compute_quant_params(self):
        """
        Calibration 끝나고 한 번 호출
        - I-BERT 모드: output_scale 계산 (수식으로)
        - PTQ fallback 모드: observer에서 scale/zero 추출
        """
        if self.PTQ is False:
            # I-BERT 모드: output scale 계산
            # Softmax output scale = 0.3585 * input_scale^2 / 2^30
            coef = 0.35815147
            n = 30
            self.output_scale = (coef * self.input_scale ** 2) / (2 ** n)

            return (self.output_scale, None)
        else:
            # PTQ fallback 모드
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

    @staticmethod
    def log_round(x):
        x_log_floor = x.log2().floor()
        big = x_log_floor
        extra_mask = (x - 2**big) >= 2**(big - 1)
        big[extra_mask] = big[extra_mask] + 1
        return big

    @staticmethod
    def int_polynomial(x_int, scaling_factor):
        coef = [0.35815147, 0.96963238, 1.]  # ax**2 + bx + c
        coef[1] /= coef[0]
        coef[2] /= coef[0]
        b_int = torch.floor(torch.tensor(coef[1]) / scaling_factor)
        c_int = torch.floor(torch.tensor(coef[2]) / scaling_factor**2)
        
        z = x_int + b_int
        z = x_int * z
        z = z + c_int
        scaling_factor = coef[0] * scaling_factor**2
        return z, scaling_factor

    @staticmethod
    def int_exp(x_int, scaling_factor):

        x0 = -0.6931  # -ln2
        n = 30  # sufficiently large integer

        x0_int = torch.floor(torch.tensor(x0) / scaling_factor) # ln2 / S
        x_int = torch.max(x_int, n * x0_int) #[-1,0]의 공간에 X를 맞추는 작업진행

        q = torch.floor(x_int / x0_int) ## 정수부 ㅇ,.ㅇ
        r = x_int - x0_int * q ## 소수부로 분해 [-1,0]

        exp_int, exp_scaling_factor = QuantIntSoft.int_polynomial(r, scaling_factor)
        exp_int = torch.clamp(torch.floor(exp_int * 2**(n - q)), min=0)
        scaling_factor = exp_scaling_factor / 2**n
        return exp_int, scaling_factor

    @staticmethod
    def int_softmax(x, scaling_factor):
        scaling_factor = scaling_factor.to(x.device)
        x_int = x / scaling_factor

        #for numerical stability
        x_int_max, _ = x_int.max(dim=-1, keepdim=True)
        x_int = x_int - x_int_max

        exp_int, exp_scaling_factor = QuantIntSoft.int_exp(x_int, scaling_factor)
        exp_int_sum = exp_int.sum(dim=-1, keepdim=True)
        return exp_int, exp_int_sum

    def forward(self, x, scale=None):
        """
        Forward pass
        Args:
            x: input tensor
            scale: input scale (I-BERT 모드에서 이전 layer로부터 전달받음)
                   channel-wise scale (tensor)이 들어오면 scalar로 변환
        Returns:
            output tensor
        """
        if self.mode == 'quantized':

            if self.PTQ is False:
                # I-BERT 모드: Integer Softmax
                if self.enable_profiling and self.profiler is not None:
                    with self.profiler.measure_time():
                        # FP32 softmax for comparison
                        fp32_output = x.softmax(dim=-1)

                        # scale이 None이면 self.input_scale 사용
                        if scale is None:
                            scale = self.input_scale

                        # channel-wise scale (tensor)이면 scalar로 변환 (max 사용)
                        if isinstance(scale, torch.Tensor) and scale.numel() > 1:
                            scale = scale.max()
                        elif isinstance(scale, torch.Tensor):
                            scale = scale.item()

                        # scalar를 tensor로 변환
                        if not isinstance(scale, torch.Tensor):
                            scale = torch.tensor(scale)

                        exp_int, exp_int_sum = self.int_softmax(x, scale)
                        softmax_out = torch.round(exp_int_sum / exp_int)
                        rounds = self.log_round(softmax_out)

                        qlog = torch.clamp(rounds, 0, 2**self.bit_type.bits - 1)
                        deq_softmax = 2**(-qlog)
                        deq_softmax = deq_softmax / deq_softmax.sum(dim=-1, keepdim=True)

                        # Update profiler with FP32 vs Integer Softmax
                        self.profiler.update_weight(fp32_output, deq_softmax.detach())

                        return deq_softmax
                else:
                    # scale이 None이면 self.input_scale 사용
                    if scale is None:
                        scale = self.input_scale

                    # channel-wise scale (tensor)이면 scalar로 변환 (max 사용)
                    if isinstance(scale, torch.Tensor) and scale.numel() > 1:
                        scale = scale.max()
                    elif isinstance(scale, torch.Tensor):
                        scale = scale.item()

                    # scalar를 tensor로 변환
                    if not isinstance(scale, torch.Tensor):
                        scale = torch.tensor(scale)

                    exp_int, exp_int_sum = self.int_softmax(x, scale)
                    softmax_out = torch.round(exp_int_sum / exp_int)
                    rounds = self.log_round(softmax_out)

                    qlog = torch.clamp(rounds, 0, 2**self.bit_type.bits - 1)
                    deq_softmax = 2**(-qlog)
                    deq_softmax = deq_softmax / deq_softmax.sum(dim=-1, keepdim=True)

                    return deq_softmax
            else:
                # PTQ 모드: Log2 Quantizer 사용
                if self.enable_profiling and self.profiler is not None:
                    with self.profiler.measure_time():
                        # FP32 softmax
                        fp32_output = x.softmax(dim=-1)

                        # Observer 기반 clipping으로 0 값 방지
                        # Log2는 (0, 1] 범위에서 동작, 너무 작은 값은 0으로 매핑됨
                        min_val = 2 ** (-(2 ** self.bit_type.bits - 1))  # 2^(-255) for 8-bit
                        x_clipped = torch.clamp(fp32_output, min=min_val, max=1.0)

                        # Log2 quantization
                        x_quant = self.quantizer.forward(x_clipped)
                        x_quant = x_quant / (x_quant.sum(dim=-1, keepdim=True) + 1e-8)

                        # Update profiler with FP32 vs Log2 Quantized
                        self.profiler.update_weight(fp32_output, x_quant.detach())

                        return x_quant
                else:
                    x = x.softmax(dim=-1)

                    # Observer 기반 clipping으로 0 값 방지
                    min_val = 2 ** (-(2 ** self.bit_type.bits - 1))  # 2^(-255) for 8-bit
                    x = torch.clamp(x, min=min_val, max=1.0)

                    x = self.quantizer.forward(x)
                    x = x / (x.sum(dim=-1, keepdim=True) + 1e-8)

                    return x

        else:  # fp32 모드
            if self.enable_profiling and self.profiler is not None:
                # FP32 mode - measure time without quantization
                with self.profiler.measure_time():
                    pass  # No quantization, just time measurement

            x = x.softmax(dim=-1)
            return x


def test_quantintsoft_profiling():
    """
    Test QuantIntSoft profiling functionality.

    Tests:
    1. PTQ mode with Log2 quantizer
    2. I-BERT mode with Integer Softmax
    3. Profiling for both modes
    """
    print("="*80)
    print("QuantIntSoft Profiling Test")
    print("="*80)

    # Create test config
    config = QuantConfig(
        bit_type=BitTypeConfig(bits=8, symmetric=True, name='int8'),
        observer_type='MinmaxObserver',
        calibration_mode='layer_wise',
        enable_profiler=True
    )

    # Test PTQ Mode
    print("\n" + "="*80)
    print("Test 1: PTQ Mode (Log2 Quantizer)")
    print("="*80)

    # Create QuantIntSoft with profiling enabled
    print("\n[1] Creating QuantIntSoft (PTQ mode) with profiling enabled")
    layer_ptq = QuantIntSoft(
        input_module=None,
        quant_config=config,
        layer_name='test_intsoft_ptq'
    )
    layer_ptq.PTQ = True
    print("  ✓ QuantIntSoft created")
    print(f"  Layer name: {layer_ptq.layer_name}")
    print(f"  Mode: PTQ (Log2)")
    print(f"  Profiling enabled: {layer_ptq.enable_profiling}")
    print(f"  Profiler object: {layer_ptq.profiler is not None}")

    # Generate test data (attention scores before softmax)
    print("\n[2] Generating calibration data")
    batch_size = 8
    num_heads = 12
    seq_len = 197
    num_calib_batches = 10

    calib_data = [torch.randn(batch_size, num_heads, seq_len, seq_len) for _ in range(num_calib_batches)]
    print(f"  Calibration batches: {num_calib_batches}")
    print(f"  Batch shape: {calib_data[0].shape}")

    # Calibration
    print("\n[3] Running calibration (PTQ mode)")
    layer_ptq.eval()
    with torch.no_grad():
        for idx, x in enumerate(calib_data):
            _ = layer_ptq.calibration(x)
            if (idx + 1) % 5 == 0:
                print(f"  Batch {idx + 1}/{num_calib_batches} completed")
    print("  ✓ Calibration completed")

    # Compute quantization parameters
    print("\n[4] Computing quantization parameters")
    scaler, zero = layer_ptq.compute_quant_params()
    print(f"  Scale shape: {scaler.shape if isinstance(scaler, torch.Tensor) else 'scalar'}")
    print(f"  Zero shape: {zero.shape if isinstance(zero, torch.Tensor) else 'scalar'}")
    print("  ✓ Quantization parameters computed")

    # Test data
    test_input = torch.randn(1, num_heads, seq_len, seq_len)

    # Quantized inference (PTQ mode)
    print("\n[5] Testing Quantized mode (PTQ with profiling)")
    layer_ptq.mode = 'quantized'
    layer_ptq.reset_profiling()

    print("  Running 100 iterations...")
    with torch.no_grad():
        for _ in range(100):
            output_quant = layer_ptq(test_input)

    print("  ✓ Quantized forward completed (100 iterations)")
    print(f"  Output shape: {output_quant.shape}")
    print(f"  Output range: [{output_quant.min():.6f}, {output_quant.max():.6f}]")
    print(f"  Output sum (per sequence): {output_quant.sum(dim=-1).mean():.6f}")

    # Get profiling results
    print("\n[6] Retrieving profiling results (PTQ mode)")
    results_ptq = layer_ptq.get_profiling_results()

    if results_ptq is None:
        print("  ✗ ERROR: get_profiling_results() returned None!")
    else:
        print("  ✓ Profiling results retrieved")
        print(f"  Result keys: {list(results_ptq.keys())}")

        # Check statistics
        if results_ptq['statistics'] is not None:
            stats = results_ptq['statistics']
            print("\n  Statistics:")
            qsnr = stats.get('qsnr', 'N/A')
            mse = stats.get('mse', 'N/A')
            print(f"    QSNR: {qsnr:.2f} dB" if isinstance(qsnr, (int, float)) else "    QSNR: {}".format(qsnr))
            print(f"    MSE: {mse:.10f}" if isinstance(mse, (int, float)) else "    MSE: {}".format(mse))

        # Check timing
        if results_ptq['time'] and 'test_intsoft_ptq' in results_ptq['time']:
            time_stats = results_ptq['time']['test_intsoft_ptq']
            print("\n  Timing:")
            print(f"    Count: {time_stats['count']}")
            print(f"    Mean: {time_stats['mean']*1000:.4f} ms")

    # Test I-BERT Mode
    print("\n" + "="*80)
    print("Test 2: I-BERT Mode (Integer Softmax)")
    print("="*80)

    # Create QuantIntSoft for I-BERT mode
    print("\n[7] Creating QuantIntSoft (I-BERT mode) with profiling enabled")
    layer_ibert = QuantIntSoft(
        input_module=None,
        quant_config=config,
        layer_name='test_intsoft_ibert'
    )
    layer_ibert.PTQ = False
    layer_ibert.input_scale = torch.tensor(0.1)  # Simulated input scale
    print("  ✓ QuantIntSoft created")
    print(f"  Mode: I-BERT (Integer Softmax)")
    print(f"  Input scale: {layer_ibert.input_scale}")

    # Compute quantization parameters (I-BERT mode)
    print("\n[8] Computing quantization parameters (I-BERT mode)")
    output_scale, _ = layer_ibert.compute_quant_params()
    print(f"  Output scale: {output_scale}")
    print("  ✓ Quantization parameters computed")

    # Quantized inference (I-BERT mode)
    print("\n[9] Testing Quantized mode (I-BERT with profiling)")
    layer_ibert.mode = 'quantized'
    layer_ibert.reset_profiling()

    print("  Running 100 iterations...")
    with torch.no_grad():
        for _ in range(100):
            output_ibert = layer_ibert(test_input, scale=layer_ibert.input_scale)

    print("  ✓ Quantized forward completed (100 iterations)")
    print(f"  Output shape: {output_ibert.shape}")
    print(f"  Output range: [{output_ibert.min():.6f}, {output_ibert.max():.6f}]")
    print(f"  Output sum (per sequence): {output_ibert.sum(dim=-1).mean():.6f}")

    # Get profiling results
    print("\n[10] Retrieving profiling results (I-BERT mode)")
    results_ibert = layer_ibert.get_profiling_results()

    if results_ibert is None:
        print("  ✗ ERROR: get_profiling_results() returned None!")
    else:
        print("  ✓ Profiling results retrieved")

        # Check statistics
        if results_ibert['statistics'] is not None:
            stats = results_ibert['statistics']
            print("\n  Statistics:")
            qsnr = stats.get('qsnr', 'N/A')
            mse = stats.get('mse', 'N/A')
            print(f"    QSNR: {qsnr:.2f} dB" if isinstance(qsnr, (int, float)) else "    QSNR: {}".format(qsnr))
            print(f"    MSE: {mse:.10f}" if isinstance(mse, (int, float)) else "    MSE: {}".format(mse))

        # Check timing
        if results_ibert['time'] and 'test_intsoft_ibert' in results_ibert['time']:
            time_stats = results_ibert['time']['test_intsoft_ibert']
            print("\n  Timing:")
            print(f"    Count: {time_stats['count']}")
            print(f"    Mean: {time_stats['mean']*1000:.4f} ms")

    # Summary
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)

    passed_tests = []
    failed_tests = []

    # PTQ mode tests
    if layer_ptq.profiler is not None:
        passed_tests.append("PTQ: Profiler initialization")
    else:
        failed_tests.append("PTQ: Profiler initialization")

    if results_ptq and results_ptq['statistics'] is not None:
        passed_tests.append("PTQ: Statistics collection")
    else:
        failed_tests.append("PTQ: Statistics collection")

    if results_ptq and results_ptq['time'] and 'test_intsoft_ptq' in results_ptq['time']:
        passed_tests.append("PTQ: Timing profiling")
    else:
        failed_tests.append("PTQ: Timing profiling")

    # I-BERT mode tests
    if layer_ibert.profiler is not None:
        passed_tests.append("I-BERT: Profiler initialization")
    else:
        failed_tests.append("I-BERT: Profiler initialization")

    if results_ibert and results_ibert['statistics'] is not None:
        passed_tests.append("I-BERT: Statistics collection")
    else:
        failed_tests.append("I-BERT: Statistics collection")

    if results_ibert and results_ibert['time'] and 'test_intsoft_ibert' in results_ibert['time']:
        passed_tests.append("I-BERT: Timing profiling")
    else:
        failed_tests.append("I-BERT: Timing profiling")

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

    test_quantintsoft_profiling()

