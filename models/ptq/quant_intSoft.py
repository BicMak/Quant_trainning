import torch
import torch.nn as nn

from quant_config import QuantConfig, BitTypeConfig
from .bit_type import BitType

class QuantIntSoft(nn.Module):

    def __init__(self, 
                 input_module:nn.Module,
                 quant_config:QuantConfig):
        super(QuantIntSoft, self).__init__()

        #0. observer config copy
        self.observer_config = quant_config
        self.observer_type = quant_config.observer_type
        self.bit_type = BitType(
            bits=quant_config.bit_type.bits,
            signed=quant_config.bit_type.signed,
            name=quant_config.bit_type.name
        )
        self.calibration_mode = quant_config.calibration_mode
        self.mode = 'fp32'



        # I-BERT Integer Softmax용 scale (propagation으로 전달받음)
        self.input_scale = None
        self.output_scale = None

    def calibration(self, x, input_scale=None):
        """
        Calibration 전용 메서드
        - I-BERT 모드: scale propagation 정보만 저장 (observer 사용 안 함)
        - PTQ fallback 모드: observer update 수행
        """
        return x

    def compute_quant_params(self):
        """
        Calibration 끝나고 한 번 호출
        - I-BERT 모드: output_scale 계산 (수식으로)
        - PTQ fallback 모드: observer에서 scale/zero 추출
        """
        if self.input_scale is not None:
            # I-BERT 모드: output scale 계산
            # Softmax output scale = 0.3585 * input_scale^2 / 2^30
            coef = 0.35815147
            n = 30
            self.output_scale = (coef * self.input_scale ** 2) / (2 ** n)

            return (self.output_scale, None)
        else:
            # PTQ fallback 모드
            self.scaler, self.zero = 1, 0 
            # 매서드 일관성을 위해서 임의이값을 넣어줌

            return (self.scaler, self.zero)

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
            
            mask = rounds >= 2**self.bit_type.bits
            qlog = torch.clamp(rounds, 0, 2**self.bit_type.bits - 1)
            deq_softmax = 2**(-qlog)
            deq_softmax[mask] = 0

            return deq_softmax


        else:  # fp32 모드
            x = x.softmax(dim=-1)
            return x


def main():
    """
    I-BERT Integer Softmax 테스트 함수
    """
    print("=" * 60)
    print("I-BERT Integer Softmax Test")
    print("=" * 60)

    # 1. Observer co    nfig 생성
    bit_type_config = BitTypeConfig(bits=8, signed=False, name='uint8')
    observer_config = QuantConfig(
        observer_type='PercentileObserver',
        bit_type=bit_type_config,
        calibration_mode='layer_wise'
    )

    # 2. QuantIntSoft 레이어 생성
    quant_softmax = QuantIntSoft(
        quant_args={},
        input_module=None,
        observer_config=observer_config
    )

    print(f"\n[Init] mode: {quant_softmax.mode}")
    print(f"[Init] bit_type: {quant_softmax.bit_type.bits}-bit")

    # 3. 테스트 입력 생성 (Attention logits 시뮬레이션)
    batch_size, seq_len, hidden_dim = 2, 4, 8
    x = torch.randn(batch_size, seq_len, hidden_dim) * 10  # [-30, 30] 범위

    print(f"\n[Input] shape: {x.shape}")
    print(f"[Input] range: [{x.min():.2f}, {x.max():.2f}]")

    # 4. FP32 모드 테스트
    print("\n" + "-" * 60)
    print("FP32 Mode Test")
    print("-" * 60)

    quant_softmax.mode = 'fp32'
    output_fp32 = quant_softmax(x)

    print(f"[Output] shape: {output_fp32.shape}")
    print(f"[Output] range: [{output_fp32.min():.6f}, {output_fp32.max():.6f}]")
    print(f"[Output] sum (should be ~1.0): {output_fp32.sum(dim=-1)[0, 0]:.6f}")

    # 5. Quantized 모드 테스트
    print("\n" + "-" * 60)
    print("Quantized Mode Test (I-BERT)")
    print("-" * 60)

    # Scale 설정 (이전 MatMul layer로부터 전달받았다고 가정)
    input_scale = torch.tensor(0.1)
    quant_softmax.input_scale = input_scale
    quant_softmax.compute_quant_params()

    print(f"[Scale] input_scale: {quant_softmax.input_scale:.6f}")
    print(f"[Scale] output_scale: {quant_softmax.output_scale:.10f}")

    quant_softmax.mode = 'quantized'
    output_quant = quant_softmax(x, scale=input_scale)

    print(f"[Output] shape: {output_quant.shape}")
    print(f"[Output] range: [{output_quant.min():.6f}, {output_quant.max():.6f}]")
    print(f"[Output] sum (should be ~1.0): {output_quant.sum(dim=-1)[0, 0]:.6f}")

    # 6. FP32 vs Quantized 비교
    print("\n" + "-" * 60)
    print("FP32 vs Quantized Comparison")
    print("-" * 60)

    diff = (output_fp32 - output_quant).abs()
    print(f"[Diff] mean: {diff.mean():.6f}")
    print(f"[Diff] max: {diff.max():.6f}")
    print(f"[Diff] relative error: {(diff / (output_fp32 + 1e-8)).mean():.6f}")

    # 7. Static method 독립 테스트
    print("\n" + "-" * 60)
    print("Static Method Test")
    print("-" * 60)

    # int_polynomial 테스트
    x_test = torch.tensor([0.5, -0.3, -0.1])
    scaling_factor = torch.tensor(0.01)
    poly_out, poly_scale = QuantIntSoft.int_polynomial(x_test, scaling_factor)
    print(f"[int_polynomial] input: {x_test}")
    print(f"[int_polynomial] output: {poly_out}")
    print(f"[int_polynomial] scale: {poly_scale:.6f}")

    # int_exp 테스트
    exp_out, exp_scale = QuantIntSoft.int_exp(x_test, scaling_factor)
    print(f"[int_exp] output: {exp_out}")
    print(f"[int_exp] scale: {exp_scale:.10f}")

    # log_round 테스트
    log_test = torch.tensor([0.9, 0.5, 0.1, 0.01])
    log_out = QuantIntSoft.log_round(log_test)
    print(f"[log_round] input: {log_test}")
    print(f"[log_round] output: {log_out}")
    print(f"[log_round] dequant: {2**(-log_out)}")

    print("\n" + "=" * 60)
    print("Test Completed!")
    print("=" * 60)

def test_full_attention_pipeline(observer_type='PercentileObserver'):
    """
    전체 I-BERT Attention 파이프라인 테스트
    QuantLinear → QuantAct → QuantIntSoft → Value MatMul
    """
    print(f"\n{'='*70}")
    print(f"Full I-BERT Attention Pipeline Test with {observer_type}")
    print(f"{'='*70}")

    from models.ptq.quant_layer import QuantLinear
    from models.ptq.quant_act import QAct
    import torch.nn.functional as F

    # ========== 1. Config 설정 ==========
    bit_type_config = BitTypeConfig(bits=8, signed=True, name='int8')
    observer_config = QuantConfig(
        observer_type=observer_type,
        bit_type=bit_type_config,
        calibration_mode='layer_wise'
    )

    # ========== 2. 입력 데이터 설정 ==========
    batch_size, seq_len, hidden_dim = 2, 4, 64
    scale_factor = hidden_dim ** 0.5

    print(f"\n[Setup] Batch={batch_size}, Seq={seq_len}, Hidden={hidden_dim}")
    print(f"[Setup] Scale factor (√d): {scale_factor:.4f}")

    # ========== 3. Calibration Data 생성 및 Calibration ==========
    num_batches = 32
    print(f"\n=== Calibration Phase ({num_batches} batches) ===")

    # QAct for MatMul output
    qact_matmul = QAct(quant_args={}, observer_config=observer_config)

    # QuantIntSoft for Softmax
    quant_softmax = QuantIntSoft(
        quant_args={},
        input_module=None,
        observer_config=observer_config
    )

    # Calibration loop
    for i in range(num_batches):
        Q_calib = torch.randn(batch_size, seq_len, hidden_dim)
        K_calib = torch.randn(batch_size, seq_len, hidden_dim)

        # MatMul: Q @ K^T / √d
        matmul_output = (Q_calib @ K_calib.transpose(-2, -1)) / scale_factor

        # QAct calibration
        qact_matmul.calibration(matmul_output)

        if (i + 1) % 10 == 0:
            print(f"  Calibration batch {i+1}/{num_batches} completed")

    print("=== Calibration Completed ===")

    # ========== 4. Compute Quantization Params ==========
    print("\n=== Computing Quantization Parameters ===")

    # QAct params
    act_scale, act_zero = qact_matmul.compute_quant_params()
    print(f"[QAct] Scale: {act_scale.item():.8f}, Zero: {act_zero.item():.2f}")

    # QuantIntSoft params (I-BERT mode: scale propagation)
    matmul_output_scale = act_scale.item()
    quant_softmax.input_scale = matmul_output_scale
    soft_scale, _ = quant_softmax.compute_quant_params()
    print(f"[IntSoft] Input scale: {matmul_output_scale:.8f}")
    print(f"[IntSoft] Output scale: {soft_scale:.6e}")

    # ========== 5. Inference Test ==========
    print("\n" + "="*70)
    print("Inference Phase")
    print("="*70)

    # Test data
    Q_test = torch.randn(batch_size, seq_len, hidden_dim)
    K_test = torch.randn(batch_size, seq_len, hidden_dim)
    V_test = torch.randn(batch_size, seq_len, hidden_dim)

    # --- Step 1: MatMul (Q @ K^T) ---
    print("\nStep 1: Q @ K^T (Attention Logits)")
    matmul_fp32 = (Q_test @ K_test.transpose(-2, -1)) / scale_factor
    print(f"  [FP32] Shape: {matmul_fp32.shape}")
    print(f"  [FP32] Range: [{matmul_fp32.min():.4f}, {matmul_fp32.max():.4f}]")

    # --- Step 2: QAct (Quantize MatMul output) ---
    print("\nStep 2: QAct (Activation Quantization)")
    qact_matmul.mode = 'quantized'
    matmul_quant = qact_matmul(matmul_fp32)
    print(f"  [Quant] Range: [{matmul_quant.min():.4f}, {matmul_quant.max():.4f}]")
    print(f"  [Quant] Unique values: {len(torch.unique(matmul_quant))}")

    # --- Step 3: QuantIntSoft (Integer Softmax) ---
    print("\nStep 3: QuantIntSoft (I-BERT Integer Softmax)")

    # FP32 baseline
    softmax_fp32 = torch.softmax(matmul_fp32, dim=-1)
    print(f"  [FP32] Sum (should be 1.0): {softmax_fp32.sum(dim=-1)[0, 0]:.6f}")

    # I-BERT quantized
    quant_softmax.mode = 'quantized'
    softmax_quant = quant_softmax(matmul_quant, scale=matmul_output_scale)
    print(f"  [Quant] Sum (should be ~1.0): {softmax_quant.sum(dim=-1)[0, 0]:.6f}")
    print(f"  [Quant] Range: [{softmax_quant.min():.6f}, {softmax_quant.max():.6f}]")

    # --- Step 4: Attention @ Value ---
    print("\nStep 4: Attention @ Value (Final Output)")

    # FP32 baseline
    output_fp32 = softmax_fp32 @ V_test
    print(f"  [FP32] Shape: {output_fp32.shape}")
    print(f"  [FP32] Range: [{output_fp32.min():.4f}, {output_fp32.max():.4f}]")

    # Quantized
    output_quant = softmax_quant @ V_test
    print(f"  [Quant] Shape: {output_quant.shape}")
    print(f"  [Quant] Range: [{output_quant.min():.4f}, {output_quant.max():.4f}]")

    # ========== 6. Error Analysis ==========
    print("\n" + "="*70)
    print("Error Analysis")
    print("="*70)

    # Softmax error
    softmax_mse = F.mse_loss(softmax_fp32, softmax_quant)
    softmax_cos = F.cosine_similarity(
        softmax_fp32.flatten(),
        softmax_quant.flatten(),
        dim=0
    )
    print(f"\n[Softmax Error]")
    print(f"  MSE: {softmax_mse.item():.10f}")
    print(f"  Cosine Similarity: {softmax_cos.item():.10f}")
    print(f"  Max Abs Diff: {(softmax_fp32 - softmax_quant).abs().max().item():.6f}")

    # Final output error
    output_mse = F.mse_loss(output_fp32, output_quant)
    output_cos = F.cosine_similarity(
        output_fp32.flatten(),
        output_quant.flatten(),
        dim=0
    )
    print(f"\n[Final Output Error]")
    print(f"  MSE: {output_mse.item():.10f}")
    print(f"  Cosine Similarity: {output_cos.item():.10f}")
    print(f"  Relative Error: {((output_fp32 - output_quant).abs() / (output_fp32.abs() + 1e-8)).mean():.6f}")

    # ========== 7. BitShift Analysis ==========
    print("\n" + "="*70)
    print("BitShift Analysis (Log2 Quantization)")
    print("="*70)

    unique_softmax = torch.unique(softmax_quant)
    unique_nonzero = unique_softmax[unique_softmax > 0]

    if len(unique_nonzero) > 0:
        log2_vals = torch.log2(unique_nonzero)
        is_power_of_2 = torch.allclose(log2_vals, log2_vals.round(), atol=1e-6)

        print(f"\n[Softmax Values Analysis]")
        print(f"  Total unique values: {len(unique_softmax)}")
        print(f"  Non-zero unique values: {len(unique_nonzero)}")
        print(f"  Sample values (first 10): {unique_nonzero[:10].tolist()}")
        print(f"  Log2 of samples: {log2_vals[:10].tolist()}")
        print(f"  All power-of-2? {is_power_of_2}")

        if is_power_of_2:
            print("\n  ✅ Softmax outputs are power-of-2 (BitShift-friendly!)")
        else:
            print("\n  ⚠️ Some values are not exact power-of-2")

    # Success check
    print(f"\n{'='*70}")
    if softmax_cos > 0.99 and output_cos > 0.99:
        print(f"✅ {observer_type} Pipeline Test PASSED!")
    else:
        print(f"⚠️ {observer_type} Pipeline Test: Lower similarity than expected")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    observer_types = ['MinmaxObserver', 'PercentileObserver']

    print("="*70)
    print("Testing I-BERT Attention Pipeline with Multiple Observers")
    print("="*70)

    for obs_type in observer_types:
        try:
            test_full_attention_pipeline(observer_type=obs_type)
        except Exception as e:
            print(f"\n❌ {obs_type} Test Failed")
            print(f"Error: {e}\n")

    print("="*70)
    print("All Pipeline Tests Completed!")
    print("="*70)
