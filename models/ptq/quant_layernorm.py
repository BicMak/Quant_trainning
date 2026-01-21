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

class QLayerNorm(nn.Module):
    def __init__(self,
                 input_module: nn.Module,
                 quant_config: QuantConfig):
        super().__init__()

        # quant_config에서 설정 추출
        self.input_module = input_module
        self.quant_config = quant_config
        self.observer_type = quant_config.observer_type
        self.bit_type = BitType(
            bits=quant_config.bit_type.bits,
            signed=quant_config.bit_type.signed,
            name=quant_config.bit_type.name
        )
        self.calibration_mode = quant_config.calibration_mode
        self.quant_weight = None
        self.mode = 'fp32'

        #
        self.normalized_shape = self.input_module.normalized_shape
        self.eps = self.input_module.eps
        self.num_channels = self.input_module.weight.size(0)

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

        #2. layer initialization  
        self.fwd_kwargs = dict()
        self.fwd_func = F.layer_norm

        self.weight = self.input_module.weight.clone().detach()

        self.bias = self.input_module.bias.clone().detach() if self.input_module.bias is not None else None

    
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

    def compute_quant_params(self,calib_loader):
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


    def forward(self, x):
        # LayerNorm은 FP32로 계산, 출력만 양자화
        out = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)

        if self.mode == 'quantized':
            out = self.output_quantizer.forward(out)

        return out  
            


def test_quant_layernorm(observer_type='PercentileObserver', use_ptf=True):
    # ========== 1. Config 설정 ==========
    bit_config = BitTypeConfig(bits=8, signed=True, name='int8')
    quant_config = QuantConfig(
        calibration_mode='channel_wise',
        bit_type=bit_config,
        observer_type=observer_type
    )

    # ========== 2. LayerNorm 포함 모델 생성 ==========
    class SimpleTransformerBlock(nn.Module):
        def __init__(self, hidden_dim=768):
            super().__init__()
            self.ln1 = nn.LayerNorm(hidden_dim)
            self.ln2 = nn.LayerNorm(hidden_dim)
            self.fc1 = nn.Linear(hidden_dim, hidden_dim * 4)
            self.fc2 = nn.Linear(hidden_dim * 4, hidden_dim)

        def forward(self, x):
            x = self.ln1(x)
            x = F.gelu(self.fc1(x))
            x = self.fc2(x)
            x = self.ln2(x)
            return x

    hidden_dim = 768
    model = SimpleTransformerBlock(hidden_dim)

    print(f"\n{'='*60}")
    print(f"Testing QLayerNorm with {observer_type} (PTF: {use_ptf})")
    print(f"{'='*60}")
    print("Original LayerNorm layers:")
    for name, module in model.named_modules():
        if isinstance(module, nn.LayerNorm):
            print(f"  {name}: {module}")

    # ========== 3. QLayerNorm으로 변환 ==========
    quant_ln_layers = {}
    for name, layer in model.named_modules():
        if isinstance(layer, nn.LayerNorm):
            quant_layer = QLayerNorm(
                input_module=layer,
                quant_config=quant_config
            )
            quant_ln_layers[name] = quant_layer
            print(f"Converted: {name}")

    print(f"\nQLayerNorm 개수: {len(quant_ln_layers)}")

    # ========== 4. 더미 Calibration 데이터 생성 ==========
    num_batches = 32
    batch_size = 16
    seq_len = 197  # ViT: 196 patches + 1 cls token

    print(f"\n더미 데이터 생성: {num_batches} batches x {batch_size} x {seq_len} x {hidden_dim}")
    calib_data = [
        torch.randn(batch_size, seq_len, hidden_dim)
        for _ in range(num_batches)
    ]

    # ========== 5. Calibration 수행 (1차: observer update) ==========
    print("\n=== Calibration 시작 (1차: Observer Update) ===")
    for batch_idx, x in enumerate(calib_data):
        # ln1 calibration
        out = quant_ln_layers['ln1'].calibration(x)
        out = F.gelu(model.fc1(out))
        out = model.fc2(out)
        # ln2 calibration
        out = quant_ln_layers['ln2'].calibration(out)

        if (batch_idx + 1) % 8 == 0:
            print(f"Batch {batch_idx + 1}/{num_batches} 완료")

    print("=== Calibration 1차 완료 ===")

    # ========== 6. Observer 통계 확인 ==========
    print("\n=== Observer 통계 ===")
    for name, layer in quant_ln_layers.items():
        print(f"\n{name}:")
        print(f"  Observer type: {type(layer.output_observer).__name__}")
        if layer.output_observer.min_val.shape == torch.Size([]):
            print(f"  Output min_val: {layer.output_observer.min_val.item()}")
            print(f"  Output max_val: {layer.output_observer.max_val.item()}")
        else:     
            print(f"  Output min_val: {layer.output_observer.min_val}")
            print(f"  Output max_val: {layer.output_observer.max_val}")

    # ========== 7. Quantization Params 계산 (2차: PTF alpha 탐색) ==========
    print("\n=== Quantization Parameters 계산 ===")
    for name, layer in quant_ln_layers.items():
        if use_ptf:
            scale, zero = layer.compute_quant_params(calib_loader=calib_data)
            print(f"\n{name} (PTF 적용):")
            print(f"  Alpha (first 10): {layer.alpha[:10].tolist()}")
        else:
            scale, zero = layer.compute_quant_params(calib_loader=None)
            print(f"\n{name} (PTF 미적용):")

        print(f"  Scale shape: {scale.shape}")
        print(f"  Scale (first 5): {scale[:5].tolist()}")
        print(f"  s_base: {layer.s_base:.8f}")

    # ========== 8. FP32 vs Fake Quantization 비교 ==========
    print("\n=== FP32 vs Fake Quantization 출력 비교 ===")
    test_input = torch.randn(1, seq_len, hidden_dim)

    # FP32 forward
    for layer in quant_ln_layers.values():
        layer.mode = 'fp32'

    out_fp32 = quant_ln_layers['ln1'](test_input)
    out_fp32 = F.gelu(model.fc1(out_fp32))
    out_fp32 = model.fc2(out_fp32)
    out_fp32 = quant_ln_layers['ln2'](out_fp32)

    # Fake Quantization forward
    for layer in quant_ln_layers.values():
        layer.mode = 'quantized'

    out_quant = quant_ln_layers['ln1'](test_input)
    out_quant = F.gelu(model.fc1(out_quant))
    out_quant = model.fc2(out_quant)
    out_quant = quant_ln_layers['ln2'](out_quant)

    mse = F.mse_loss(out_fp32, out_quant).item()
    max_diff = (out_fp32 - out_quant).abs().max().item()
    cosine_sim = F.cosine_similarity(out_fp32.flatten().unsqueeze(0),
                                      out_quant.flatten().unsqueeze(0)).item()

    print(f"Output shape: {out_fp32.shape}")
    print(f"MSE: {mse:.6f}")
    print(f"Max diff: {max_diff:.6f}")
    print(f"Cosine Similarity: {cosine_sim:.4f} ({cosine_sim*100:.2f}%)")

    print(f"\n=== {observer_type} (PTF: {use_ptf}) 테스트 완료 ===\n")

    return mse, cosine_sim


if __name__ == "__main__":
    print("="*60)
    print("Testing QLayerNorm with PercentileObserver (channel_wise)")
    print("="*60)

    results = []

    # PercentileObserver - PTF 미적용
    try:
        mse, cos = test_quant_layernorm(observer_type='PercentileObserver', use_ptf=False)
        results.append(('PercentileObserver (no PTF)', mse, cos))
    except Exception as e:
        print(f"PercentileObserver (no PTF) 실패: {e}")

    # PercentileObserver - PTF 적용
    try:
        mse, cos = test_quant_layernorm(observer_type='PercentileObserver', use_ptf=True)
        results.append(('PercentileObserver (PTF)', mse, cos))
    except Exception as e:
        print(f"PercentileObserver (PTF) 실패: {e}")

    # 결과 요약
    print("="*60)
    print("결과 요약")
    print("="*60)
    print(f"{'Method':<30} {'MSE':<15} {'Cosine Sim':<15}")
    print("-"*60)
    for name, mse, cos in results:
        print(f"{name:<30} {mse:<15.6f} {cos*100:<14.2f}%")
    print("="*60)