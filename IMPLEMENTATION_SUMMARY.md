# QuantLinear 구현 요약

## 변경 사항

### 1. Config 구조 변경

#### `quant_config.py`
- `QuantConfig`에 필드 추가:
  - `quantization_method: Literal['Uniform', 'Affine']`
  - `enable_profiler: bool`
  - `kl_bins: int` (kv_bins → kl_bins로 이름 변경)

#### `utils/config_loader.py`
- `load_config_from_yaml()` 반환 타입 변경:
  - 기존: `LayerQuantConfig` 객체
  - 변경: `dict` with keys: `'output'`, `'weight'`, `'activation'`

- YAML 구조 변경:
  ```yaml
  outputs:      # Output quantization config
    calibration_mode: channel_wise
    bit_type:
      bits: 8
      signed: false
      name: int8
    observer_type: PercentileObserver
    quantization_method: Uniform
    percentile_alpha: 0.9995
    enable_profiler: true

  weight:       # Weight quantization config
    calibration_mode: channel_wise
    bit_type:
      bits: 8
      signed: true
      name: int8
    observer_type: MinmaxObserver
    quantization_method: Uniform
    enable_profiler: true

  activation:   # (Optional) Activation quantization config
    ...
  ```

### 2. QuantLinear 클래스 변경

#### 초기화 (`__init__`)
```python
def __init__(self,
             input_module: nn.Module,
             out_config: QuantConfig,      # Output config
             weight_config: QuantConfig,    # Weight config (별도)
             layer_name: str = 'qlinear'):
```

**주요 변경점:**
1. `out_config`와 `weight_config`를 분리해서 받음
2. Weight observer/quantizer와 Output observer/quantizer 각각 초기화
3. `output_profiler`와 `weight_profiler` 분리
4. **초기화 시 `quantize_weight()` 자동 호출** (weight는 고정값)

#### 새로운 메서드: `quantize_weight()`
```python
def quantize_weight(self):
    """
    Weight quantization 전용 메서드.
    Weight는 고정값이므로 초기화 시 한 번만 실행.
    """
    with torch.no_grad():
        # 1. Weight observer 업데이트
        self.weight_observer.update(self.weight)

        # 2. Quantization params 계산
        weight_scaler, weight_zero = self.weight_observer.get_quantization_params()

        # 3. Weight quantization 수행
        self.quant_weight = torch.clamp(...)

        # 4. Weight profiler 업데이트
        if self.enable_profiling and self.weight_profiler is not None:
            dequant_weight = (self.quant_weight - zero_reshaped) * scaler_reshaped
            self.weight_profiler.update_weight(self.weight, dequant_weight.detach())

        # 5. Params 저장
        self.weight_scaler = weight_scaler
        self.weight_zero = weight_zero
```

#### 수정된 메서드: `calibration()`
```python
def calibration(self, x):
    """
    Calibration 전용 - output activation만 처리.
    Weight는 이미 quantize_weight()에서 처리됨.
    """
    with torch.no_grad():
        # FP32 forward로 output 계산
        output = self.fwd_func(x, self.weight, self.bias, **self.fwd_kwargs)

        # Output observer만 업데이트 (activation calibration)
        self.output_observer.update(output)

    return output
```

#### 수정된 메서드: `compute_output_quant_params()` (기존 `compute_quant_params()`)
```python
def compute_output_quant_params(self):
    """
    Calibration 끝난 후 output quantization params 계산.
    Weight는 이미 quantize_weight()에서 처리됨.

    Returns:
        tuple: (weight_params, output_params)
            - weight_params: (scaler, zero) - 이미 계산된 값
            - output_params: (scaler, zero) - 새로 계산한 값
    """
    # Output quantization params 계산
    output_scaler, output_zero = self.output_observer.get_quantization_params()

    # Output quantizer에 params 설정
    self.output_quantizer.update_quantization_params(output_scaler, output_zero)

    # Weight params는 이미 quantize_weight()에서 계산됨
    return (self.weight_scaler, self.weight_zero), (self.output_scaler, self.output_zero)
```

#### Profiler 관련 변경
- `self.profiler` → `self.output_profiler`로 변경
- `get_profiling_results()`: output과 weight 각각 분리해서 반환
  ```python
  {
      'output': {
          'statistics': {...},
          'histogram': {...},
          'time': {...},
          'memory': {...}
      },
      'weight': {
          'statistics': {...},
          'histogram': {...},
          'time': {...},
          'memory': {...}
      }
  }
  ```

#### Forward 메서드
```python
def forward(self, x):
    if self.mode == 'quantized':
        # 1. Dequantize weights (int8 -> fp32)
        dequant_weight = self.weight_quantizer.forward(self.quant_weight)

        # 2. Linear operation in fp32
        out_fp32 = self.fwd_func(x, dequant_weight, self.bias, **self.fwd_kwargs)

        # 3. Output fake quantization
        output = self.output_quantizer.forward(out_fp32)

        # 4. Output profiler 업데이트
        if self.enable_profiling and self.output_profiler is not None:
            self.output_profiler.update_weight(fp32_output, output.detach())

        return output
    else:  # fp32
        return self.fwd_func(x, self.weight, self.bias, **self.fwd_kwargs)
```

## 사용 예시

```python
# 1. Config 로드
from utils.config_loader import load_config_from_yaml

configs = load_config_from_yaml('configs/attn_config.yaml')
output_config = configs['output']    # Output quantization config
weight_config = configs['weight']    # Weight quantization config

# 2. QuantLinear 생성 (weight 자동 quantize됨)
from models.ptq.quant_linear import QuantLinear

layer = QuantLinear(
    input_module=linear_layer,
    out_config=output_config,
    weight_config=weight_config,
    layer_name='test_layer'
)
# → 초기화 시 quantize_weight() 자동 호출됨

# 3. Calibration (activation만 처리)
for x in calib_data:
    layer.calibration(x)

# 4. Output quantization params 계산
(weight_params, output_params) = layer.compute_output_quant_params()

# 5. Inference
layer.mode = 'quantized'
output = layer(input_data)

# 6. Profiling 결과 조회 (분리된 결과)
results = layer.get_profiling_results()
print(f"Output QSNR: {results['output']['statistics']['qsnr']:.2f} dB")
print(f"Weight QSNR: {results['weight']['statistics']['qsnr']:.2f} dB")
```

## 핵심 설계 원칙

1. **Weight는 고정값** → 초기화 시 즉시 quantize
2. **Calibration은 activation만** → output observer만 업데이트
3. **Config 분리** → weight와 output을 독립적으로 설정
4. **Profiler 분리** → weight와 output의 profiling 결과를 각각 추적

## 테스트

테스트 함수 `test_quantlinear_profiling()` 포함:
- Weight quantization이 초기화 시 자동으로 수행되는지 확인
- Calibration이 output만 처리하는지 확인
- Profiling 결과가 weight와 output으로 분리되는지 확인
- QSNR, MSE 등 통계 정보 검증

```bash
python models/ptq/quant_linear.py
```
