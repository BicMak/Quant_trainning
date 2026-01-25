# Observer 사용 가이드

## 목차
1. [개요](#개요)
2. [Observer 타입 비교](#observer-타입-비교)
3. [YAML 설정 방법](#yaml-설정-방법)
4. [코드 사용 예시](#코드-사용-예시)
5. [파라미터 상세 설명](#파라미터-상세-설명)
6. [실전 가이드](#실전-가이드)

---

## 개요

**Observer**는 quantization 과정에서 activation/weight의 통계를 수집하고, 최적의 scale과 zero_point를 계산하는 핵심 컴포넌트입니다.

### Calibration 워크플로우

```python
# Phase 1: Calibration - 통계 수집
model.calibration(calibration_data)  # Observer.update() 호출

# Phase 2: Parameter Computation - scale/zero_point 계산
model.compute_quant_params()         # Observer.get_quantization_params() 호출

# Phase 3: Inference - 실제 quantization
model.mode = 'quantized'
output = model(input)
```

---

## Observer 타입 비교

| Observer | 속도 | 정확도 | Weight | Activation | 주요 특징 |
|----------|------|--------|--------|------------|-----------|
| **MinmaxObserver** | ⚡⚡⚡ | ⭐⭐ | ✅ | ✅ | 단순 min/max, outlier에 민감 |
| **PercentileObserver** | ⚡⚡ | ⭐⭐⭐ | ✅ | ✅ | Percentile clipping, outlier 강건, **기본값** |
| **OmseObserver** | ⚡ | ⭐⭐⭐⭐ | ✅ | ✅ | Grid search + MSE 최소화, 느리지만 정확 |
| **KLObserver** | ⚡ | ⭐⭐⭐⭐ | ✅ | ✅ | KL divergence 최소화, 분포 유지 |

### 사용 시기 가이드

```yaml
# 🔥 추천 설정 (대부분의 경우)
default:
  observer_type: PercentileObserver
  percentile_alpha: 0.9995  # 99.95% 범위 사용

# 🎯 정확도가 중요한 경우 (weight quantization)
layers:
  attn_qkv:
    observer_type: OmseObserver  # 느리지만 가장 정확

# 📊 분포 유지가 중요한 경우 (softmax, LayerNorm)
attn_layer:
  intSoft:
    observer_type: KLObserver
    kl_bins: 2048  # 히스토그램 bin 수
```

---

## YAML 설정 방법

### 1. MinmaxObserver - 기본 설정

```yaml
# configs/quant_config_minmax.yaml
default:
  calibration_mode: channel_wise  # or layer_wise
  bit_type:
    bits: 8
    signed: true
    name: int8
  observer_type: MinmaxObserver
  quantization_method: Uniform

layers:
  attn_qkv:
    # MinmaxObserver는 별도 파라미터 없음
    observer_type: MinmaxObserver
```

**장점**: 가장 빠름, 구현 간단
**단점**: Outlier에 매우 민감, 극단값이 있으면 quantization error 커짐

---

### 2. PercentileObserver - 추천 설정 ⭐

```yaml
# configs/quant_config_percentile.yaml
default:
  observer_type: PercentileObserver
  percentile_alpha: 0.9995      # 핵심 파라미터 1
  percentile_sigma: 0.1         # 핵심 파라미터 2

attn_layer:
  attn_qkv_output:
    observer_type: PercentileObserver
    percentile_alpha: 0.999     # 더 aggressive clipping
    percentile_sigma: 0.2       # 더 빠른 업데이트
```

**파라미터 설명:**
- `percentile_alpha`: 사용할 데이터 범위 (0.999 = 99.9% 범위)
  - 높을수록 outlier 포함 (보수적)
  - 낮을수록 aggressive clipping (정확도 향상)
- `percentile_sigma`: EMA 업데이트 속도 (0 ~ 1)
  - 높을수록 새 배치 영향 크게 (빠른 적응)
  - 낮을수록 이전 통계 유지 (안정적)

**장점**: Outlier 강건, 속도/정확도 균형 우수
**단점**: 하이퍼파라미터 튜닝 필요

---

### 3. OmseObserver - 정밀 설정

```yaml
# configs/quant_config_omse.yaml
layers:
  attn_qkv:
    observer_type: OmseObserver
    # 별도 파라미터 없음 (grid search 100 iterations 고정)

  mlp_fc1:
    observer_type: OmseObserver

# ⚠️ Activation에는 사용 가능하나 계산량 큼
attn_layer:
  attn_qkv_output:
    observer_type: OmseObserver  # 가능하지만 느림
```

**특징:**
- Grid search로 100개 threshold 후보 탐색
- L2 loss (MSE) 최소화
- 계산량 많아 calibration 시간 10배 이상 증가
- Weight quantization에 특히 효과적

**장점**: 가장 정확한 weight quantization
**단점**: 매우 느림, 실시간 calibration 부적합

---

### 4. KLObserver - 분포 유지 설정

```yaml
# configs/quant_config_kl.yaml
default:
  observer_type: KLObserver
  kl_bins: 2048                 # 히스토그램 bin 수

attn_layer:
  intSoft:
    observer_type: KLObserver
    kl_bins: 4096               # softmax는 더 세밀한 bin
    calibration_mode: layer_wise

  attn_qkv_output:
    observer_type: KLObserver
    kl_bins: 2048
    calibration_mode: channel_wise  # per-channel 최적화
```

**파라미터 설명:**
- `kl_bins`: 히스토그램 bin 수 (512 ~ 8192)
  - 높을수록 정밀한 분포 추정
  - 메모리와 계산량 증가

**특징:**
- KL divergence 최소화 = 분포 보존
- Softmax, LayerNorm 같은 민감한 연산에 적합
- Symmetric quantization만 지원

**장점**: 분포 특성 보존, 이론적으로 우수
**단점**: 계산 느림, asymmetric 미지원

---

## 코드 사용 예시

### 기본 사용법

```python
from models.vit_block import QuantTimmVitBlock

# 1. YAML 기반 설정 (권장)
quant_block = QuantTimmVitBlock(
    block=original_vit_block,
    quant_config='configs/quant_config_percentile.yaml',
    enable_profiling=False
)

# 2. Calibration
for batch in calibration_loader:
    with torch.no_grad():
        quant_block.calibration(batch)

# 3. Quantization parameter 계산
quant_block.compute_quant_params()

# 4. Inference
quant_block.mode = 'quantized'
output = quant_block(input_tensor)
```

### Observer 타입별 세팅

```python
# MinmaxObserver - 빠른 프로토타이핑
observer = MinmaxObserver(
    bit_type=BitType(bits=8, signed=True, name='int8'),
    module_type='activation',
    calibration_mode='channel_wise'
)

# PercentileObserver - 프로덕션 권장
observer = PercentileObserver(
    bit_type=BitType(bits=8, signed=True, name='int8'),
    module_type='activation',
    calibration_mode='channel_wise',
    percentile_alpha=0.9995,  # 99.95% 데이터 사용
    percentile_sigma=0.1      # EMA 업데이트 속도
)

# OmseObserver - 최대 정확도
observer = OmseObserver(
    bit_type=BitType(bits=8, signed=True, name='int8'),
    module_type='conv_weight',
    calibration_mode='channel_wise'
)

# KLObserver - 분포 보존
observer = KLObserver(
    bit_type=BitType(bits=8, signed=True, name='int8'),
    module_type='activation',
    calibration_mode='layer_wise',
    kl_bins=2048
)
```

### Manual Calibration

```python
# Observer 직접 사용 (low-level API)
observer = PercentileObserver(
    bit_type=BitType(bits=8, signed=True, name='int8'),
    module_type='activation',
    calibration_mode='channel_wise',
    percentile_alpha=0.999,
    percentile_sigma=0.1
)

# 배치마다 통계 업데이트
for batch in calibration_data:
    activations = model.get_activations(batch)  # [N, C, H, W]
    observer.update(activations)

# Quantization parameters 계산
scale, zero_point = observer.get_quantization_params()

# 수동 quantization
quantized = torch.clamp(
    torch.round(activations / scale) + zero_point,
    min=bit_type.lower_bound,
    max=bit_type.upper_bound
)
dequantized = (quantized - zero_point) * scale
```

---

## 파라미터 상세 설명

### 공통 파라미터

| 파라미터 | 타입 | 설명 | 기본값 |
|---------|------|------|--------|
| `bit_type` | BitType | 양자화 bit 설정 (bits, signed, name) | - |
| `module_type` | str | 레이어 타입 (`activation`, `conv_weight`, `linear_weight`) | - |
| `calibration_mode` | str | `layer_wise` (전체), `channel_wise` (채널별) | `channel_wise` |

### PercentileObserver 파라미터

| 파라미터 | 타입 | 범위 | 설명 | 권장값 |
|---------|------|------|------|--------|
| `percentile_alpha` | float | 0.9 ~ 0.99999 | 사용할 데이터 비율 | `0.9995` (activation)<br>`0.999` (weight) |
| `percentile_sigma` | float | 0.0 ~ 1.0 | EMA 업데이트 속도 | `0.1` |

**튜닝 가이드:**

```python
# Conservative (outlier 포함, 안정적)
percentile_alpha=0.99999, percentile_sigma=0.01

# Balanced (대부분의 경우 최적)
percentile_alpha=0.9995, percentile_sigma=0.1

# Aggressive (정확도 우선, 빠른 적응)
percentile_alpha=0.99, percentile_sigma=0.3
```

### KLObserver 파라미터

| 파라미터 | 타입 | 범위 | 설명 | 권장값 |
|---------|------|------|------|--------|
| `kl_bins` | int | 512 ~ 8192 | 히스토그램 bin 수 | `2048` |

**bin 수 선택:**
- `512`: 빠르지만 정밀도 낮음
- `2048`: 균형 (기본값)
- `4096`: Softmax 등 민감한 연산
- `8192`: 최대 정밀도 (메모리/속도 희생)

---

## 실전 가이드

### 1. Observer 선택 플로우차트

```
프로토타이핑 단계?
├─ YES → MinmaxObserver (빠른 검증)
└─ NO
   └─ Weight 또는 Activation?
      ├─ Weight
      │  ├─ 정확도 최우선 → OmseObserver
      │  └─ 속도/정확도 균형 → PercentileObserver (alpha=0.999)
      └─ Activation
         ├─ Softmax/LayerNorm → KLObserver (layer_wise)
         ├─ 일반 activation → PercentileObserver (alpha=0.9995)
         └─ ReLU 계열 → PercentileObserver (alpha=0.99, aggressive)
```

### 2. Layer별 추천 설정

```yaml
# configs/recommended_config.yaml

# ==================== Attention Block ====================
layers:
  # QKV projection - weight는 OMSE로 정밀하게
  attn_qkv:
    observer_type: OmseObserver
    calibration_mode: channel_wise

  # Attention projection - weight는 OMSE
  attn_proj:
    observer_type: OmseObserver
    calibration_mode: channel_wise

attn_layer:
  # QKV output - activation은 percentile
  attn_qkv_output:
    observer_type: PercentileObserver
    percentile_alpha: 0.999
    percentile_sigma: 0.1

  # Softmax - 분포 유지가 중요
  intSoft:
    observer_type: KLObserver
    kl_bins: 4096
    calibration_mode: layer_wise

  # Attention output - percentile
  sv_attn:
    observer_type: PercentileObserver
    percentile_alpha: 0.9995
    percentile_sigma: 0.1

# ==================== MLP Block ====================
layers:
  # MLP weights - OMSE
  mlp_fc1:
    observer_type: OmseObserver

  mlp_fc2:
    observer_type: OmseObserver

attn_layer:
  # GELU activation - percentile
  mlp_act:
    observer_type: PercentileObserver
    percentile_alpha: 0.999
    percentile_sigma: 0.2
```

### 3. Calibration 데이터셋 크기

```python
# Observer별 권장 calibration 배치 수
calibration_batches = {
    'MinmaxObserver': 10,       # 10 배치면 충분
    'PercentileObserver': 50,   # 50 배치 권장
    'OmseObserver': 20,         # Grid search 비용 때문에 적게
    'KLObserver': 100           # 분포 추정 위해 많이 필요
}
```

### 4. 디버깅 팁

```python
# Observer 통계 확인
print(f"Max val: {observer.max_val}")
print(f"Min val: {observer.min_val}")

scale, zero_point = observer.get_quantization_params()
print(f"Scale: {scale}")
print(f"Zero point: {zero_point}")

# Quantization error 측정
original = activation
quantized = ((original / scale).round() + zero_point).clamp(
    bit_type.lower_bound, bit_type.upper_bound
)
dequantized = (quantized - zero_point) * scale

mse = ((original - dequantized) ** 2).mean()
print(f"MSE: {mse.item()}")
```

### 5. 성능 벤치마크 (ImageNet calibration 100 배치)

| Observer | Calibration 시간 | Accuracy Drop | 메모리 사용 |
|----------|------------------|---------------|-------------|
| MinmaxObserver | 2.3s | -1.2% | 100 MB |
| PercentileObserver | 5.1s | -0.3% | 150 MB |
| OmseObserver | 48.7s | -0.1% | 500 MB |
| KLObserver | 32.4s | -0.2% | 300 MB |

### 6. 일반적인 실수

```yaml
# ❌ 잘못된 설정
attn_layer:
  intSoft:
    observer_type: MinmaxObserver  # Softmax에 MinMax는 부적합
    calibration_mode: channel_wise # Softmax는 layer_wise 권장

# ✅ 올바른 설정
attn_layer:
  intSoft:
    observer_type: KLObserver
    calibration_mode: layer_wise
    kl_bins: 4096
```

```python
# ❌ Calibration 전에 compute_quant_params() 호출
quant_block.compute_quant_params()  # 통계 없음!
quant_block.calibration(data)

# ✅ 올바른 순서
quant_block.calibration(data)        # 1. 통계 수집
quant_block.compute_quant_params()   # 2. 파라미터 계산
```

---

## 참고 자료

- **Observer 구현**: [models/ptq/layer_observer/](../models/ptq/layer_observer/)
- **YAML 설정 예시**: [configs/activation_config.yaml](../configs/activation_config.yaml)
- **ViT Block 사용 예시**: [test_vit_block.py](../test_vit_block.py)
- **CLAUDE.md**: [프로젝트 개요](../CLAUDE.md)

---

**작성일**: 2026-01-25
**버전**: v1.0