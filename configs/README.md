# Quantization Configuration Guide

단일 YAML 파일로 모든 레이어의 양자화 설정을 관리합니다.

## 파일 구조

### quant_config_int8.yaml

```yaml
# Global Default Configuration
default:
  calibration_mode: channel_wise
  bit_type:
    bits: 8
    signed: true
    name: int8
  observer_type: PercentileObserver
  quantization_method: Uniform
  percentile_alpha: 0.95
  percentile_sigma: 0.01
  kv_bins: 2048

# Weight Quantization (Linear layers)
layers:
  attn_qkv: { ... }
  attn_proj: { ... }
  mlp_fc1: { ... }
  mlp_fc2: { ... }
  norm1: { ... }
  norm2: { ... }

# Activation Quantization
attn_layer:
  attn_qkv_output: { ... }
  kv_act: { ... }
  sv_attn: { ... }
  mlp_act: { ... }
  mlp_act2: { ... }
  intSoft: { ... }

# Residual Quantization
residual_layer:
  residual1: { ... }
  residual2: { ... }
```

## 사용 방법

### 방법 1: YAML 파일 경로 전달 (권장)

```python
from models.vit_block import QuantTimmVitBlock

quant_block = QuantTimmVitBlock(
    block=original_block,
    quant_config='configs/quant_config_int8.yaml',  # YAML 파일 경로
    enable_profiling=True
)
```

### 방법 2: LayerQuantConfig 객체 사용

```python
from utils.config_loader import load_config_from_yaml
from models.vit_block import QuantTimmVitBlock

layer_config = load_config_from_yaml('configs/quant_config_int8.yaml')
quant_block = QuantTimmVitBlock(
    block=original_block,
    quant_config=layer_config,  # LayerQuantConfig 객체
    enable_profiling=True
)
```

### 방법 3: 기존 방식 (단일 QuantConfig)

```python
from quant_config import QuantConfig, BitTypeConfig
from models.vit_block import QuantTimmVitBlock

bit_config = BitTypeConfig(bits=8, signed=True, name='int8')
quant_config = QuantConfig(
    calibration_mode='channel_wise',
    bit_type=bit_config,
    observer_type='PercentileObserver'
)
quant_block = QuantTimmVitBlock(
    block=original_block,
    quant_config=quant_config,  # QuantConfig 객체 (모든 레이어 동일 설정)
    enable_profiling=True
)
```

## 레이어별 설정 커스터마이징

### 예시: 특정 레이어를 INT4로 변경

```yaml
layers:
  attn_qkv:
    calibration_mode: channel_wise
    bit_type:
      bits: 4  # INT8 → INT4로 변경
      signed: true
      name: int4
    observer_type: OmseObserver  # Observer도 변경 가능
    quantization_method: Uniform
    percentile_alpha: 0.99  # Alpha 값 조정
    percentile_sigma: 0.01
```

### 예시: Mixed Precision 설정

```yaml
# Weights는 INT4, Activations는 INT8
layers:
  attn_qkv:
    bit_type:
      bits: 4
      name: int4
  mlp_fc1:
    bit_type:
      bits: 4
      name: int4

attn_layer:
  attn_qkv_output:
    bit_type:
      bits: 8
      name: int8
  kv_act:
    bit_type:
      bits: 8
      name: int8
```

## Config Loader 설정 우선순위

1. **Layer-specific config** (layers/attn_layer/residual_layer 섹션)
2. **Default config** (default 섹션)

레이어별 설정이 없으면 자동으로 default 설정을 사용합니다.

## 테스트

```bash
# Config loader 테스트
python test_config_loader.py

# ViT Block 전체 테스트
python -m models.vit_block
```

## 주요 레이어

### Weight Layers (layers)
- `attn_qkv`: Attention QKV projection weights
- `attn_proj`: Attention output projection weights
- `mlp_fc1`: MLP first layer weights
- `mlp_fc2`: MLP second layer weights
- `norm1`: LayerNorm 1 (before attention)
- `norm2`: LayerNorm 2 (before MLP)

### Activation Layers (attn_layer)
- `attn_qkv_output`: QKV output activations
- `kv_act`: Key-Value activations (before softmax)
- `sv_attn`: Softmax-Value attention output
- `mlp_act`: MLP activation (after GELU)
- `mlp_act2`: MLP second activation
- `intSoft`: IntSoftmax quantization

### Residual Layers (residual_layer)
- `residual1`: First residual connection (after attention)
- `residual2`: Second residual connection (after MLP)

## Parameters 설명

- `calibration_mode`: `layer_wise` or `channel_wise`
  - `layer_wise`: 레이어 전체에 하나의 scale 사용
  - `channel_wise`: 채널별로 개별 scale 사용 (더 정확)

- `observer_type`: 양자화 범위 결정 방법
  - `MinmaxObserver`: Min/Max 값 사용
  - `PercentileObserver`: Percentile 기반 (outlier 제거)
  - `OmseObserver`: OMSE (Outlier-aware MSE) 기반
  - `KLObserver`: KL Divergence 기반

- `percentile_alpha`: Percentile threshold (0.95 = 상위 5% outlier 제거)
- `percentile_sigma`: Smoothing parameter
