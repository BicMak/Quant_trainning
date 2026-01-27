# QKV 양자화 디버그 요약

**날짜**: 2026-01-26
**상태**: 근본 원인 파악 완료, 수정은 다음 세션에서 진행 예정

---

## 핵심 발견사항

### 진짜 문제: QKV Linear 가중치 양자화

광범위한 디버깅을 통해 어텐션 양자화 실패의 근본 원인을 파악했습니다:

**QKV Linear 레이어의 가중치 양자화가 치명적으로 잘못된 출력을 생성함 (-10.55 dB QSNR)**

#### `debug_qkv_fixed_input.py`의 증거:

```
동일한 고정 입력으로 테스트 (seed=123):

QKV Linear 레이어:
  FP32 출력 범위:   [-5.5534, 6.2902]
  양자화 출력 범위:  [-36.8068, 38.8413]  ← 6배 더 큼!
  QSNR: -10.55 dB  ← 치명적!

Q/K/V 활성화 양자화 (FP32 QKV 출력 사용):
  Q 활성화 QSNR:  39.48 dB  ← 우수
  K 활성화 QSNR:  34.61 dB  ← 우수
  V 활성화 QSNR:  29.15 dB  ← 양호
```

#### 주요 관찰 사항:

1. **가중치 프로파일링은 좋은 QSNR을 보임** (37.71 dB) 하지만 **순전파는 실패** (-10.55 dB)
2. **Q/K/V 활성화 양자화는 완벽하게 작동** (29-39 dB QSNR)
3. **출력 범위 폭발**: 양자화된 QKV 출력이 FP32보다 6배 큼
4. 이것은 Q/K/V 활성화 문제에 대한 모든 이전 가설을 무효화함

---

## 현재 설정 (attn_config.yaml)

### QKV Linear 레이어 (문제의 원인):
```yaml
attn_qkv:
  weight:
    calibration_mode: channel_wise
    observer_type: MinmaxObserver      # ← 이게 문제로 의심됨
    signed: True
    bits: 8
  output:
    output_quant_enable: False
```

### Q/K/V 활성화 레이어 (정상 작동):
```yaml
attn_q_out, attn_k_out, attn_v_input:
  calibration_mode: layer_wise
  observer_type: MinmaxObserver
  signed: False
  bits: 8
```

---

## 조사 타임라인

### 1단계: 프로파일러 시스템 업데이트 ✅
- 모든 양자화 레이어를 배치 단위 프로파일러를 사용하도록 업데이트
- StatProfiler의 중복 `outlier_ratio` 키 버그 수정
- 수정된 파일:
  - `models/ptq/quant_linear.py`
  - `models/ptq/quant_act.py`
  - `models/ptq/quant_intSoft.py`
  - `models/ptq/quant_layernorm.py`
  - `models/ptq/layer_profiler/StatProfiler.py`

### 2단계: Q/K/V 활성화 조사 ❌
- **초기 가설**: Q/K/V 활성화 양자화가 낮은 QSNR 유발 (11-15 dB)
- **시도한 수정**: PercentileObserver + layer_wise로 변경
- **결과**: 더 나빠짐 (9-10 dB QSNR)
- **이유**: 과도하게 공격적인 클리핑 (90% 이상의 데이터 잘림)
- **결론**: 잘못된 방향 - 활성화 양자화가 문제가 아니었음

### 3단계: 근본 원인 격리 ✅
- FP32/양자화 모드에서 동일한 입력을 사용하는 `debug_qkv_fixed_input.py` 생성
- **발견**: QKV Linear 출력이 범위에서 6배 차이남
- **결론**: QKV Linear의 가중치 양자화가 근본 원인

---

## 기술 분석

### 왜 가중치 프로파일링은 좋은 QSNR을 보이는데 순전파는 실패할까?

**가설**: 행렬 곱셈에서 누적된 양자화 오차

```
가중치 양자화 QSNR: 37.71 dB    ← 개별 가중치 값은 FP32에 가까움
                    ↓
행렬 곱셈: W_quant @ X
                    ↓
출력 QSNR: -10.55 dB               ← 누적된 오차가 폭발
```

가능한 원인:
1. **가중치의 MinmaxObserver가 이상치를 포함**하여 스케일을 왜곡
2. **channel_wise 모드**가 QKV 가중치에 적절하지 않을 수 있음 (3개 행렬이 연결됨)
3. **스케일 불일치**: 가중치 스케일이 활성화 범위와 제대로 정렬되지 않을 수 있음

### 입력/출력 데이터 흐름:

```
입력: [1, 197, 768]
  ↓
QKV Linear (양자화된 가중치)
  ↓
FP32 출력:   [-5.55, 6.29]   ← 예상됨
양자화 출력:  [-36.81, 38.84] ← 6배 더 큼! ← 문제
  ↓
Q, K, V 분리
  ↓
Q/K/V 활성화 양자화
  ↓
Q, K, V 출력 (QSNR 29-39 dB) ← 이건 괜찮음
  ↓
Q/K/V 값이 이미 왜곡되어 있어서 어텐션 계산 실패
```

---

## 오늘 수정한 버그

### 1. StatProfiler 중복 키 버그
**파일**: `models/ptq/layer_profiler/StatProfiler.py:24`
```python
# 제거됨 (올바른 outlier_ratio를 덮어쓰고 있었음):
'outlier_ratio': StatProfiler._outlier_ratio(original),
```

### 2. 프로파일러 체크 업데이트
**파일**: 모든 양자화 레이어 클래스
```python
# 이전:
if self.profiler.weight is not None:

# 이후:
if len(self.profiler.weight_batch_list) > 0:
```

### 3. QAct 하드코딩된 양자화 방법 (파악됨, 아직 수정 안됨)
**파일**: `models/ptq/quant_act.py:42-46`
```python
# 현재 (잘못됨):
self.quantizer = build_quantizer(
    quantizer_str='uniform',  # ← 하드코딩됨, 설정 무시
    bit_type=self.bit_type,
    module_type='activation'
)

# 이렇게 되어야 함 (QuantLinear처럼):
self.quantizer = build_quantizer(
    quantizer_str=quant_config.quantization_method.lower(),
    bit_type=self.bit_type,
    module_type='activation'
)
```

---

## 다음 단계 (다음 세션)

### 우선순위 1: QKV Linear 가중치 양자화 수정

**옵션 A: Observer 타입 변경 (권장)**
```yaml
attn_qkv:
  weight:
    calibration_mode: channel_wise
    observer_type: PercentileObserver  # MinmaxObserver에서 변경
    percentile_alpha: 0.9995           # 이상치 제거
    signed: True
    bits: 8
```

**옵션 B: Calibration 모드 변경**
```yaml
attn_qkv:
  weight:
    calibration_mode: layer_wise       # channel_wise에서 변경
    observer_type: MinmaxObserver
    signed: True
    bits: 8
```

**옵션 C: 하이브리드 - 가중치 행렬별 스케일**
- QKV 가중치는 3개 행렬의 연결: [W_q | W_k | W_v]
- 현재 channel_wise: 출력 채널별 스케일
- 고려사항: Q, K, V 가중치 행렬에 대해 별도의 스케일 사용

### 우선순위 2: QAct 하드코딩된 양자화 방법 수정

QuantLinear처럼 QAct가 설정에서 `quantization_method`를 읽도록 수정

### 우선순위 3: 실제 데이터로 검증

현재 테스트는 `torch.randn()` 사용. 실제 ImageNet 이미지로 테스트 필요

---

## 오늘 생성한 파일

1. **debug_qkv_fixed_input.py** ✅
   - 고정된 입력으로 QKV Linear와 Q/K/V 활성화를 별도로 테스트
   - 근본 원인 파악: QKV Linear 가중치 양자화

2. **debug_qkv/test_qkv_only.py** (생성했지만 실행 안함)
   - 4개 레이어만 테스트하는 간소화된 테스트
   - 빠른 검증에 사용 가능

3. **QKV_QUANTIZATION_DEBUG_SUMMARY.md** (원본 파일)
   - 오늘 작업의 종합 요약

---

## 테스트 명령어

### 메인 디버깅 스크립트:
```bash
python debug_qkv_fixed_input.py
```

### 전체 어텐션 파이프라인 테스트:
```bash
python test_embed_to_attn.py
```

### QKV 전용 테스트:
```bash
python debug_qkv/test_qkv_only.py
```

---

## 주요 지표

### 수정 전 (현재 상태):
```
QKV Linear 출력:  -10.55 dB  ← 반드시 수정 필요
Q 활성화:       39.48 dB   ← 이미 양호
K 활성화:       34.61 dB   ← 이미 양호
V 활성화:       29.15 dB   ← 이미 양호
attn_v_out:      4.81 dB   ← QKV로부터 전파된 오차
```

### 수정 후 목표:
```
QKV Linear 출력:  > 30 dB    ← 목표
Q 활성화:       > 30 dB    ← 유지
K 활성화:       > 30 dB    ← 유지
V 활성화:       > 25 dB    ← 유지
attn_v_out:     > 20 dB    ← 개선될 것
```

---

## 결론

조사를 통해 어텐션 양자화 실패의 근본 원인을 성공적으로 파악했습니다:

**QKV Linear 레이어의 가중치 양자화가 -10.55 dB QSNR의 출력을 생성하여, FP32 대비 6배의 범위 폭발을 일으킴.**

Q/K/V 활성화 양자화가 문제라는 이전 가정은 잘못되었습니다. 활성화 양자화 레이어는 우수하게 작동합니다 (29-39 dB QSNR). 문제는 전적으로 QKV Linear 레이어의 가중치 양자화에 있습니다.

다음 세션에서 집중할 사항:
1. QKV 가중치에 PercentileObserver 테스트
2. 연결된 Q/K/V 가중치 행렬에 channel_wise가 적절한지 조사
3. 수정이 하위 어텐션 계산 오류를 해결하는지 검증

---

**세션 종료**: 2026-01-26
