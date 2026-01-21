

# 🚀 FQ-ViT: Vision Transformer Block Quantization Guide

본 문서는 `QuantTimmVitBlock` 클래스를 통해 실현되는 **Fully Quantized ViT**의 구조와 각 컴포넌트별 양자화 전략을 설명합니다. 패치 임베딩 이후의 모든 Transformer Block을 **정수 연산(Integer-only)** 가능 상태로 만드는 것을 목표로 합니다.

## 📌 핵심 설계 요약

* **Target**: `timm` 라이브러리 기반의 ViT Block
* **Strategy**: 
    1. LOG scale 양자화로 
* **Scope**: 임베딩 레이어를 제외한 인코더 블록 전체 (패치 임베딩은 FP32 유지)

---

## 🛠 주요 양자화 컴포넌트

### 1. PTF (Power-of-Two Factor) LayerNorm

ViT는 채널별 데이터 분포의 차이가 큽니다. 이를 해결하기 위해 `QLayerNorm`을 사용합니다.

* **동작**: 채널별 스케일 인자를 **2의 거듭제곱()**으로 제한합니다.
* **효과**: 비싼 부동소수점 나눗셈을 **Bit-shift** 연산으로 대체하여 NPU 효율을 극대화합니다.

### 2. LIS (Log-Int-Softmax)

Attention Score()의 극심한 불균형 분포를 처리합니다.

* **동작**: Softmax의 출력을 일반적인 Uniform INT8이 아닌 **Log-domain(Log-integer)**으로 양자화합니다.
* **효과**:  연산 시 일반적인 곱셈기 대신 **Shift-Add**만으로 연산이 가능해집니다.

### 3. MLP & GELU (LUT 기반)

GELU와 같은 비선형 활성화 함수를 처리합니다.

* **동작**: `QAct` 모듈이 `INT32` 입력을 받아 미리 계산된 **Look-Up Table(LUT)**을 통해 `INT8`로 즉시 사상합니다.
* **효과**: 복잡한 초월 함수() 연산을 메모리 참조 한 번으로 끝냅니다.

---

## 🔄 Block 내부 데이터 흐름 (Data Flow)

| 단계 | 연산 모듈 | 입력 비트 | 출력 비트 | 비고 |
| --- | --- | --- | --- | --- |
| **Norm 1** | `QLayerNorm` | INT8 | **INT8** | PTF 적용 |
| **QKV Gen** | `QuantLinear` | INT8 | **INT32** | 정수 누적(Accumulation) |
| **Re-quant** | `attn_qkv_output` | INT32 | **INT8** | 다음 MatMul 입력용 |
| **Score** | `q @ k.T` | INT8 | **INT32** | Attention Score |
| **Softmax** | `QuantIntSoft` | INT32 | **Log-INT8** | LIS (Log-domain 변환) |
| **Context** | `attn @ v` | Log-INT8 | **INT32** | **Bit-shift 연산 수행** |
| **Residual** | `Add` | INT8 + INT8 | **INT8** | `residual1/2`를 통한 재양자화 |

---

## 📝 구현 특징 및 주의사항

### Calibration (통계 수집)

`calibration()` 메서드는 PTQ(Post-Training Quantization) 과정에서 각 레이어의 최적 `scale`과 `zero_point`를 계산하기 위해 활성화 값의 통계를 수집합니다.

* **PTF/LIS 파라미터**: 단순 Min-Max가 아닌 Power-of-two 제약 조건을 만족하는 파라미터를 탐색합니다.

### Forward Pass (추론)

* **임베딩 제외**: 사용자 요청에 따라 패치 임베딩 단계는 양자화하지 않고 원본을 유지하여 입력 정보의 해상도를 보존합니다.
* **재양자화 포인트**: 각 Linear 연산 이후와 Residual Add 이후에 반드시 `QAct`를 배치하여 비트 수가 커지는 것을 방지하고 다시 `INT8`로 정렬합니다.

---

### 💡 다음 단계 제안

이 문서 구조가 정리가 잘 되었나요? 이 내용을 바탕으로 다음 작업 중 하나를 도와드릴 수 있습니다:

1. **`QuantIntSoft` 내부의 LIS 수식** 구현 상세 정리
2. **`QLayerNorm`의 PTF(Power-of-two)** 스케일 계산 로직 문서화
3. **AdaRound**를 적용하기 위한 `get_quantized_layers` 최적화 방법

어떤 부분을 더 구체화해 드릴까요?