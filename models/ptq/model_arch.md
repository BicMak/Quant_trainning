

# 🤖 AI Quantization Layer Architecture Design

이 설계는 모델의 연산 로직과 양자화 파라미터 산출, 그리고 성능 측정을 완전히 분리하여 **확장성**과 **메모리 효율성**을 극대화하는 것을 목표로 합니다.

---

## 1. 개별 클래스 R&R (Roles & Responsibilities)

### 🟢 Observer (관찰자)

* **역할:** 양자화 파라미터 결정의 '뇌' 역할.
* **핵심 기능:** * 입력된 데이터의 분포(Min/Max, Histogram 등)를 분석.
* 양자화에 필요한 **Scale()**과 **Zero-point()**를 산출.


* **특징:** 수치 최적화 알고리즘(KL Divergence, MSE 등)을 내포함.

### 🔵 Quantizer (양자화기)

* **역할:** 실제 수치 변환 수행.
* **핵심 기능:** * Forward 시 **Fake Quantization**() 수행.
* STP(Straight Through Estimator) 등을 통해 역전파 시 gradient 전달 유지.


* **특징:** 레이어 연산 함수(`torch.F`)와 결합하여 실제 출력을 변환함.

### 🟣 Profiler (분석기)

* **역할:** 연산 결과 모니터링 및 성능 측정.
* **핵심 기능:** * **Stat:** 양자화 전/후의 오차(SQNR, MSE) 측정, forward로 추론이 되는 그순간에 대한 결과값만 캡쳐해서 저장 하도록 구현
* **Hist:** 데이터 분포 시각화용 데이터 수집.
* **System:** 추론 시간(Time) 및 메모리(Memory) 점유율 측정.


* **메모리 관리:** 텐서 복제가 아닌 **포인터 참조(Reference)** 방식을 사용하여 메모리 2배 증가 방지. 통계치 계산 직후 참조를 해제(`None`)하여 배치가 끝나면 즉시 가비지 컬렉션 유도.

---

## 2. 상단 레이어 (Wrapper/Manager)의 역할

상단 레이어는 실제 `torch.nn.Module`을 상속받아 외부 인터페이스를 담당하는 **오케스트레이터**입니다.

1. **객체 초기화:** 내부적으로 `Observer`, `Quantizer`, `Profiler`를 생성하고 조립함.
2. **Module Swapping:** 기존 PyTorch 기본 레이어(Linear, Conv2d)를 감싸거나 대체하여 양자화 전용 레이어로 전환.
3. **인터페이스 단일화:** 외부(Trainer/Inference 엔진)와의 소통 창구 역할. 내부 복잡한 계산 로직을 은닉함.
4. **연산 위임:** 레이어 자체는 직접 수치를 계산하지 않고 `torch.nn.functional` 기반의 연산 함수만 호출함.
5. **모드 제어:** * **Weight:** 초기 1회 또는 특정 주기마다 업데이트 후 고정(Static).
* **Activation:** 실시간 배치마다 갱신(Dynamic)하며 Profiling 수행.



---

## 3. 데이터 흐름 및 타겟팅 (Data Flow)

| 대상 (Target) | 수집 시점 | 프로파일링 방식 | 특징 |
| --- | --- | --- | --- |
| **Activation (출력)** | 매 Forward 시 | **실시간 갱신** | 배치가 지나갈 때만 참조하고 즉시 해제 |
| **Weight (가중치)** | 초기화/학습 완료 시 | **1회성 업데이트** | 고정된 값으로 유지하여 연산 부하 최소화 |

---

## 4. 메모리 최적화 핵심 전략

> **"관심 없는 데이터는 소유하지 않는다."**

* **포인터 참조:** Profiler가 레이어의 출력을 가져갈 때 `detach()`된 참조만 활용하여 연산 그래프와 메모리 중복 방지.
* **증분 계산(Incremental):** 전체 배치 데이터를 쌓아두지 않고, `Mean`, `MSE` 등은 배치 단위로 누적 합산하여 RAM 터짐 방지.
* **Life-cycle 관리:** 최종 결과값(통계치) 도출이 끝나면 원본 텐서 변수를 `None`으로 밀어버려 즉시 메모리 회수.

---

**추후 진행 사항:**

* [ ] `StatProfiler` 내부에 RAM 보호를 위한 **Incremental Average** 로직 구현.
* [ ] `nn.Conv2d`를 자동으로 `QuantizedConv2d`로 변환하는 **Swapping logic** 작성.

---

정리해 드린 내용 중 수정이 필요하거나, 구체적으로 더 명시하고 싶은 기술적 디테일이 있다면 말씀해 주세요! 이 내용을 바탕으로 실제 코드 구현 단계로 넘어가 볼까요?