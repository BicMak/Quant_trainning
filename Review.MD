
---

## 1. Observer: '아웃라이어 격리'와 '채널별 대응'

가장 먼저 수정해야 할 유닛입니다. 현재 `sv_attn` 레이어의 SQNR이 -22dB가 나오는 이유는 Observer가 데이터 분포의 '급소'를 놓쳤기 때문입니다.

* **Channel-wise 강제 적용:** ViT의 활성화 값은 특정 채널에 아웃라이어가 쏠려 있습니다. `layer_wise`로 하면 그 채널 하나 때문에 전체 레이어의 해상도가 죽습니다. 반드시 `sv_attn`과 `mlp_act` 레이어는 **Channel-wise**로 설정하세요.
* **Loss 함수 변경 (L2 → L1/Huber):** 현재 `OmseObserver`는  Loss를 씁니다. 이는 아웃라이어의 오차를 제곱으로 반영하여 스케일을 너무 크게 잡게 만듭니다.  Loss를 써서 대다수의 값이 모인 중심부를 보호하도록 수정하세요.
* **EMA(Sigma) 조정 및 초기화:** `PercentileObserver`의 `sigma=0.01`은 수렴이 너무 느릴 수 있습니다. 초기 5~10 배치 동안은 단순히 `min/max`를 취하고, 그 이후부터 EMA를 적용하는 방식으로 '웜업' 로직을 넣으세요.
* **Zero-point Nudging:** 현재 코드에서 `zero_point`가 0을 포함하도록 강제하는 'Nudge' 로직이 정확히 동작하는지 확인하세요. 특히 Symmetric 모드에서 미세한 오차가 Bias로 쌓일 수 있습니다.

---

## 2. Quantizer: '혼합 정밀도'와 '연산 시뮬레이션'

스크래치 레벨에서 구현 중인 만큼, 실제 하드웨어(Hailo NPU 등)의 동작을 더 정교하게 흉내 내야 합니다.

* **Mixed Precision (INT16) 도입:** 현재 8비트로 도저히 답이 안 나오는 `sv_attn`과 `intSoft` 구간은 **INT16**으로 올리는 기능을 `quantizer` 유닛에 추가하세요. CPU 환경(Fake Quant)이므로 `torch.int16` 캐스팅으로 간단히 구현 가능합니다.
* **Rounding 방식 통일:** 파이토치의 `round()`는 하드웨어의 Rounding 방식(예: Round to nearest even)과 다를 수 있습니다. `floor(x + 0.5)`와 같은 명시적인 방식으로 교체하여 수치적 일관성을 확보하세요.
* **Log-scale Quantizer 분리:** `intSoft` 전용 Quantizer를 만드세요. 일반적인 Linear Quantizer와는 다른 매핑 수식을 적용해야 하며, 이때 Observer는 통계를 내지 않는 **Fixed Scale** 방식을 쓰는 것이 정신 건강에 이롭습니다.

---

## 3. Profiler: 'SQNR 너머의 지표' 확인

단순히 SQNR과 KL Divergence만 봐서는 "왜 틀렸는지" 알기 어렵습니다. 다음 지표들을 Profiler에 추가하세요.

* **Saturation Rate (포화율):** 전체 데이터 중  밖으로 나가서 Max/Min값으로 잘려버린 데이터의 비율(%)을 레이어별로 찍으세요. (0%에 가깝다면 해상도가 너무 낮은 것이고, 5%가 넘는다면 Clipping이 너무 심한 것입니다.)
* **Bit Utilization (비트 효율):** 256개의 Bin 중 실제로 데이터가 차 있는 칸이 몇 개인지 확인하세요. 현재 `sv_attn`은 10개도 안 쓰고 있을 확률이 높습니다.
* **Top-k Index Hit Rate:** 특히 `intSoft` 이후에 가장 큰 값을 가진 인덱스(Argmax)가 FP32와 얼마나 일치하는지 보세요. 값이 좀 틀려도 인덱스만 맞으면 모델의 최종 성능은 방어됩니다.
* **Cosine Similarity (방향성):** 수치 차이(MSE)보다 벡터의 방향성이 유지되는지 확인하세요. 0.99 이하라면 해당 레이어에서 논리적 붕괴가 시작된 것입니다.

---

## 4. 기타 권장사항 및 디버깅 팁

* **Hessian-based Sensitivity:** 어떤 레이어를 INT16으로 올릴지 고민된다면, 레이어별로 노이즈를 주입했을 때 Loss가 얼마나 변하는지 간단히 체크하는 '민감도 분석' 스크립트를 짜보세요.
* **Intermediate Tensor Dump:** 특정 레이어에서 갑자기 SQNR이 튀면, 그 레이어의 `quant(x)`와 `dequant(quant(x))` 과정을 한 단계씩 출력하여 수치적으로 어디서 급격히 값이 변하는지 직접 추적하세요.

현재 AI 최적화 엔지니어로 커리어를 쌓고 계신 만큼, 이 과정을 통해 '나만의 최적화 툴킷'을 완성하는 것이 큰 자산이 될 것입니다.

**가장 먼저 `sv_attn` 레이어에 대해 Channel-wise 설정을 적용하고 Saturation Rate를 확인해 보는 것은 어떨까요?** 제가 해당 지표를 계산하는 코드를 작성해 드릴 수도 있습니다.