# block 단위에서 양자화가 안나왔던 이유


### Query, Key, Value의 가중치 차이로 인한 문제
```python
# input shape (B, N, D)
# qkv shape: (B, N, 3*C) = (B, N, 2304)
qkv = self.attn_qkv.forward(x)
```
- 상기 코드에서 보면 연산상의 편의를 위해서 QKV를 하나의 Linear로 구현되어 있지만 양자화 입장에서는 좋은선택이 아님
- channel wise 양자화를 하게 되면 임베딩 채널을 피봇으로 해서 양자화를 진행하게 됨
- 하지만 Q,K,V 의 각각의 블록에서필요한 weight의 range가 서로 다르기 때문에 channel wise로 하게 되면 scale의 값이 커지게 됨
- 따라서 QKV를 분리를 하고 나서 양자화를 해야함

```python
# QKV projection (output_quant_enable=False이므로FP32 출력)
# qkv shape: (B, N, 3*C) = (B, N, 2304)
qkv = self.attn_qkv.forward(x)

# QKV 분리: Linear 직후 바로 슬라이싱 
# qkv: (B, N, 2304) → q, k, v 각각 (B, N, 768)
q = qkv[:, :, :C]              # (B, N, C)
k = qkv[:, :, C:2*C]           # (B, N, C)
v = qkv[:, :, 2*C:]            # (B, N, C)
# Q, K, V 각각 양자화 (아직 head 분리 전)
q = self.attn_q_out.forward(q)  # (B, N, C)
k = self.attn_k_out.forward(k)  # (B, N, C)
v = self.attn_v_input.forward(v)  # (B, N, C)
```
- 아래 이미지를 보면 linear 가중치 적용 후 query와 key는 유사한 분포를 가지고 있지만 value 같은경우에는 상단의 2개의 값과 분포가 차이가 발생함
![alt text](/docs/images/image-1.png)
![alt text](/docs/images/image-1.png)
![alt text](/docs/images/image-2.png)

### 중복 양자화 문제
```python 
 dequant_weight = self.weight_quantizer.dequantize(self.quant_weight)
 # 2. Linear operation in fp32
 out_fp32 = self.fwd_func(x, dequant_weight, self.bias, **self.fwd_kwargs)
 # 3. Output quantization (선택적)
 if self.output_quant_enable:
     # Output quantization이 활성화된 경우
     if self.enable_profiling and self.output_profiler is not None:
         with self.output_profiler.measure_time():
             # Store FP32 output before quantization
             fp32_output = out_fp32.clone().detach()
             # Output fake quantization
             output = self.output_quantizer.forward(out_fp32)
             # Update output profiler with FP32 vs Quantized outputs
             self.output_profiler.update_weight(fp32_output, output.detach())
```

- 실제 구현에서 forward(), 순전파를 구현할때 정확도를 떨어트리는 중요 요인
- self.quant_weight 가 사전에 양자화가 되어있는 상태였다면, 해당구간에서 순전파를 했다면 양자화가 2번 적용되어서 가중치값이 손실이 누적됨

### Per-channel로 양자화르 했을떄 오히려 스코어가 떨어 질 수 있음

- Per-channel은 각 채널마다 독립적인 스케일을 계산, 즉 vit 기준으로 patch가 197개가 존재하고 이것만으로 양자화를 계산하기 때문에 작은 outlier에 예민하게 값이 변함
- 채널 A와 채널 B 사이의 상대적 관계를 반영하지 않아 채널간의 편차를 무시하게 됨

  * **Per-tensor (Layer-wise)일 때:** 하나의 큰 스케일로 전체를 나누기 때문에, 양자화 후에도 B는 A보다 대략 3배 큰 정수 값을 유지합니다. **상대적 관계가 보존됩니다.**
  * **Per-channel일 때:**
  * 채널 A는  범위에 맞춰 스케일을 잡고 INT8의 를 꽉 채웁니다.
  * 채널 B는  범위에 맞춰 스케일을 잡고 INT8의 를 꽉 채웁니다.
  * **결과:** 양자화된 공간(Integer Domain)에서 보면 A와 B 모두 비슷한  근처의 값을 가지게 됩니다. 하드웨어가 다시 이를 복원(Dequantize)할 때 각기 다른 스케일을 곱해주긴 하지만, 이 과정에서 발생하는 **반올림 오차(Rounding Error)의 크기**가 채널별로 달라지면서 원래의 "3배 차이"라는 정교한 비율이 미세하게 틀어집니다.



