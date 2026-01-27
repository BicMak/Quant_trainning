# QKV Quantization Debug Summary

**Date**: 2026-01-26
**Status**: Root cause identified, fix pending next session

---

## Critical Discovery

### The Real Problem: QKV Linear Weight Quantization

After extensive debugging, we identified the root cause of attention quantization failure:

**QKV Linear layer weight quantization produces catastrophically wrong outputs (-10.55 dB QSNR)**

#### Evidence from `debug_qkv_fixed_input.py`:

```
Test with SAME fixed input (seed=123):

QKV Linear Layer:
  FP32 output range:   [-5.5534, 6.2902]
  Quant output range:  [-36.8068, 38.8413]  ← 6x larger!
  QSNR: -10.55 dB  ← CATASTROPHIC

Q/K/V Activation Quantization (using FP32 QKV output):
  Q activation QSNR:  39.48 dB  ← EXCELLENT
  K activation QSNR:  34.61 dB  ← EXCELLENT
  V activation QSNR:  29.15 dB  ← GOOD
```

#### Key Observations:

1. **Weight profiling shows good QSNR** (37.71 dB) but **forward pass fails** (-10.55 dB)
2. **Q/K/V activation quantization works perfectly** (29-39 dB QSNR)
3. **Output range explosion**: Quantized QKV output is 6x larger than FP32
4. This invalidates all previous hypotheses about Q/K/V activation issues

---

## Current Configuration (attn_config.yaml)

### QKV Linear Layer (THE PROBLEM):
```yaml
attn_qkv:
  weight:
    calibration_mode: channel_wise
    observer_type: MinmaxObserver      # ← Suspect this is the issue
    signed: True
    bits: 8
  output:
    output_quant_enable: False
```

### Q/K/V Activation Layers (WORKING FINE):
```yaml
attn_q_out, attn_k_out, attn_v_input:
  calibration_mode: layer_wise
  observer_type: MinmaxObserver
  signed: False
  bits: 8
```

---

## Investigation Timeline

### Phase 1: Profiler System Update ✅
- Updated all quantization layers to use batch-wise profiler
- Fixed StatProfiler duplicate `outlier_ratio` key bug
- Files modified:
  - `models/ptq/quant_linear.py`
  - `models/ptq/quant_act.py`
  - `models/ptq/quant_intSoft.py`
  - `models/ptq/quant_layernorm.py`
  - `models/ptq/layer_profiler/StatProfiler.py`

### Phase 2: Q/K/V Activation Investigation ❌
- **Initial hypothesis**: Q/K/V activation quantization causing low QSNR (11-15 dB)
- **Attempted fix**: Changed to PercentileObserver + layer_wise
- **Result**: Made it WORSE (9-10 dB QSNR)
- **Reason**: Over-aggressive clipping (90%+ data clipped)
- **Conclusion**: Wrong direction - activation quantization wasn't the problem

### Phase 3: Root Cause Isolation ✅
- Created `debug_qkv_fixed_input.py` to use identical inputs for FP32/Quantized
- **Discovery**: QKV Linear output differs by 6x in range
- **Conclusion**: Weight quantization in QKV Linear is the root cause

---

## Technical Analysis

### Why Weight Profiling Shows Good QSNR but Forward Pass Fails?

**Hypothesis**: Accumulated quantization error in matrix multiplication

```
Weight quantization QSNR: 37.71 dB    ← Individual weight values close to FP32
                    ↓
Matrix multiplication: W_quant @ X
                    ↓
Output QSNR: -10.55 dB               ← Accumulated error explodes
```

Possible causes:
1. **MinmaxObserver on weights includes outliers** that distort the scale
2. **channel_wise mode** may not be appropriate for QKV weights (3 matrices concatenated)
3. **Scale mismatch**: Weight scales may not properly align with activation ranges

### Input/Output Data Flow:

```
Input: [1, 197, 768]
  ↓
QKV Linear (quantized weights)
  ↓
Output FP32:   [-5.55, 6.29]   ← Expected
Output Quant:  [-36.81, 38.84] ← 6x larger! ← PROBLEM
  ↓
Split Q, K, V
  ↓
Q/K/V Activation Quantization
  ↓
Q, K, V outputs (QSNR 29-39 dB) ← These are fine
  ↓
Attention computation fails because Q/K/V values are already distorted
```

---

## Bugs Fixed Today

### 1. StatProfiler Duplicate Key Bug
**File**: `models/ptq/layer_profiler/StatProfiler.py:24`
```python
# REMOVED (was overwriting correct outlier_ratio):
'outlier_ratio': StatProfiler._outlier_ratio(original),
```

### 2. Profiler Check Updates
**Files**: All quantization layer classes
```python
# BEFORE:
if self.profiler.weight is not None:

# AFTER:
if len(self.profiler.weight_batch_list) > 0:
```

### 3. QAct Hardcoded Quantization Method (IDENTIFIED, NOT YET FIXED)
**File**: `models/ptq/quant_act.py:42-46`
```python
# CURRENT (WRONG):
self.quantizer = build_quantizer(
    quantizer_str='uniform',  # ← Hardcoded, ignores config
    bit_type=self.bit_type,
    module_type='activation'
)

# SHOULD BE (like QuantLinear):
self.quantizer = build_quantizer(
    quantizer_str=quant_config.quantization_method.lower(),
    bit_type=self.bit_type,
    module_type='activation'
)
```

---

## Next Steps (For Next Session)

### Priority 1: Fix QKV Linear Weight Quantization

**Option A: Change Observer Type (Recommended)**
```yaml
attn_qkv:
  weight:
    calibration_mode: channel_wise
    observer_type: PercentileObserver  # Change from MinmaxObserver
    percentile_alpha: 0.9995           # Remove outliers
    signed: True
    bits: 8
```

**Option B: Change Calibration Mode**
```yaml
attn_qkv:
  weight:
    calibration_mode: layer_wise       # Change from channel_wise
    observer_type: MinmaxObserver
    signed: True
    bits: 8
```

**Option C: Hybrid - Per-weight-matrix scales**
- QKV weight is concatenation of 3 matrices: [W_q | W_k | W_v]
- Current channel_wise: per-output-channel scale
- Consider: separate scales for Q, K, V weight matrices

### Priority 2: Fix QAct Hardcoded Quantization Method

Make QAct read `quantization_method` from config like QuantLinear does.

### Priority 3: Verify with Real Data

Current tests use `torch.randn()`. Test with actual ImageNet images.

---

## Files Created Today

1. **debug_qkv_fixed_input.py** ✅
   - Tests QKV Linear and Q/K/V activation separately with fixed input
   - Identified root cause: QKV Linear weight quantization

2. **debug_qkv/test_qkv_only.py** (created but not executed)
   - Simplified test for 4 layers only
   - Can be used for quick verification

3. **QKV_QUANTIZATION_DEBUG_SUMMARY.md** (this file)
   - Comprehensive summary of today's work

---

## Test Commands

### Main debugging script:
```bash
python debug_qkv_fixed_input.py
```

### Full attention pipeline test:
```bash
python test_embed_to_attn.py
```

### QKV-only test:
```bash
python debug_qkv/test_qkv_only.py
```

---

## Key Metrics

### Before Fix (Current State):
```
QKV Linear output:  -10.55 dB  ← MUST FIX
Q activation:       39.48 dB   ← Already good
K activation:       34.61 dB   ← Already good
V activation:       29.15 dB   ← Already good
attn_v_out:          4.81 dB   ← Propagated error from QKV
```

### Target After Fix:
```
QKV Linear output:  > 30 dB    ← Target
Q activation:       > 30 dB    ← Maintain
K activation:       > 30 dB    ← Maintain
V activation:       > 25 dB    ← Maintain
attn_v_out:         > 20 dB    ← Will improve
```

---

## Conclusion

The investigation successfully identified the root cause of attention quantization failure:

**QKV Linear layer weight quantization produces outputs with -10.55 dB QSNR, causing a 6x range explosion compared to FP32.**

Previous assumptions about Q/K/V activation quantization being the problem were incorrect. The activation quantization layers work excellently (29-39 dB QSNR). The issue lies entirely in the weight quantization of the QKV Linear layer.

Next session should focus on:
1. Testing PercentileObserver for QKV weights
2. Investigating if channel_wise is appropriate for concatenated Q/K/V weight matrix
3. Verifying the fix resolves downstream attention computation errors

---

**Session End**: 2026-01-26
