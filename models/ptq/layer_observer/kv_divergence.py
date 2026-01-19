# Copyright (c) MEGVII Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn as nn


from .base import BaseObserver
from .utils import lp_loss
from ..bit_type import BitType 


class KVObserver(BaseObserver):

    def __init__(self,
                 bit_type,
                 module_type,
                 calibration_mode,
                 hist_bins=2048):
        super(KVObserver, self).__init__(bit_type, module_type, calibration_mode)
                                           
        self.max_val = None
        self.min_val = None
        self.percentile_sigma = 0.01  #Fixed
        self.percentile_alpha = 0.99999     #Fixed
        self.symmetric = self.bit_type.signed
        
        self.start = 200 #Fixed
        self.hist_bins = hist_bins
        self.v = None
        self.device = None

    def update(self, v:torch.Tensor):
        if self.device is None:
            self.device = v.device.type
        
        if self.device != v.device.type:
            raise ValueError(
                "Device type mismatch in observer."
                f"Expected device type: {self.device}, but got: {v.device.type}")
        
        # CUDA histogram을 위해 CPU로 이동 ✓
        v_cpu = v.cpu() if v.device.type == 'cuda' else v
        v_reshaped = self.reshape_tensor(v_cpu)
        self.v = v_cpu.detach()
        
        #1. cal min, max value layer wise only
        self._cal_minmax(v_reshaped)

        #2 update histogram
        hist_origin, bin_edge= torch.histogram(
            v_reshaped, 
            bins=self.hist_bins, 
            range=(self.min_val.item(), self.max_val.item())
            )
        print(f"[DEBUG] Histogram shape: {hist_origin.shape}")

        #3. compute origin probability
        hist_origin_P = hist_origin / (hist_origin.sum() + self.eps)
        min_kl_div = float('inf')
        best_threshold = None

        print(f"[DEBUG] hist_origin sum: {hist_origin_P.sum().item()}")
        print(f"[DEBUG] Range: [{self.min_val:.4f}, {self.max_val:.4f}]")

        for idx in range(self.start, self.hist_bins):
            threshold = bin_edge[idx]  # start 이후로는 0으로 채움
            kl_div = self._compute_kl(v_reshaped,hist_origin_P, threshold)
            # if idx % 100 == 0:
            #     print(f"Threshold: {threshold:.4f}, KL Divergence: {kl_div:.6f}")
            if kl_div < min_kl_div:
                min_kl_div = kl_div
                best_threshold = threshold

        print(best_threshold)
        self.max_val = best_threshold
        self.min_val = -best_threshold

    def _compute_kl(self, v_reshaped, hist_origin_P, threshold):
        v_clipped = torch.clip(v_reshaped, max=threshold, min=-threshold)
        
        #quant to int8, symetiric zero_point
        scale = torch.abs(threshold) / 127.0
        v_quantized = torch.round(v_clipped / scale).clamp(-127, 127)
        v_dequantized = v_quantized * scale

        #make histogram again        
        hist_ranged, _ = torch.histogram(v_dequantized, 
                                         bins=self.hist_bins, 
                                         range=(self.min_val.item(), self.max_val.item()))
        hist_ranged_P = hist_ranged / (hist_ranged.sum() + self.eps)
        mask = hist_origin_P > 0  # P > 0인 항만
        
        frac = (hist_origin_P[mask] / (hist_ranged_P[mask] + self.eps)).log()
        kl_div = (hist_origin_P[mask] * frac).sum()

        return kl_div

    def _cal_minmax(self, v:torch.Tensor):
        # channel-wise needs too much time.
                   
        if self.calibration_mode == 'channel_wise':
            cur_max = torch.quantile(v, self.percentile_alpha, dim = 1)
            cur_min = torch.quantile(v, 1.0 - self.percentile_alpha, dim = 1)
        elif self.calibration_mode == 'layer_wise':
            cur_max = torch.quantile(v.reshape(-1), self.percentile_alpha)
            cur_min = torch.quantile(v.reshape(-1), 1.0 - self.percentile_alpha)                                           

        if self.max_val is None:
            self.max_val = cur_max.max()
            self.min_val = cur_min.min()
        else: 
            self.max_val = self.max_val + self.percentile_sigma * (cur_max - self.max_val)
            self.min_val = self.min_val + self.percentile_sigma * (cur_min - self.min_val)

    def get_quantization_params(self, *args, **kwargs):
        max_val = self.max_val
        min_val = self.min_val

        qmax = self.bit_type.upper_bound
        qmin = self.bit_type.lower_bound

        if self.symmetric:
            #symmetric quant paras
            max_val = torch.max(-min_val, max_val)
            scale = max_val / (float(qmax - qmin) / 2)
            scale.clamp_(self.eps)
            zero_point = torch.zeros_like(max_val, dtype=torch.int64)
        else:
            raise ValueError("Only symmetric quantization is supported in KVObserver.")         
        return scale, zero_point
    
def main():
    print("=" * 70)
    print("KVObserver 테스트: Layer-wise vs Channel-wise")
    print("=" * 70)
    
    # GPU 사용 가능 여부 확인
    device_list = ['cpu']
    if torch.cuda.is_available():
        device_list.append('cuda')
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("✓ CUDA not available, testing CPU only")
    
    print("\n⚠️  Note: KL Divergence는 histogram 기반으로 계산량이 많습니다.")
    print()
    
    # 테스트할 calibration 모드
    calibration_modes = ['layer_wise', 'channel_wise']
    
    # 테스트 케이스
    test_cases = [
        {
            'name': 'Conv2d Weight',
            'layer': nn.Conv2d(3, 8, kernel_size=3, padding=1),
            'input_shape': (4, 3, 32, 32),
            'module_type': 'conv_weight',
            'use_weight': True,
            'bit_type': BitType(bits=8, signed=True, name='int8_signed'),
            'hist_bins': 2048
        },
        {
            'name': 'Conv Activation',
            'layer': nn.Sequential(
                nn.Conv2d(3, 8, kernel_size=3, padding=1),
                nn.ReLU()
            ),
            'input_shape': (4, 3, 32, 32),
            'module_type': 'activation',
            'use_weight': False,
            'bit_type': BitType(bits=8, signed=True, name='int8_signed'),
            'hist_bins': 2048
        },
    ]
    
    # Device별 테스트
    for device in device_list:
        print("=" * 70)
        print(f"Device: {device.upper()}")
        print("=" * 70)
        
        for test_case in test_cases:
            for calib_mode in calibration_modes:
                print(f"\n{'='*70}")
                print(f"[{test_case['name']}] - Calibration: {calib_mode.upper()}")
                print(f"{'='*70}")
                
                # 레이어를 device로 이동
                layer = test_case['layer'].to(device)
                
                # Observer 생성
                try:
                    observer = KVObserver(
                        bit_type=test_case['bit_type'],
                        module_type=test_case['module_type'],
                        calibration_mode=calib_mode,
                        hist_bins=test_case['hist_bins']
                    )
                    
                    print(f"Observer config:")
                    print(f"  - BitType: {test_case['bit_type'].name} (signed={test_case['bit_type'].signed})")
                    print(f"  - Calibration mode: {calib_mode}")
                    print(f"  - Histogram bins: {test_case['hist_bins']}")
                    print(f"  - Module type: {test_case['module_type']}")
                    print()
                    
                    if test_case['use_weight']:
                        # Weight 테스트
                        weight = layer.weight if isinstance(layer, (nn.Conv2d, nn.Linear)) else layer[0].weight
                        print(f"Weight shape: {weight.shape}")
                        print(f"Weight device: {weight.device}")
                        print(f"Weight range: [{weight.min():.4f}, {weight.max():.4f}]")
                        print(f"Weight mean: {weight.mean():.4f}, std: {weight.std():.4f}")
                        print()
                        
                        # KL divergence 계산
                        print("Computing KL divergence...")
                        observer.update(weight)
                        
                    else:
                        # Activation 테스트
                        dummy_input = torch.randn(*test_case['input_shape']).to(device)
                        with torch.no_grad():
                            output = layer(dummy_input)
                        
                        print(f"Activation shape: {output.shape}")
                        print(f"Activation device: {output.device}")
                        print(f"Activation range: [{output.min():.4f}, {output.max():.4f}]")
                        print(f"Activation mean: {output.mean():.4f}, std: {output.std():.4f}")
                        print()
                        
                        # KL divergence 계산
                        print("Computing KL divergence...")
                        observer.update(output)
                    
                    # Quantization parameters 가져오기
                    scale, zero_point = observer.get_quantization_params()
                    
                    print(f"\n[최종 Quantization Parameters]")
                    print(f"Scale shape: {scale.shape}")
                    print(f"Scale: {scale.item():.6f}")
                    print(f"Zero Point: {zero_point.item()}")
                    print(f"Optimal threshold: [{observer.min_val:.4f}, {observer.max_val:.4f}]")
                    
                    # Symmetric 검증
                    print(f"\n[SYMMETRIC 검증]")
                    print(f"✓ Zero point == 0: {zero_point.item() == 0}")
                    print(f"✓ Symmetric threshold: {abs(observer.min_val.item()) == abs(observer.max_val.item())}")
                    
                except Exception as e:
                    print(f"\n❌ 에러 발생: {type(e).__name__}")
                    print(f"   {str(e)}")
                    import traceback
                    traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("✓ 테스트 완료!")
    print("=" * 70)
    print("\n[테스트 요약]")
    print(f"- Devices tested: {', '.join(device_list)}")
    print(f"- Calibration modes: layer_wise, channel_wise")
    print(f"- Layer types: Conv Weight, Conv Activation")
    print(f"- Histogram bins: 2048")
    print("\n[예상 동작]")
    print("✓ Layer-wise: 단일 scale/zero_point (전체 레이어)")
    print("✓ Channel-wise: 현재 구현은 layer-wise와 동일하게 동작할 가능성 있음")
    print("  (KL divergence는 구조적으로 single threshold 탐색)")

if __name__ == "__main__":
    main()