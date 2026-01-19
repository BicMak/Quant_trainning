# Copyright (c) MEGVII Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn as nn

from ..bit_type import BitType 
from .base import BaseObserver

class PercentileObserver(BaseObserver):
    def __init__(self,
                 bit_type,
                 module_type,
                 calibration_mode,
                 percentile_sigma=0.01,
                 percentile_alpha=0.99999):
        super(PercentileObserver, self).__init__(bit_type=bit_type,
                                                  module_type=module_type,
                                                  calibration_mode=calibration_mode)
        self.percentile_sigma = percentile_sigma
        self.percentile_alpha = percentile_alpha
        self.bit_type = bit_type
        self.symmetric = self.bit_type.signed
        self.max_val = None
        self.min_val = None
        self.device = None


    def update(self, v:torch.Tensor):
        #device check 1st time
        if self.device is None:
            self.device = v.device.type
        
        # test device match
        if self.device != v.device.type:
            raise ValueError(
                "Device type mismatch in observer."
                f"Expected device type: {self.device}, but got: {v.device.type}")
       
        v = self.reshape_tensor(v)

        if self.calibration_mode == 'channel_wise':
            cur_max = torch.quantile(v, self.percentile_alpha, dim = 1)
            cur_min = torch.quantile(v, 1.0 - self.percentile_alpha, dim = 1)
        elif self.calibration_mode == 'layer_wise':
            cur_max = torch.quantile(v.reshape(-1), self.percentile_alpha)
            cur_min = torch.quantile(v.reshape(-1), 1.0 - self.percentile_alpha)                                           

        if self.max_val is None:
            self.max_val = cur_max
        else:
            self.max_val = self.max_val + \
                self.percentile_sigma * (cur_max - self.max_val)
        
        if self.min_val is None:
            self.min_val = cur_min
        else:
            self.min_val = self.min_val + \
                self.percentile_sigma * (cur_min - self.min_val)

    def get_quantization_params(self, *args, **kwargs):
        max_val = self.max_val
        min_val = self.min_val

        qmax = self.bit_type.upper_bound
        qmin = self.bit_type.lower_bound

        scale = torch.ones_like(max_val, dtype=torch.float32)
        zero_point = torch.zeros_like(max_val, dtype=torch.int64)

        if self.symmetric:
            max_val = torch.max(-min_val, max_val)
            scale = max_val / (float(qmax - qmin) / 2)
            scale.clamp_(self.eps)
            zero_point = torch.zeros_like(max_val, dtype=torch.int64)
        else:
            scale = (max_val - min_val) / float(qmax - qmin)
            scale.clamp_(self.eps)
            zero_point = qmin - torch.round(min_val / scale)
            zero_point.clamp_(qmin, qmax)
        return scale, zero_point

def main():
    print("=" * 70)
    print("PercentileObserver 테스트: Symmetric vs Asymmetric Quantization")
    print("=" * 70)
    
    # GPU 사용 가능 여부 확인
    device_list = ['cpu']
    if torch.cuda.is_available():
        device_list.append('cuda')
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("✓ CUDA not available, testing CPU only")
    
    print("\n")
    
    # Symmetric vs Asymmetric 비교를 위한 테스트 케이스
    test_cases = [
        {
            'name': 'Conv2d Weight - Symmetric',
            'layer': nn.Conv2d(3, 16, kernel_size=3, padding=1),
            'input_shape': (4, 3, 32, 32),
            'module_type': 'conv_weight',
            'use_weight': True,
            'bit_type': BitType(bits=8, signed=True, name='int8_signed'),  # symmetric
            'percentile_sigma': 0.1,
            'percentile_alpha': 0.99
        },
        {
            'name': 'Conv2d Weight - Asymmetric',
            'layer': nn.Conv2d(3, 16, kernel_size=3, padding=1),
            'input_shape': (4, 3, 32, 32),
            'module_type': 'conv_weight',
            'use_weight': True,
            'bit_type': BitType(bits=8, signed=False, name='uint8'),  # asymmetric
            'percentile_sigma': 0.1,
            'percentile_alpha': 0.99
        },
        {
            'name': 'Linear Weight - Symmetric',
            'layer': nn.Linear(64, 32),
            'input_shape': (4, 64),
            'module_type': 'linear_weight',
            'use_weight': True,
            'bit_type': BitType(bits=8, signed=True, name='int8_signed'),
            'percentile_sigma': 0.1,
            'percentile_alpha': 0.99
        },
        {
            'name': 'Linear Weight - Asymmetric',
            'layer': nn.Linear(64, 32),
            'input_shape': (4, 64),
            'module_type': 'linear_weight',
            'use_weight': True,
            'bit_type': BitType(bits=8, signed=False, name='uint8'),
            'percentile_sigma': 0.1,
            'percentile_alpha': 0.99
        },
        {
            'name': 'Conv Activation - Symmetric',
            'layer': nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, padding=1),
                nn.Tanh()  # Tanh는 음수값도 나옴
            ),
            'input_shape': (4, 3, 32, 32),
            'module_type': 'activation',
            'use_weight': False,
            'bit_type': BitType(bits=8, signed=True, name='int8_signed'),
            'percentile_sigma': 0.1,
            'percentile_alpha': 0.99
        },
        {
            'name': 'Conv Activation - Asymmetric (ReLU)',
            'layer': nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, padding=1),
                nn.ReLU()  # ReLU는 0 이상만
            ),
            'input_shape': (4, 3, 32, 32),
            'module_type': 'activation',
            'use_weight': False,
            'bit_type': BitType(bits=8, signed=False, name='uint8'),
            'percentile_sigma': 0.1,
            'percentile_alpha': 0.99
        },
    ]
    
    # Device별 테스트
    for device in device_list:
        print("=" * 70)
        print(f"Device: {device.upper()}")
        print("=" * 70)
        
        for idx, test_case in enumerate(test_cases, 1):
            print(f"\n[Test {idx}] {test_case['name']}")
            print("-" * 70)
            
            # 레이어와 입력을 해당 device로 이동
            layer = test_case['layer'].to(device)
            
            # Observer 생성
            observer = PercentileObserver(
                test_case['bit_type'],
                test_case['module_type'], 
                'channel_wise',
                percentile_sigma=test_case['percentile_sigma'],
                percentile_alpha=test_case['percentile_alpha']
            )
            
            is_symmetric = test_case['bit_type'].signed
            print(f"Quantization Mode: {'SYMMETRIC' if is_symmetric else 'ASYMMETRIC'}")
            print(f"BitType: {test_case['bit_type'].name} (bits={test_case['bit_type'].bits}, signed={test_case['bit_type'].signed})")
            print(f"Range: [{test_case['bit_type'].lower_bound}, {test_case['bit_type'].upper_bound}]")
            print()
            
            # 배치별 동적 업데이트
            num_batches = 5
            
            if test_case['use_weight']:
                # Weight 테스트
                weight = layer.weight if isinstance(layer, (nn.Conv2d, nn.Linear)) else layer[0].weight
                print(f"Weight shape: {weight.shape}")
                print(f"Weight range: [{weight.min():.4f}, {weight.max():.4f}]")
                print()
                
                # 여러 배치로 update
                for batch_idx in range(num_batches):
                    observer.update(weight)
                
            else:
                # Activation 테스트
                print(f"Activation shape: {test_case['input_shape']}")
                print()
                
                for batch_idx in range(num_batches):
                    with torch.no_grad():
                        batch_input = torch.randn(*test_case['input_shape']).to(device)
                        batch_output = layer(batch_input)
                        
                        if batch_idx == 0:
                            print(f"Sample activation range: [{batch_output.min():.4f}, {batch_output.max():.4f}]")
                        
                        observer.update(batch_output)
            
            # 최종 Quantization parameters 계산
            scale, zero_point = observer.get_quantization_params()
            
            print(f"\n[최종 Quantization Parameters]")
            print(f"Scale shape: {scale.shape}")
            
            if scale.numel() > 1:
                print(f"Scale mean: {scale.mean():.6f}")
                print(f"Scale range: [{scale.min():.6f}, {scale.max():.6f}]")
                print(f"Sample scales (first 3): {scale.flatten()[:3].tolist()}")
            else:
                print(f"Scale: {scale.item():.6f}")
            
            print(f"\nZero Point shape: {zero_point.shape}")
            if zero_point.numel() > 1:
                print(f"Zero Point range: [{zero_point.min()}, {zero_point.max()}]")
                print(f"Sample zero points (first 3): {zero_point.flatten()[:3].tolist()}")
            else:
                print(f"Zero Point: {zero_point.item()}")
            
            # Symmetric/Asymmetric 검증
            print(f"\n[{('SYMMETRIC' if is_symmetric else 'ASYMMETRIC')} 검증]")
            
            if is_symmetric:
                # Symmetric: zero_point는 모두 0이어야 함
                all_zero = (zero_point == 0).all().item()
                print(f"✓ Zero point == 0: {all_zero}")
                if not all_zero:
                    print(f"  ⚠️ WARNING: Symmetric mode인데 zero_point가 0이 아닌 값이 있습니다!")
                    print(f"  Non-zero values: {zero_point[zero_point != 0]}")
                
                # Symmetric: qmin과 qmax가 대칭이어야 함
                qmin = test_case['bit_type'].lower_bound
                qmax = test_case['bit_type'].upper_bound
                print(f"✓ Quantization range: [{qmin}, {qmax}]")
                print(f"  Range is symmetric: {abs(qmin) == qmax or abs(qmin) == qmax + 1}")
                
            else:
                # Asymmetric: zero_point가 0이 아닐 수 있음
                has_nonzero = (zero_point != 0).any().item()
                print(f"✓ Has non-zero zero_points: {has_nonzero}")
                if has_nonzero:
                    nonzero_count = (zero_point != 0).sum().item()
                    total_count = zero_point.numel()
                    print(f"  Non-zero zero_points: {nonzero_count}/{total_count}")
                
                # Asymmetric: qmin은 0이어야 함 (unsigned)
                qmin = test_case['bit_type'].lower_bound
                qmax = test_case['bit_type'].upper_bound
                print(f"✓ Quantization range: [{qmin}, {qmax}]")
                print(f"  qmin == 0: {qmin == 0}")
            
            print()
    
    print("\n" + "=" * 70)
    print("✓ 모든 테스트 완료!")
    print("=" * 70)
    print("\n[테스트 요약]")
    print(f"- Devices tested: {', '.join(device_list)}")
    print(f"- Symmetric 테스트: Conv Weight, Linear Weight, Conv Activation (Tanh)")
    print(f"- Asymmetric 테스트: Conv Weight, Linear Weight, Conv Activation (ReLU)")
    print(f"- Batches per test: {num_batches}")
    print(f"- Total test cases: {len(device_list) * len(test_cases)}")
    print("\n[검증 항목]")
    print("✓ Symmetric: zero_point == 0, 대칭 범위 (-128~127)")
    print("✓ Asymmetric: zero_point != 0 가능, 비대칭 범위 (0~255)")

if __name__ == "__main__":
    main()