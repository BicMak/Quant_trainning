# Copyright (c) MEGVII Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn as nn

from .base import BaseObserver
from .utils import lp_loss
from ..ptq.bit_type import BitType 


class OmseObserver(BaseObserver):
    def __init__(self, 
                 module_type, 
                 bit_type, 
                 calibration_mode):
        super(OmseObserver, self).__init__(module_type, bit_type,
                                           calibration_mode)
        self.symmetric = self.bit_type.signed
        self.max_val = None
        self.min_val = None
        self.device = None
        self.v = None

    def update(self, v):

        #device check 1st time
        if self.device is None:
            self.device = v.device.type
        
        # test device match
        if self.device != v.device.type:
            raise ValueError(
                "Device type mismatch in observer."
                f"Expected device type: {self.device}, but got: {v.device.type}")

        v = self.reshape_tensor(v)
        self.v = v.detach()

        cur_max = v.max(axis=1).values
        if self.max_val is None:
            self.max_val = cur_max
        else:
            self.max_val = torch.max(cur_max, self.max_val)
        
        cur_min = v.min(axis=1).values
        if self.min_val is None:
            self.min_val = cur_min
        else:
            self.min_val = torch.min(cur_min, self.min_val)

        if self.calibration_mode == 'layer_wise':
            self.max_val = self.max_val.max()
            self.min_val = self.min_val.min()

    def get_quantization_params(self):
        max_val = self.max_val
        min_val = self.min_val
        qmax = self.bit_type.upper_bound
        qmin = self.bit_type.lower_bound

        best_score = 1e+10
        for i in range(90):
            new_max = max_val * (1.0 - (i * 0.01))
            new_min = min_val * (1.0 - (i * 0.01))

            if self.symmetric:
                new_max = torch.max(-new_min, new_max)
                new_scale = new_max / (float(qmax - qmin) / 2)
                new_zero_point = torch.zeros_like(new_max, dtype=torch.int64)
            else:
                new_scale = (new_max - new_min) / float(qmax - qmin)
                new_zero_point = qmin - torch.round(new_min / new_scale)
                new_zero_point.clamp_(qmin, qmax)
                
            new_scale.clamp_(self.eps)
            new_scale.unsqueeze(1)
            # 명시적으로 차원 확장
            new_scale_expanded = new_scale.unsqueeze(1)  # [16] → [16, 1]
            new_zero_point_expanded = new_zero_point.unsqueeze(1)  # [16, 1]

            inputs_q = ((self.v / new_scale_expanded + new_zero_point_expanded).round().clamp(
                qmin, qmax) - new_zero_point_expanded) * new_scale_expanded

            # L_p norm minimization as described in LAPQ
            # https://arxiv.org/abs/1911.07190
            score = lp_loss(self.v, inputs_q, p=2.0, reduction='all')
            if score < best_score:
                print(f"Best score updated: {score}")
                print(f"original max: {self.max_val}, original min: {self.min_val}")
                print(f"New max: {new_max}, New min: {new_min}")
                best_score = score
                self.max_val = new_max
                self.min_val = new_min
                scale = new_scale
                zero_point = new_zero_point
        
        return scale, zero_point
    
def main():
    print("=" * 70)
    print("OmseObserver 테스트: Grid Search 기반 최적화")
    print("=" * 70)
    
    # GPU 사용 가능 여부 확인
    device_list = ['cpu']
    if torch.cuda.is_available():
        device_list.append('cuda')
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("✓ CUDA not available, testing CPU only")
    
    print("\n⚠️  Note: OMSE는 grid search 방식으로 계산량이 많아 시간이 걸립니다.")
    print()
    
    # 테스트할 레이어 타입들
    test_cases = [
        {
            'name': 'Conv2d Weight - Symmetric',
            'layer': nn.Conv2d(3, 8, kernel_size=3, padding=1),  # 채널 수 줄임 (속도)
            'input_shape': (4, 3, 32, 32),
            'module_type': 'conv_weight',
            'use_weight': True,
            'bit_type': BitType(bits=8, signed=True, name='int8_signed'),
        },
        {
            'name': 'Conv2d Weight - Asymmetric',
            'layer': nn.Conv2d(3, 8, kernel_size=3, padding=1),
            'input_shape': (4, 3, 32, 32),
            'module_type': 'conv_weight',
            'use_weight': True,
            'bit_type': BitType(bits=8, signed=False, name='uint8'),
        },
        {
            'name': 'Linear Weight - Symmetric',
            'layer': nn.Linear(64, 16),  # output 차원 줄임 (속도)
            'input_shape': (4, 64),
            'module_type': 'linear_weight',
            'use_weight': True,
            'bit_type': BitType(bits=8, signed=True, name='int8_signed'),
        },
        {
            'name': 'Conv Activation - Asymmetric (ReLU)',
            'layer': nn.Sequential(
                nn.Conv2d(3, 8, kernel_size=3, padding=1),
                nn.ReLU()
            ),
            'input_shape': (4, 3, 16, 16),  # 크기 줄임 (속도)
            'module_type': 'activation',
            'use_weight': False,
            'bit_type': BitType(bits=8, signed=False, name='uint8'),
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
            observer = OmseObserver(
                test_case['bit_type'],
                test_case['module_type'], 
                'channel_wise'
            )
            
            is_symmetric = test_case['bit_type'].signed
            print(f"Quantization Mode: {'SYMMETRIC' if is_symmetric else 'ASYMMETRIC'}")
            print(f"BitType: {test_case['bit_type'].name} (bits={test_case['bit_type'].bits}, signed={test_case['bit_type'].signed})")
            print(f"Range: [{test_case['bit_type'].lower_bound}, {test_case['bit_type'].upper_bound}]")
            print()
            
            if test_case['use_weight']:
                # Weight 테스트
                weight = layer.weight if isinstance(layer, (nn.Conv2d, nn.Linear)) else layer[0].weight
                print(f"Weight shape: {weight.shape}")
                print(f"Weight device: {weight.device}")
                print(f"Weight range: [{weight.min():.4f}, {weight.max():.4f}]")
                print()
                
                # Update
                print("Grid search 시작 (90 iterations)...")
                observer.update(weight)
                
            else:
                # Activation 테스트
                dummy_input = torch.randn(*test_case['input_shape']).to(device)
                with torch.no_grad():
                    output = layer(dummy_input)
                
                print(f"Activation shape: {output.shape}")
                print(f"Activation device: {output.device}")
                print(f"Activation range: [{output.min():.4f}, {output.max():.4f}]")
                print()
                
                # Update
                print("Grid search 시작 (90 iterations)...")
                observer.update(output)
            
            # Quantization parameters 계산 (grid search 실행)
            print("\nGrid search progress:")
            print("-" * 50)
            scale, zero_point = observer.get_quantization_params()
            print("-" * 50)
            
            print(f"\n[최종 Quantization Parameters]")
            print(f"Scale shape: {scale.shape}")
            
            if scale.numel() > 1:
                print(f"Scale mean: {scale.mean():.6f}")
                print(f"Scale range: [{scale.min():.6f}, {scale.max():.6f}]")
                print(f"Sample scales (first 5): {scale.flatten()[:5].tolist()}")
            else:
                print(f"Scale: {scale.item():.6f}")
            
            print(f"\nZero Point shape: {zero_point.shape}")
            if zero_point.numel() > 1:
                print(f"Zero Point range: [{zero_point.min()}, {zero_point.max()}]")
                print(f"Sample zero points (first 5): {zero_point.flatten()[:5].tolist()}")
            else:
                print(f"Zero Point: {zero_point.item()}")
            
            # Symmetric/Asymmetric 검증
            print(f"\n[{('SYMMETRIC' if is_symmetric else 'ASYMMETRIC')} 검증]")
            
            if is_symmetric:
                all_zero = (zero_point == 0).all().item()
                print(f"✓ Zero point == 0: {all_zero}")
                if not all_zero:
                    print(f"  ⚠️ WARNING: Symmetric mode인데 zero_point가 0이 아닌 값이 있습니다!")
            else:
                has_nonzero = (zero_point != 0).any().item()
                print(f"✓ Has non-zero zero_points: {has_nonzero}")
                if has_nonzero:
                    nonzero_count = (zero_point != 0).sum().item()
                    total_count = zero_point.numel()
                    print(f"  Non-zero zero_points: {nonzero_count}/{total_count}")
            
            print()
    
    print("\n" + "=" * 70)
    print("✓ 모든 테스트 완료!")
    print("=" * 70)
    print("\n[테스트 요약]")
    print(f"- Devices tested: {', '.join(device_list)}")
    print(f"- Layer types tested: Conv Weight, Linear Weight, Conv Activation")
    print(f"- Optimization method: Grid search (90 iterations) with L2 loss")
    print(f"- Total test cases: {len(device_list) * len(test_cases)}")
    print("\n[OMSE 특징]")
    print("✓ Grid search로 최적의 clipping range 탐색")
    print("✓ L2 loss (MSE) 최소화")
    print("✓ 계산량이 많아 MinMax/Percentile보다 느림")
    print("✓ 더 정확한 quantization 가능")

if __name__ == "__main__":
    main()
