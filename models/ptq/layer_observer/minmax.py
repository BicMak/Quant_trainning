# Copyright (c) MEGVII Inc. and its affiliates.
# All Rights Reserved.
import torch
import torch.nn as nn
from ..bit_type import BitType 
from .base import BaseObserver

class MinmaxObserver(BaseObserver):
    def __init__(self, 
                 bit_type, 
                 module_type, 
                 calibration_mode):
        super(MinmaxObserver, self).__init__(bit_type, 
                                             module_type,
                                             calibration_mode)
        self.symmetric = self.bit_type.signed
        self.max_val = None
        self.min_val = None
        self.device = None

    def update(self, v):
        #1. update self.max_val and self.min_val

        #device check 1st time
        if self.device is None:
            self.device = v.device.type
        
        # test device match
        if self.device != v.device.type:
            raise ValueError(
                "Device type mismatch in observer."
                f"Expected device type: {self.device}, but got: {v.device.type}")

        v = self.reshape_tensor(v)
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

    def get_quantization_params(self, *args, **kwargs):
        max_val = self.max_val
        min_val = self.min_val

        qmax = self.bit_type.upper_bound
        qmin = self.bit_type.lower_bound

        scale = torch.ones_like(max_val, dtype=torch.float32)
        zero_point = torch.zeros_like(max_val, dtype=torch.int64)

        if self.symmetric:
            #symmetric quant paras
            max_val = torch.max(-min_val, max_val)
            scale = max_val / (float(qmax - qmin) / 2)
            scale.clamp_(self.eps)
            zero_point = torch.zeros_like(max_val, dtype=torch.int64)
        else:
            #asymmetric quant paras
            scale = (max_val - min_val) / float(qmax - qmin)
            scale.clamp_(self.eps)
            zero_point = qmin - torch.round(min_val / scale)
            zero_point.clamp_(qmin, qmax)
        
        return scale, zero_point

def main():
    print("=" * 70)
    print("테스트 시작: CPU/GPU, 여러 레이어 타입, 배치 처리")
    print("=" * 70)
    
    # GPU 사용 가능 여부 확인
    device_list = ['cpu']
    if torch.cuda.is_available():
        device_list.append('cuda')
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("✓ CUDA not available, testing CPU only")
    
    print("\n")
    
    # 테스트할 레이어 타입들
    test_cases = [
        {
            'name': 'Conv2d Weight',
            'layer': nn.Conv2d(3, 16, kernel_size=3, padding=1),
            'input_shape': (4, 3, 32, 32),
            'module_type': 'conv_weight',
            'use_weight': True,
            'bit_type': BitType(bits=8, signed=True, name='int8_signed')
        },
        {
            'name': 'Linear Weight',
            'layer': nn.Linear(64, 32),
            'input_shape': (4, 64),
            'module_type': 'linear_weight',
            'use_weight': True,
            'bit_type': BitType(bits=8, signed=True, name='int8_signed')
        },
        {
            'name': 'Conv Activation (ReLU)',
            'layer': nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, padding=1),
                nn.ReLU()
            ),
            'input_shape': (4, 3, 32, 32),
            'module_type': 'activation',
            'use_weight': False,
            'bit_type': BitType(bits=8, signed=False, name='uint8')
        },
        {
            'name': 'Linear Activation',
            'layer': nn.Sequential(
                nn.Linear(64, 32),
                nn.ReLU()
            ),
            'input_shape': (4, 64),
            'module_type': 'activation',
            'use_weight': False,
            'bit_type': BitType(bits=8, signed=False, name='uint8')
        }
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
            dummy_input = torch.randn(*test_case['input_shape']).to(device)
            
            # Observer 생성
            observer = MinmaxObserver(
                test_case['bit_type'], 
                test_case['module_type'], 
                'channel_wise'
            )
            
            if test_case['use_weight']:
                # Weight 테스트
                weight = layer.weight if isinstance(layer, (nn.Conv2d, nn.Linear)) else layer[0].weight
                print(f"Weight shape: {weight.shape}")
                print(f"Weight device: {weight.device}")
                print(f"Weight range: [{weight.min():.4f}, {weight.max():.4f}]")
                
                # 배치 처리 시뮬레이션 (같은 weight를 3번 update)
                for batch_idx in range(3):
                    observer.update(weight)
                
                print(f"✓ Batch updates completed: 3 batches")
                
            else:
                # Activation 테스트
                with torch.no_grad():
                    output = layer(dummy_input)
                
                print(f"Activation shape: {output.shape}")
                print(f"Activation device: {output.device}")
                print(f"Activation range: [{output.min():.4f}, {output.max():.4f}]")
                
                # 배치 처리 시뮬레이션 (서로 다른 입력으로 3번 update)
                for batch_idx in range(3):
                    with torch.no_grad():
                        batch_input = torch.randn(*test_case['input_shape']).to(device)
                        batch_output = layer(batch_input)
                        observer.update(batch_output)
                
                print(f"✓ Batch updates completed: 3 batches")
            
            # Quantization parameters 계산
            scale, zero_point = observer.get_quantization_params()
            
            print(f"\n[Quantization Parameters]")
            print(f"BitType: {test_case['bit_type'].name} (bits={test_case['bit_type'].bits}, signed={test_case['bit_type'].signed})")
            print(f"Scale shape: {scale.shape}")
            print(f"Scale device: {scale.device}")
            print(f"Scale range: [{scale.min():.6f}, {scale.max():.6f}]")
            print(f"Zero point range: [{zero_point.min()}, {zero_point.max()}]")
            
            # Channel별 샘플 출력 (최대 5개)
            num_samples = min(5, scale.numel())
            if scale.numel() > 1:
                print(f"Sample scales (first {num_samples}): {scale.flatten()[:num_samples].tolist()}")
                print(f"Sample zero points (first {num_samples}): {zero_point.flatten()[:num_samples].tolist()}")
            else:
                print(f"Scale: {scale.item():.6f}")
                print(f"Zero point: {zero_point.item()}")
    
    print("\n" + "=" * 70)
    print("✓ 모든 테스트 완료!")
    print("=" * 70)
    print("\n[테스트 요약]")
    print(f"- Devices tested: {', '.join(device_list)}")
    print(f"- Layer types tested: Conv Weight, Linear Weight, Conv Activation, Linear Activation")
    print(f"- Batch processing: 3 batches per test case")
    print(f"- Total test cases: {len(device_list) * len(test_cases)}")

if __name__ == "__main__":
    main()
