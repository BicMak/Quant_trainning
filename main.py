# Copyright (c) MEGVII Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn as nn

from layer_observer.kv_divergence import KVObserver
from layer_observer.minmax import MinmaxObserver
from layer_observer.bit_type import BitType
from layer_quantizer import build_quantizer

def main():
    print("=" * 70)
    print("테스트 시작: CPU/GPU, 여러 레이어 타입, 배치 처리")
    print("=" * 70)
    
    # 테스트할 레이어 타입들
    test_bit_type = [
        {
            'bit_type': BitType(bits=8, signed=True, name='uint8'),
            'observer_type' : 'channel_wise'
        },
        {
            'bit_type': BitType(bits=8, signed=False, name='uint8'),
            'observer_type' : 'channel_wise'
        },
        {
            'bit_type': BitType(bits=8, signed=True, name='uint8'),
            'observer_type' : 'layer_wise'
        },
        {
            'bit_type': BitType(bits=8, signed=False, name='uint8'),
            'observer_type' : 'layer_wise'
        },
    ]

    test_cases = [
        {
            'name': 'Conv2d Weight',
            'layer': nn.Conv2d(3, 16, kernel_size=3, padding=1),
            'input_shape': (4, 3, 32, 32),
            'module_type': 'conv_weight',
            'use_weight': True,
        },
        {
            'name': 'Linear Weight',
            'layer': nn.Linear(64, 32),
            'input_shape': (4, 64),
            'module_type': 'linear_weight',
            'use_weight': True,
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
        },
        {
            'name': 'Linear Activation',
            'layer': nn.Sequential(
                nn.Linear(64, 32),
                nn.Tanh()
            ),
            'input_shape': (4, 64),
            'module_type': 'activation',
            'use_weight': False,
        }
    ]
    
    # Device별 테스트
    for idx, test_case in enumerate(test_cases, 1):
        for bit_type_case in test_bit_type:
            test_case['bit_type'] = bit_type_case['bit_type']
            test_case['observer_type'] = bit_type_case['observer_type']
            print(f"\n[Test {test_case['name']}] signed: {test_case['bit_type'].signed}  - {test_case['observer_type']}")
            print("-" * 70)
                
            # 레이어와 입력을 해당 device로 이동
            layer = test_case['layer']
            dummy_input = torch.randn(*test_case['input_shape'])*3
            # Observer 생성
            observer = MinmaxObserver(
                test_case['bit_type'], 
                test_case['module_type'], 
                test_case['observer_type']
            )

            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.normal_(layer.weight, mean=0.0, std=1.0)
            else: 
                for sub_layer in layer:
                    if isinstance(sub_layer, (nn.Conv2d, nn.Linear)):
                        nn.init.normal_(sub_layer.weight, mean=0.0, std=1.0)

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
                        batch_input = torch.randn(*test_case['input_shape'])
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
    print(f"- Layer types tested: Conv Weight, Linear Weight, Conv Activation, Linear Activation")
    print(f"- Batch processing: 3 batches per test case")

if __name__ == "__main__":
    main()
