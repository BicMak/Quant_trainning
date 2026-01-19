# Copyright (c) MEGVII Inc. and its affiliates.
# All Rights Reserved.
import torch
import torch.nn as nn
from collections import namedtuple
from bit_type import BitType 
from base import BaseObserver
import matplotlib.pyplot as plt
import numpy as np

def visualize_histogram_overlay(hist_conv, min_val_conv, max_val_conv):
    """모든 채널을 한 그래프에 overlay"""
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    num_channels = len(hist_conv) if isinstance(hist_conv, list) else hist_conv.shape[0]
    
    for c in range(min(num_channels, 16)):  # 16개까지만
        hist = hist_conv[c].cpu().numpy() if hasattr(hist_conv[c], 'cpu') else hist_conv[c].numpy()
        min_val = float(min_val_conv[c])
        max_val = float(max_val_conv[c])
        
        bins = np.linspace(min_val, max_val, len(hist) + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        ax.plot(bin_centers, hist, label=f'Channel {c}', alpha=0.6, linewidth=2)
    
    ax.set_xlabel('Weight Value', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Conv Weight Distribution by Channel', fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('conv_weight_histogram_overlay.png', dpi=150, bbox_inches='tight')
    plt.show()

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

    def update(self, v):
        #1. update self.max_val and self.min_val
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



    def get_histogram_params(self, v):
        v = self.reshape_tensor(v)

        if self.calibration_mode == 'layer_wise':
            hist, _ = torch.histogram(v, bins=256, range=(self.min_val, self.max_val))
            if self.hist_val is None:
                self.hist_val = hist
            else:
                self.hist_val += hist  # 누적
        else:
            self.hist_val = []
            for c in range(v.shape[0]):
                v_ch = v[:, c]
                print("Channel:", c)
                print(self.min_val.shape)
                print(f"min_val shape: {self.min_val.shape}, max_val shape: {self.max_val.shape}")
                print(f"Min val: {self.min_val[c]}, Max val: {self.max_val[c]}")
                min_val = float(self.min_val[c])
                max_val = float(self.max_val[c])

                hist, _ = torch.histogram(v_ch, bins=256, range=(min_val, max_val))
                self.hist_val.append(hist)
        return self.hist_val, (self.min_val, self.max_val)

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
    # BitType 설정: int8 (signed=True)
    bit_type = BitType(bits=8, signed=False, name='int8')
    
    # 더미 데이터 생성
    batch_size = 4
    channels = 16
    height, width = 32, 32
    
    print("=" * 70)
    print("1. Conv Layer Test")
    print("=" * 70)
    
    # Conv2d 레이어 생성 및 통과
    conv = nn.Conv2d(3, channels, kernel_size=3, padding=1)
    dummy_input = torch.randn(batch_size, 3, height, width)
    conv_output = conv(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Conv weight shape: {conv.weight.shape}")
    print(f"Conv weight range: [{conv.weight.min():.4f}, {conv.weight.max():.4f}]")
    
    observer_conv = MinmaxObserver(bit_type, 'conv_weight', 'channel_wise')
    observer_conv.update(conv.weight)
    hist_conv, (min_val_conv, max_val_conv) = observer_conv.get_histogram_params(conv.weight)
    visualize_histogram_overlay(hist_conv, min_val_conv, max_val_conv)


    scale_conv, zp_conv = observer_conv.get_quantization_params()
    
    print(f"Scale shape: {scale_conv.shape}")
    print(f"Scale: {scale_conv}")
    print(f"Zero Point: {zp_conv}")
    
    # ===================================================================
    print("\n" + "=" * 70)
    print("2. Activation (ReLU) Test")
    print("=" * 70)
    relu_bit_type = BitType(bits=8, signed=False, name='int8')
    print(f"Activation BitType: {relu_bit_type.name}, Bits: {relu_bit_type.bits}, Signed: {relu_bit_type.signed}")

    relu = nn.ReLU()
    relu_output = relu(conv_output)
    
    print(f"ReLU output shape: {relu_output.shape}")
    print(f"ReLU output range: [{relu_output.min():.4f}, {relu_output.max():.4f}]")
    
    observer_relu = MinmaxObserver(relu_bit_type, 'activation', 'channel_wise')
    observer_relu.update(relu_output)
    scale_relu, zp_relu = observer_relu.get_quantization_params()
    hist_act, (min_val_act, max_val_act) = observer_relu.get_histogram_params(relu_output)
    visualize_histogram_overlay(hist_act, min_val_act, max_val_act)



    print(f"Scale shape: {scale_relu.shape}")
    print(f"Scale: {scale_relu}")
    print(f"Zero Point: {zp_relu}")
    
if __name__ == "__main__":
    main()
