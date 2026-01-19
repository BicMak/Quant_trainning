# Copyright (c) MEGVII Inc. and its affiliates. All Rights Reserved.
import torch


class BaseObserver:
    def __init__(self, 
                 bit_type, 
                 module_type, 
                 calibration_mode):
        
        self.module_type = module_type
        self.bit_type = bit_type
        self.calibration_mode = calibration_mode

        self.eps = torch.finfo(torch.float32).eps
        self.v_channel = None

    def reshape_tensor(self, v):
        if not isinstance(v, torch.Tensor):
            raise TypeError(
                f"Expected torch.Tensor but got {type(v).__name__}. "
                f"Please convert input to tensor before calling this method."
            )       
        v = v.detach()

        #case1 : conv_weight
        if self.module_type  == 'conv_weight':
            #conv weight shape : (out_channels, in_channels, kH, kW)
            v = v.reshape(v.shape[0], -1)
            # output channel number
            self.v_channel = v.shape[0] # output channel 수

        #case2 : linear_weight
        elif self.module_type == 'linear_weight':
            #linear weight shape : (out_features, in_features)
            v = v.reshape(v.shape[0], -1)
            self.v_channel = v.shape[0] # output channel 수
        
        #case3 : activation
        elif self.module_type == 'activation':
            # CNN model activation shape : (N, C, H, W)
            if len(v.shape) == 4:
                v = v.permute(0, 2, 3, 1)
                v = v.reshape(-1, v.shape[-1])
                v = v.transpose(0, 1)
                self.v_channel = v.shape[0]  # channel 수
            
            # linear : (batch, seq_len, embedding_dim)
            else:
                v = v.reshape(-1, v.shape[-1])
                v = v.transpose(0, 1)
                self.v_channel = v.shape[0]  # channel 수

        else:
            raise NotImplementedError
        
        return v

    def update(self, v):
        # update self.max_val and self.min_val
        raise NotImplementedError

    def get_quantization_params(self, *args, **kwargs):
        raise NotImplementedError
    


