# Copyright (c) MEGVII Inc. and its affiliates. All Rights Reserved.
import torch


class BaseObserver:
    def __init__(self,
                 bit_type,
                 module_type,
                 calibration_mode,
                 num_heads=None,
                 head_dim=None):

        self.module_type = module_type
        self.bit_type = bit_type
        self.calibration_mode = calibration_mode

        # Head-wise 설정 (ViT attention용)
        self.num_heads = num_heads
        self.head_dim = head_dim

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
                # head_wise: (B, N, C) → (num_heads, B*N*head_dim)
                if self.calibration_mode == 'head_wise' and self.num_heads is not None:
                    # v: (B, N, C) where C = num_heads * head_dim
                    B, N, C = v.shape
                    # (B, N, num_heads, head_dim) → (num_heads, B*N*head_dim)
                    v = v.reshape(B, N, self.num_heads, self.head_dim)
                    v = v.permute(2, 0, 1, 3)  # (num_heads, B, N, head_dim)
                    v = v.reshape(self.num_heads, -1)  # (num_heads, B*N*head_dim)
                    self.v_channel = self.num_heads  # head 수
                else:
                    # 기존 방식: channel_wise or layer_wise
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
    


