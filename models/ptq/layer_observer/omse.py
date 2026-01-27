# Copyright (c) MEGVII Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn as nn

from .base import BaseObserver
from .utils import lp_loss
from ..bit_type import BitType 


class OmseObserver(BaseObserver):
    def __init__(self, bit_type, module_type, calibration_mode,
                 num_heads=None, head_dim=None):
        # Activation도 이제 지원 가능!
        super().__init__(bit_type, module_type, calibration_mode,
                         num_heads=num_heads, head_dim=head_dim)
        
        self.symmetric = self.bit_type.symmetric
        self.max_val = None
        self.min_val = None
        self.calibration_data = []  # FP32 배치들을 저장
        self.max_batches = 10  # 메모리 제한 (선택사항)
    
    def update(self, v):
        """배치마다 호출 - FP32 저장 + min/max 업데이트"""
        v = self.reshape_tensor(v)
        
        # 1. FP32 데이터 저장 (나중에 OMSE 계산용)
        self.calibration_data.append(v.detach().cpu())  # CPU로 옮겨서 GPU 메모리 절약
        
        # 메모리 제한 (선택사항)
        if len(self.calibration_data) > self.max_batches:
            self.calibration_data.pop(0)  # 오래된 배치 제거
        
        # 2. Min/Max 통계 업데이트 (layer/channel-wise)
        cur_max = v.max(axis=1).values
        cur_min = v.min(axis=1).values
        
        if self.max_val is None:
            self.max_val = cur_max
            self.min_val = cur_min
        else:
            self.max_val = torch.max(cur_max, self.max_val)
            self.min_val = torch.min(cur_min, self.min_val)
        
        # 3. layer_wise면 스칼라로 변환
        if self.calibration_mode == 'layer_wise':
            self.max_val = self.max_val.max()
            self.min_val = self.min_val.min()
    
    def get_quantization_params(self):
        """Grid search with all calibration data"""
        # calibration_data가 비어있는 경우 처리
        if len(self.calibration_data) == 0 or self.max_val is None:
            raise ValueError("No calibration data collected. Call update() first.")

        # 저장된 모든 배치를 합쳐서 OMSE 계산
        all_data = torch.cat(self.calibration_data, dim=1)  # 모든 배치 concat

        # GPU로 옮겨서 grid search
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        all_data = all_data.to(device)

        max_val = self.max_val.to(device)
        min_val = self.min_val.to(device)
        qmax = self.bit_type.upper_bound
        qmin = self.bit_type.lower_bound

        # 최소 scale 값 (0으로 나누기 방지)
        eps = 1e-8

        best_score = 1e+10
        scale = None
        zero_point = None
        search_iterations = 100
        keep_ratios = torch.logspace(0, -4, steps=search_iterations).to(device)

        for keep_ratio in keep_ratios:
            new_max = max_val * keep_ratio
            new_min = min_val * keep_ratio

            if self.symmetric:
                new_max = torch.max(-new_min, new_max)
                new_scale = new_max / (float(qmax - qmin) / 2)
                new_scale = torch.clamp(new_scale, min=eps)  # 최소값 보장
                new_zero_point = torch.zeros_like(new_max, dtype=torch.int64)
            else:
                new_scale = (new_max - new_min) / float(qmax - qmin)
                new_scale = torch.clamp(new_scale, min=eps)  # 최소값 보장
                new_zero_point = qmin - torch.round(new_min / new_scale)
                new_zero_point = new_zero_point.clamp(qmin, qmax)

            # Quantize & Dequantize
            if self.calibration_mode == 'layer_wise':
                inputs_q = ((all_data / new_scale + new_zero_point).round().clamp(
                    qmin, qmax) - new_zero_point) * new_scale
            else:
                # channel_wise
                new_scale_expanded = new_scale.unsqueeze(1)
                new_zero_point_expanded = new_zero_point.unsqueeze(1)
                inputs_q = ((all_data / new_scale_expanded + new_zero_point_expanded).round().clamp(
                    qmin, qmax) - new_zero_point_expanded) * new_scale_expanded

            # L2 loss
            score = lp_loss(all_data, inputs_q, p=1.0, reduction='all')

            # NaN 체크 추가
            if not torch.isnan(score) and not torch.isinf(score) and score < best_score:
                best_score = score
                self.max_val = new_max
                self.min_val = new_min
                scale = new_scale
                zero_point = new_zero_point

        # 메모리 정리
        del all_data
        self.calibration_data.clear()  # 더 이상 필요 없음

        # Fallback: 만약 scale이 여전히 None이면
        if scale is None:
            if self.symmetric:
                scale = torch.clamp(max_val / (float(qmax - qmin) / 2), min=eps)
                zero_point = torch.zeros_like(max_val, dtype=torch.int64)
            else:
                scale = torch.clamp((max_val - min_val) / float(qmax - qmin), min=eps)
                zero_point = qmin - torch.round(min_val / scale)
                zero_point = zero_point.clamp(qmin, qmax)

        return scale, zero_point

    
