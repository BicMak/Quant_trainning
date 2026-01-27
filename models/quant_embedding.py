import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Union
from pathlib import Path

from .ptq.quant_act import QAct
from quant_config import QuantConfig
from utils.config_loader import load_config_from_yaml


class QEmbedding(nn.Module):
    """
    Quantized Patch Embedding for ViT models.

    Quantizes the output of patch embedding (after Conv2d projection, cls_token, pos_embed).

    Forward flow:
        Input [B, 3, 224, 224]
        -> Conv2d proj [B, 768, 14, 14]
        -> Flatten [B, 768, 196]
        -> Transpose [B, 196, 768]
        -> CLS token concat [B, 197, 768]
        -> Positional embedding add [B, 197, 768]
        -> QAct (output quantization) [B, 197, 768]
    """

    def __init__(self,
                 model,  # Full ViT model (to access cls_token, pos_embed)
                 config_path: Union[str, Path]):
        super().__init__()

        self.original_model = model
        patch_embed = model.patch_embed

        # Config 로드: YAML 파일에서 각 레이어별 config를 로드
        configs = load_config_from_yaml(config_path)

        # Enable profiling은 config에서 가져옴
        self.enable_profiling = configs.get('output', configs.get('activation')).enable_profiler

        # === Patch Embedding Projection (Conv2d) ===
        # timm의 patch_embed.proj는 Conv2d (in_channels=3, out_channels=768, kernel_size=16, stride=16)
        # 이미 pretrained weight가 있으므로 그대로 사용
        self.proj = patch_embed.proj

        # norm은 Identity이므로 사용 안함
        self.norm = patch_embed.norm

        # === CLS token and Positional Embedding ===
        # These are learnable parameters from the pretrained model
        self.cls_token = model.cls_token
        self.pos_embed = model.pos_embed

        # === Output Quantization ===
        # pos_embed 후 출력을 양자화
        self.embed_output = QAct(
            quant_config=configs.get('activation', configs['output']),
            act_module=None,
            layer_name='embed_output'
        )

        # Embedding parameters
        self.img_size = patch_embed.img_size
        self.patch_size = patch_embed.patch_size
        self.num_patches = patch_embed.num_patches
        self.embed_dim = patch_embed.proj.out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with quantized output.

        Args:
            x: Input tensor [B, 3, H, W]

        Returns:
            Quantized patch embeddings with cls_token and pos_embed [B, num_patches+1, embed_dim]
        """
        # 1. Input: [Batch, Channels, Height, Width]
        # 예: [B, 3, 224, 224]

        # 2. Projection (Conv2d 연산)
        # patch_embed.proj를 통과하면 패치 크기만큼 줄어들고 채널은 Embed Dim으로 확장됨
        x = self.proj(x)
        # 결과: [B, 768, 14, 14] (패치 16, 임베딩 768 기준)

        # 3. Flatten (2D 특징 맵을 1차원 시퀀스로 변환)
        # 마지막 두 차원(H, W)을 하나로 합칩니다.
        x = x.flatten(2)
        # 결과: [B, 768, 196] (14*14 = 196개 토큰)

        # 4. Transpose (Transformer 입력을 위해 차원 교환)
        # (Batch, Dim, Seq) -> (Batch, Seq, Dim) 순서로 바꿉니다.
        x = x.transpose(1, 2)
        # 결과: [B, 196, 768]

        # 5. CLS token 추가
        # cls_token을 배치 크기만큼 확장하고 앞에 concat
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        # 결과: [B, 197, 768] (1 cls_token + 196 patches)

        # 6. Positional Embedding 추가
        x = x + self.pos_embed
        # 결과: [B, 197, 768]


        # 7. Output Quantization
        x = self.embed_output.forward(x)
        # 결과: [B, 197, 768] (quantized)

        return x

    def calibration(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calibration pass to collect statistics.

        Args:
            x: Input tensor [B, 3, H, W]

        Returns:
            Patch embeddings with cls_token and pos_embed [B, num_patches+1, embed_dim]
        """
        # 1. Input: [Batch, Channels, Height, Width]
        # 예: [B, 3, 224, 224]

        # 2. Projection (Conv2d 연산)
        x = self.proj(x)
        # 결과: [B, 768, 14, 14]

        # 3. Flatten
        x = x.flatten(2)
        # 결과: [B, 768, 196]

        # 4. Transpose
        x = x.transpose(1, 2)
        # 결과: [B, 196, 768]

        # 5. CLS token 추가
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        # 결과: [B, 197, 768]

        # 6. Positional Embedding 추가
        x = x + self.pos_embed
        # 결과: [B, 197, 768]


        # 7. Output Calibration
        x = self.embed_output.calibration(x)

        return x

    def compute_quant_params(self, calib_loader=None):
        """Compute quantization parameters for output layer."""
        # Output QAct layer: compute_quant_params() 사용
        self.embed_output.compute_quant_params()

    def set_mode(self, mode: str):
        """Set quantization mode ('fp32' or 'quantized')."""
        if hasattr(self.embed_output, 'mode'):
            self.embed_output.mode = mode

    def get_quantized_layers(self) -> Dict[str, nn.Module]:
        """Returns all quantized layers."""
        return {
            'embed_output': self.embed_output,
        }

    def get_profiler(self) -> Dict:
        """Get profiling results from all layers"""
        if not self.enable_profiling:
            return {}

        results = {}
        results['embed_output'] = self.embed_output.get_profiler()
        return results
