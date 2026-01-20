import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from .ptq.quant_linear import QuantLinear
from .ptq.quant_intSoft import QuantIntSoft
from .ptq.quant_act import QAct
from .ptq.quant_layernorm import QLayerNorm
from quant_config import QuantConfig, BitTypeConfig

from torch.nn.modules.container import Sequential


class QuantTimmVitBlock(nn.Module):
    """
    Quantized ViT Block for timm models.
    timm의 vit_base_patch16_224 등에서 사용.
    """

    def __init__(self,
                 block,  # timm.models.vision_transformer.Block
                 observer_config: QuantConfig):
        super().__init__()

        self.original_block = block

        # === Attention 부분 ===
        # timm은 qkv가 하나로 합쳐져 있음 (in_features → 3 * out_features)
        self.attn_qkv = QuantLinear(
            quant_args={},
            input_module=block.attn.qkv,
            observer_config=observer_config
        )
        self.attn_proj = QuantLinear(
            quant_args={},
            input_module=block.attn.proj,
            observer_config=observer_config
        )

        # === MLP 부분 ===
        self.mlp_fc1 = QuantLinear(
            quant_args={},
            input_module=block.mlp.fc1,
            observer_config=observer_config
        )
        self.mlp_fc2 = QuantLinear(
            quant_args={},
            input_module=block.mlp.fc2,
            observer_config=observer_config
        )

        # === LayerNorm ===
        self.norm1 = QLayerNorm(
            quant_args={},
            input_module=block.norm1,
            observer_config=observer_config
        )
        self.norm2 = QLayerNorm(
            quant_args={},
            input_module=block.norm2,
            observer_config=observer_config
        )

        # === 그대로 쓰는 것들 ===
        self.attn_drop = block.attn.attn_drop
        self.proj_drop = block.attn.proj_drop
        self.mlp_act = block.mlp.act
        self.drop_path1 = block.drop_path1
        self.drop_path2 = block.drop_path2

        # Attention 파라미터
        self.num_heads = block.attn.num_heads
        self.head_dim = block.attn.head_dim
        self.scale = block.attn.scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with quantized layers."""
        # === Attention block ===
        B, N, C = x.shape

        # norm1 → qkv
        x_norm = self.norm1(x)
        qkv = self.attn_qkv(x_norm)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # (B, num_heads, N, head_dim)

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Combine heads
        x_attn = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x_attn = self.attn_proj(x_attn)
        x_attn = self.proj_drop(x_attn)

        # Residual 1
        x = x + self.drop_path1(x_attn)

        # === MLP block ===
        x_norm = self.norm2(x)
        x_mlp = self.mlp_fc1(x_norm)
        x_mlp = self.mlp_act(x_mlp)
        x_mlp = self.mlp_fc2(x_mlp)

        # Residual 2
        x = x + self.drop_path2(x_mlp)

        return x

    def get_quantized_layers(self) -> Dict[str, nn.Module]:
        """Returns all quantized layers."""
        return {
            'attn_qkv': self.attn_qkv,
            'attn_proj': self.attn_proj,
            'mlp_fc1': self.mlp_fc1,
            'mlp_fc2': self.mlp_fc2,
            'norm1': self.norm1,
            'norm2': self.norm2,
        }

    def calibration(self, x: torch.Tensor) -> torch.Tensor:
        """Calibration pass to collect statistics."""
        B, N, C = x.shape

        # norm1 → qkv
        x_norm = self.norm1.calibration(x)
        qkv = self.attn_qkv.calibration(x_norm)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # Attention (FP32)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_attn = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x_attn = self.attn_proj.calibration(x_attn)
        x_attn = self.proj_drop(x_attn)

        x = x + self.drop_path1(x_attn)

        # MLP
        x_norm = self.norm2.calibration(x)
        x_mlp = self.mlp_fc1.calibration(x_norm)
        x_mlp = self.mlp_act(x_mlp)
        x_mlp = self.mlp_fc2.calibration(x_mlp)

        x = x + self.drop_path2(x_mlp)

        return x

    def compute_quant_params(self):
        """Compute quantization parameters for all layers."""
        for layer in self.get_quantized_layers().values():
            if hasattr(layer, 'compute_quant_params'):
                layer.compute_quant_params()

    def set_mode(self, mode: str):
        """Set quantization mode ('fp32' or 'quantized')."""
        for layer in self.get_quantized_layers().values():
            if hasattr(layer, 'mode'):
                layer.mode = mode




class QuantVitBlock(nn.Module):
    """
    Implementation of Quantized VitPose Backbone Block.
    Wraps VitPoseBackboneLayer for layer-wise quantization with AdaRound.
    """
    
    def __init__(self,
                 vit_block: Sequential,
                 weight_quant_params: Dict = None,
                 act_quant_params: Dict = None):
        super().__init__()
        
        self.vit_block = vit_block
        self.weight_quant_params = weight_quant_params or {}
        self.act_quant_params = act_quant_params or {}
        
        # Quantize Linear layers in attention
        # query, key, value
        self.attention_query = QuantLinear(
            vit_block.attention.attention.query, 
            self.weight_quant_params  # ← Dict 전달
        )
        self.attention_key = QuantLinear(
            vit_block.attention.attention.key,
            self.weight_quant_params
        )
        self.attention_value = QuantLinear(
            vit_block.attention.attention.value,
            self.weight_quant_params
        )

        self.attention_output_dense = QuantLinear(
            vit_block.attention.output.dense,
            self.weight_quant_params
        )

        self.mlp_fc1 = QuantLinear(vit_block.mlp.fc1, self.weight_quant_params)
        self.mlp_fc2 = QuantLinear(vit_block.mlp.fc2, self.weight_quant_params)
                
        # Store original block references
        self.layernorm_before = vit_block.layernorm_before
        self.layernorm_after = vit_block.layernorm_after
        self.attention_dropout = vit_block.attention.output.dropout
        self.mlp_activation = vit_block.mlp.activation
        
        # Store original non-quantized modules for reference
        self.original_block = vit_block
    
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with quantized attention and MLP.
        """
        # Pre-normalization
        x_normalized = self.layernorm_before(x)
        
        # Self-attention block
        # Replace original linear layers with quantized versions
        batch_size, seq_len, hidden_dim = x_normalized.shape
        
        # Quantized attention computation
        query = self.attention_query(x_normalized)
        key = self.attention_key(x_normalized)
        value = self.attention_value(x_normalized)
        
        # Multi-head attention
        num_heads = self.original_block.attention.attention.num_attention_heads
        attention_head_size = hidden_dim // num_heads
        
        query = query.reshape(batch_size, seq_len, num_heads, attention_head_size).transpose(1, 2)
        key = key.reshape(batch_size, seq_len, num_heads, attention_head_size).transpose(1, 2)
        value = value.reshape(batch_size, seq_len, num_heads, attention_head_size).transpose(1, 2)
        
        # Attention scores
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / (attention_head_size ** 0.5)
        attention_probs = torch.softmax(attention_scores, dim=-1)
        attention_probs = self.attention_dropout(attention_probs)
        
        # Context layer
        context = torch.matmul(attention_probs, value)
        context = context.transpose(1, 2).contiguous()
        context = context.reshape(batch_size, seq_len, hidden_dim)
        
        # Quantized attention output
        attention_out = self.attention_output_dense(context)
        attention_out = self.attention_dropout(attention_out)
        
        # Residual + activation
        attention_out = attention_out + x
        
        # Post-normalization before MLP
        mlp_input = self.layernorm_after(attention_out)
        
        # Quantized MLP
        mlp_out = self.mlp_fc1(mlp_input)
        mlp_out = self.mlp_activation(mlp_out)
        mlp_out = self.mlp_fc2(mlp_out)
        
        # Final residual
        out = mlp_out + attention_out
        
        return out
    
    def get_quantized_layers(self) -> Dict[str, nn.Linear]:
        """
        Returns all quantizable Linear layers in this block.
        Useful for AdaRound parameter collection.
        """
        return {
            'attention_query': self.attention_query,
            'attention_key': self.attention_key,
            'attention_value': self.attention_value,
            'attention_output_dense': self.attention_output_dense,
            'mlp_fc1': self.mlp_fc1,
            'mlp_fc2': self.mlp_fc2,
        }
    
    def enable_adaround(self):
        """
        Enable AdaRound mode for quantized layers.
        """
        for name, layer in self.get_quantized_layers().items():
            if hasattr(layer, 'enable_adaround'):
                layer.enable_adaround()
    
    def disable_adaround(self):
        """
        Disable AdaRound mode and apply final quantization.
        """
        for name, layer in self.get_quantized_layers().items():
            if hasattr(layer, 'disable_adaround'):
                layer.disable_adaround()

    def fix_quant_blocks(self):
        """
        Enable AdaRound mode for quantized layers.
        """
        for name, layer in self.get_quantized_layers().items():
            if hasattr(layer, 'fix_quant_layer'):
                layer.fix_quant_layer()


class QuantVitPoseHead(nn.Module):
    """
    Implementation of Quantized VitPose conv layer
    Wraps VitPoseBackboneLayer for layer-wise quantization with AdaRound.
    """
    
    def __init__(self,
                 vit_block: VitPoseSimpleDecoder,
                 weight_quant_params: Dict = None,
                 act_quant_params: Dict = None):
        super().__init__()
        
        self.vit_block = vit_block
        self.weight_quant_params = weight_quant_params or {}
        self.act_quant_params = act_quant_params or {}
    
        # Head components
        self.activation = vit_block.activation        # ReLU
        self.upsampling = vit_block.upsampling        # Upsample
        self.conv = QuantLayer(vit_block.conv, self.weight_quant_params)  # Conv2d quantized
        
        # Store original non-quantized modules for reference
        self.original_block = vit_block
    
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with quantized attention and MLP.
        """
        #! x shape = torch.Size([32, 192, 768])

        batch_size, _, hidden_dim = x.shape
        h, w = 16, 12  # *기존 크기가 고정되어있다고 가정

        # Quantized MLP
        x = x.reshape(batch_size, h, w, hidden_dim).permute(0, 3, 1, 2)
        x = self.activation(x)
        x = self.upsampling(x)
        x = self.conv(x)
        return x
    
    def get_quantized_layers(self) -> Dict[str, nn.Linear]:
        """
        Returns all quantizable Linear layers in this block.
        Useful for AdaRound parameter collection.
        """
        return {
            'conv': self.conv,
        }
    
    def enable_adaround(self):
        """
        Enable AdaRound mode for quantized layers.
        """
        for name, layer in self.get_quantized_layers().items():
            if hasattr(layer, 'enable_adaround'):
                layer.enable_adaround()
    
    def disable_adaround(self):
        """
        Disable AdaRound mode and apply final quantization.
        """
        for name, layer in self.get_quantized_layers().items():
            if hasattr(layer, 'disable_adaround'):
                layer.disable_adaround()

    def fix_quant_blocks(self):
        """
        Enable AdaRound mode for quantized layers.
        """
        for name, layer in self.get_quantized_layers().items():
            if hasattr(layer, 'fix_quant_layer'):
                layer.fix_quant_layer()

class QuantVitPoseBackbone(nn.Module):
    """
    Wrapper for entire VitPose backbone with block-wise quantization.
    """
    
    def __init__(self,
                 vitpose_backbone,
                 vitpose_head,
                 weight_quant_params: Dict = None,
                 act_quant_params: Dict = None):
        super().__init__()
        
        self.patch_embeddings = vitpose_backbone.embeddings.patch_embeddings
        self.embeddings_dropout = vitpose_backbone.embeddings.dropout
        self.layernorm = vitpose_backbone.layernorm

        #Vit head
        self.decoder_block =  QuantVitPoseHead(
            vitpose_head,
            weight_quant_params,
            act_quant_params
        )
        
        # Quantize all encoder blocks
        self.encoder_blocks = nn.ModuleList([
            QuantVitPoseBlock(
                vitpose_backbone.encoder.layer[i],
                weight_quant_params,
                act_quant_params
            )
            for i in range(len(vitpose_backbone.encoder.layer))
        ])


    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through quantized backbone.
        """
        # Patch embedding
        B, C, H, W = x.shape
        x = self.patch_embeddings(x)
        x = self.embeddings_dropout(x)
        
        # Pass through quantized encoder blocks
        for block in self.encoder_blocks:
            x = block(x)
        
        # Final layer norm
        x = self.layernorm(x)
        
        x = self.decoder_block(x)
        
        return x
    
    def get_all_quantized_layers(self) -> Dict[int, Dict[str, nn.Linear]]:
        """
        Returns all quantized layers organized by block index.
        """
        all_layers = {}
        for i, block in enumerate(self.encoder_blocks):
            all_layers[i] = block.get_quantized_layers()
        idx = len(all_layers)
        all_layers[idx] = self.decoder_block.get_quantized_layers()

        return all_layers
    
    def enable_adaround_all_blocks(self):
        """
        Enable AdaRound for all blocks.
        """
        for block in self.encoder_blocks:
            block.enable_adaround()

        self.decoder_block.enable_adaround()

    
    def enable_adaround_single_blocks(self,idx):
        """
        Enable AdaRound for all blocks.
        """
        num_encoder = len(self.encoder_blocks)
        
        # Encoder blocks
        for i in range(num_encoder):
            if i == idx:
                self.encoder_blocks[i].enable_adaround()
            else:
                self.encoder_blocks[i].disable_adaround()
        
        # Head conv
        if idx == num_encoder:  # idx가 encoder 개수와 같으면 head 활성화
            self.decoder_block.enable_adaround()
        else:
            self.decoder_block.disable_adaround()
            

    def disable_adaround_all_blocks(self):
        """
        Disable AdaRound for all blocks.
        """
        for block in self.encoder_blocks:
            block.disable_adaround()
        self.decoder_block.disable_adaround()

    def fix_quant_blocks(self,idx):
        """
        Disable AdaRound for all blocks.
        """
        num_encoder = len(self.encoder_blocks)
        
        # Encoder blocks
        for i in range(num_encoder):
            if i == idx:
                self.encoder_blocks[i].fix_quant_blocks()
            else:
                continue