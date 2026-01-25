import timm
import torch
import torch.nn as nn

# 1. 모델 생성 (qk_norm=False, layerscale=False 설정)
model = timm.create_model('vit_base_patch16_224', pretrained=True)

# 2. 첫 번째 블록(Block 0) 가져오기
block0 = model.blocks[0]

print(f"=== [Structure of model.blocks[0]] ===")
# Block의 주요 구성 요소 (norm1, attn, ls1, norm2, mlp, ls2)
for name, module in block0.named_children():
    print(f"Layer: {name:<10} | Type: {type(module)}")

print("\n" + "="*50)
print("=== [Deep Dive: model.blocks[0].attn] ===")
# Attention 내부의 세부 구성 요소 (qkv, q_norm, k_norm, proj 등)
attn_module = block0.attn
for name, module in attn_module.named_children():
    print(f"Sub-Layer: {name:<10} | Type: {type(module)}")

print("\n" + "="*50)
print("=== [Deep Dive: model.blocks[0].mlp] ===")
# MLP 내부의 세부 구성 요소 (fc1, act, fc2 등)
mlp_module = block0.mlp
for name, module in mlp_module.named_children():
    print(f"Sub-Layer: {name:<10} | Type: {type(module)}")