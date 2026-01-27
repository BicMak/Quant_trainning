import timm
import torch
import torch.nn as nn

# 1. 모델 생성
model = timm.create_model('vit_base_patch16_224', pretrained=True)

print("="*80)
print("Full ViT Model Structure - All Layers")
print("="*80)

# 전체 모델의 모든 named_modules 출력
for name, module in model.named_modules():
    # 너무 길면 줄여서 보여줌
    module_type = str(type(module)).split("'")[1]
    print(f"{name:<60} | {module_type}")

print("\n" + "="*80)
print("Full ViT Model Structure - Named Children Only")
print("="*80)

# Top-level children만 출력
for name, module in model.named_children():
    module_type = str(type(module)).split("'")[1]
    print(f"{name:<30} | {module_type}")

    # 각 child의 하위 레이어들도 출력
    if name in ['patch_embed', 'blocks', 'norm', 'head']:
        for sub_name, sub_module in module.named_children():
            sub_type = str(type(sub_module)).split("'")[1]
            print(f"  └─ {sub_name:<26} | {sub_type}")

            # blocks의 첫 번째만 더 깊이 들어감
            if name == 'blocks' and sub_name == '0':
                for subsub_name, subsub_module in sub_module.named_children():
                    subsub_type = str(type(subsub_module)).split("'")[1]
                    print(f"      └─ {subsub_name:<22} | {subsub_type}")

                    # attn과 mlp는 더 깊이
                    if subsub_name in ['attn', 'mlp']:
                        for subsubsub_name, subsubsub_module in subsub_module.named_children():
                            subsubsub_type = str(type(subsubsub_module)).split("'")[1]
                            print(f"          └─ {subsubsub_name:<18} | {subsubsub_type}")

print("\n" + "="*80)
print("Model Summary")
print("="*80)
print(f"Total number of blocks: {len(model.blocks)}")
print(f"Patch embed output dim: {model.patch_embed.proj.out_channels}")
print(f"Number of heads: {model.blocks[0].attn.num_heads}")
print(f"Head dim: {model.blocks[0].attn.head_dim}")
print(f"Embed dim: {model.embed_dim}")
