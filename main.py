import timm
import torch

# ViT 모델 불러오기
model = timm.create_model('vit_base_patch16_224', pretrained=True)
model.eval()

# 테스트 입력
x = torch.randn(1, 3, 224, 224)

# Forward
with torch.no_grad():
    out = model(x)

print(f"Model: {model.__class__.__name__}")
print(f"Input shape: {x.shape}")
print(f"Output shape: {out.shape}")

# 레이어 정보 출력
print("\n=== Layer Info ===")
for name, module in model.named_modules():
    print(f"{name}: {module.__class__.__name__}")
