import torch

class HistProfiler:
    @staticmethod
    def compute(original: torch.Tensor, quantized: torch.Tensor,
                bins=256) -> dict:
        # 공통 범위 설정 (두 텐서의 min/max 중 더 넓은 범위)
        min_val = min(original.min().item(), quantized.min().item())
        max_val = max(original.max().item(), quantized.max().item())

        # Histogram 계산
        orig_hist = torch.histc(original.float(), bins=bins, min=min_val, max=max_val)
        quant_hist = torch.histc(quantized.float(), bins=bins, min=min_val, max=max_val)

        # 정규화 (확률 분포로 변환)
        orig_prob = (orig_hist + 1e-10) / (orig_hist.sum() + bins * 1e-10)
        quant_prob = (quant_hist + 1e-10) / (quant_hist.sum() + bins * 1e-10)

        # KL Divergence: KL(P||Q) = sum(P * log(P/Q))
        kl_div = (orig_prob * torch.log(orig_prob / quant_prob)).sum().item()

        # Jensen-Shannon Divergence (symmetric version of KL)
        m = (orig_prob + quant_prob) / 2
        js_div = 0.5 * (orig_prob * torch.log(orig_prob / m)).sum().item() + \
                 0.5 * (quant_prob * torch.log(quant_prob / m)).sum().item()

        return {
            'original_hist': orig_hist.cpu(),
            'quantized_hist': quant_hist.cpu(),
            'original_range': (original.min().item(), original.max().item()),
            'quantized_range': (quantized.min().item(), quantized.max().item()),
            'kl_divergence': kl_div,
            'js_divergence': js_div,
        }
    

