import torch
import torch.nn.functional as F

class StatProfiler:
    @staticmethod
    def compute(original: torch.Tensor, quantized: torch.Tensor) -> dict:
        return {
            # Quantized statistics
            'min': quantized.min().item(),
            'max': quantized.max().item(),
            'mean': quantized.mean().item(),
            'std': quantized.std().item(),
            'outlier_ratio': StatProfiler._outlier_ratio(quantized),


            # Original (FP32) statistics
            'original_min': original.min().item(),
            'original_max': original.max().item(),
            'original_mean': original.mean().item(),
            'original_std': original.std().item(),
            'original_outlier_ratio': StatProfiler._outlier_ratio(original),

            # Comparison metrics
            'mse': F.mse_loss(original, quantized).item(),
            'cosine_sim': F.cosine_similarity(
                original.flatten().unsqueeze(0),
                quantized.flatten().unsqueeze(0)
            ).item(),
            'qsnr': StatProfiler._qsnr(original, quantized),
        }
    
    @staticmethod
    def _outlier_ratio(x, threshold=3.0):
        mean, std = x.mean(), x.std()
        outliers = ((x - mean).abs() > threshold * std).sum()
        return (outliers / x.numel()).item()
    
    @staticmethod
    def _qsnr(original, quantized):
        # QSNR = 10 * log10(signal_power / noise_power)
        noise = original - quantized
        signal_power = (original ** 2).mean()
        noise_power = (noise ** 2).mean() + 1e-10
        return (10 * torch.log10(signal_power / noise_power)).item()
