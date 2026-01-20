import torch 

class HistProfiler:
    @staticmethod
    def compute(original: torch.Tensor, quantized: torch.Tensor, 
                bins=256) -> dict:
        return {
            'original_hist': torch.histc(original.float(), bins=bins).cpu(),
            'quantized_hist': torch.histc(quantized.float(), bins=bins).cpu(),
            'original_range': (original.min().item(), original.max().item()),
            'quantized_range': (quantized.min().item(), quantized.max().item()),
        }
    

