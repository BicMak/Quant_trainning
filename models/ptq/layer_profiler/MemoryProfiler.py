import torch    

class MemoryProfiler:
    def __init__(self):
        self.records = {}
    
    def _forward_hook(self, name):
        def hook(module, input, output):
            self.records[name] = {
                'input_size': input[0].numel() * input[0].element_size(),
                'output_size': output.numel() * output.element_size(),
                'weight_size': sum(p.numel() * p.element_size() 
                                   for p in module.parameters()),
            }
            if torch.cuda.is_available():
                self.records[name]['cuda_allocated'] = torch.cuda.memory_allocated()
        return hook
    
    def attach(self, model):
        for name, module in model.named_modules():
            module.register_forward_hook(self._forward_hook(name))
    
    def get_results(self):
        return self.records
