import torch
from .StatProfiler import StatProfiler
from .HistProfiler import HistProfiler
from .inferProfiler import TimeProfiler
from .MemoryProfiler import MemoryProfiler

class profiler:
    def __init__(self, name):
        self.stat_data = None
        self.hist_data = None
        self.time_profiler = TimeProfiler()
        self.memory_profiler = MemoryProfiler()
        self.name = name

        # Batch-wise storage for activation profiling
        self.weight_batch_list = []  # List of weight tensors (each can be batched)
        self.quant_batch_list = []   # List of quantized tensors (each can be batched)

    def update_weight(self, weight, quant_weight):
        """
        Update weight/activation tensors.

        Args:
            weight: Original tensor, can be:
                    - Single tensor (weight profiling)
                    - Batched tensor [B, ...] (activation profiling)
            quant_weight: Quantized tensor (same shape as weight)

        Behavior:
            - Stores each tensor in batch list
            - When get_statistic() is called, all batches are concatenated
        """
        # Store the batch tensors (will be concatenated later)
        self.weight_batch_list.append(weight.detach().cpu() if hasattr(weight, 'detach') else weight)
        self.quant_batch_list.append(quant_weight.detach().cpu() if hasattr(quant_weight, 'detach') else quant_weight)


    @staticmethod
    def _check_null(weight, quantweight):
        if (weight is None) or (quantweight is None):
            raise ValueError("Please update weight first using update_weight()")
        else:
            return True

    def get_statistic(self):
        """
        Get statistical analysis of weight and quantized weight.

        Behavior:
            - If batches have been accumulated via update_weight():
              Concatenates all batches along dim=0 and computes statistics on full dataset
            - Otherwise: Returns None

        Returns:
            dict: Statistics (QSNR, MSE, min, max, mean, std, etc.)
                  Computed on all accumulated data
        """
        if len(self.weight_batch_list) > 0 and len(self.quant_batch_list) > 0:
            # Concatenate all batches into single tensor
            # Each batch: [B, ...], after cat: [total_samples, ...]
            weight_cat = torch.cat(self.weight_batch_list, dim=0)
            quant_cat = torch.cat(self.quant_batch_list, dim=0)

            # Compute statistics on full dataset
            self.stat_data = StatProfiler.compute(weight_cat, quant_cat)
        else:
            # No data accumulated
            return None

        return self.stat_data

    def get_hist(self):
        """
        Get histogram data of weight and quantized weight.

        Behavior:
            - Concatenates all accumulated batches and computes histogram on full dataset

        Returns:
            dict: Histogram data
        """
        if len(self.weight_batch_list) > 0 and len(self.quant_batch_list) > 0:
            # Concatenate all batches into single tensor
            weight_cat = torch.cat(self.weight_batch_list, dim=0)
            quant_cat = torch.cat(self.quant_batch_list, dim=0)

            # Compute histogram on full dataset
            self.hist_data = HistProfiler.compute(weight_cat, quant_cat)
        else:
            # No data accumulated
            return None

        return self.hist_data

    def measure_time(self, func=None, *args, **kwargs):
        """
        Measure execution time using context manager.

        Usage:
            with profiler.measure_time():
                # code to measure

        Or pass a function:
            result = profiler.measure_time(func, *args, **kwargs)
        """
        if func is not None:
            with self.time_profiler.measure(self.name):
                return func(*args, **kwargs)
        else:
            return self.time_profiler.measure(self.name)

    def get_time_record(self):
        """Get timing profiler results"""
        return self.time_profiler.get_results()

    def attach_memory_profiler(self, model):
        """Attach memory profiler hooks to model"""
        self.memory_profiler.attach(model)

    def get_memory_record(self):
        """Get memory profiler results"""
        return self.memory_profiler.get_results()

    def reset_time_profiler(self):
        """Reset time profiler records"""
        self.time_profiler.reset()

    def reset_memory_profiler(self):
        """Reset memory profiler records"""
        self.memory_profiler = MemoryProfiler()

    def clear_batches(self):
        """Clear accumulated batch data"""
        self.weight_batch_list = []
        self.quant_batch_list = []

    def get_batch_count(self):
        """Get number of accumulated batches"""
        return len(self.weight_batch_list)


