import torch
from .StatProfiler import StatProfiler
from .HistProfiler import HistProfiler
from .inferProfiler import TimeProfiler
from .MemoryProfiler import MemoryProfiler

class profiler:
    def __init__(self, name):
        self.weight = None
        self.quant_weight = None
        self.stat_data = None
        self.hist_data = None
        self.time_profiler = TimeProfiler()
        self.memory_profiler = MemoryProfiler()
        self.name = name

    def update_weight(self, weight, quant_weight):
        self.weight = weight
        self.quant_weight = quant_weight

    @staticmethod
    def _check_null(weight, quantweight):
        if (weight is None) or (quantweight is None):
            raise ValueError("Please update weight first using update_weight()")
        else:
            return True

    def get_statistic(self):
        """Get statistical analysis of weight and quantized weight"""
        profiler._check_null(self.weight, self.quant_weight)
        self.stat_data = StatProfiler.compute(self.weight, self.quant_weight)
        return self.stat_data

    def get_hist(self):
        """Get histogram data of weight and quantized weight"""
        profiler._check_null(self.weight, self.quant_weight)
        self.hist_data = HistProfiler.compute(self.weight, self.quant_weight)
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


