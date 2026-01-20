import time
from contextlib import contextmanager


class TimeProfiler:
    def __init__(self):
        self.records = {}

    @contextmanager
    def measure(self, name):
        start = time.perf_counter()
        yield
        elapsed = time.perf_counter() - start
        if name not in self.records:
            self.records[name] = []
        self.records[name].append(elapsed)

    def get_results(self):
        return {
            name: {
                'mean': sum(times) / len(times),
                'total': sum(times),
                'count': len(times),
                'min': min(times),
                'max': max(times),
            }
            for name, times in self.records.items()
        }

    def reset(self):
        self.records = {}


# Alias for backward compatibility
inferProfiler = TimeProfiler
