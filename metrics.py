import csv
import os
import time
from collections import deque
from contextlib import contextmanager

class MetricsLogger:
    def __init__(self, path="metrics.csv", enabled=True, rolling=5):
        self.path = path
        self.enabled = enabled
        self.rolling = rolling
        self.recent = deque(maxlen=rolling)
        if self.enabled and not os.path.exists(self.path):
            with open(self.path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp",
                    "record_audio",
                    "transcribe",
                    "generate_response",
                    "speak",
                    "total",
                ])
        self.current = {}

    @contextmanager
    def time(self, key: str):
        start = time.perf_counter()
        yield
        duration = time.perf_counter() - start
        self.current[key] = duration

    def log_run(self):
        total = sum(v for v in self.current.values())
        self.current["total"] = total
        if self.enabled:
            with open(self.path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        time.strftime("%Y-%m-%d %H:%M:%S"),
                        self.current.get("record_audio", 0),
                        self.current.get("transcribe", 0),
                        self.current.get("generate_response", 0),
                        self.current.get("speak", 0),
                        total,
                    ]
                )
        self.recent.append(dict(self.current))
        self.print_summary()
        self.current = {}

    def rolling_average(self):
        if not self.recent:
            return {}
        keys = ["record_audio", "transcribe", "generate_response", "speak", "total"]
        avg = {k: 0 for k in keys}
        for entry in self.recent:
            for k in keys:
                avg[k] += entry.get(k, 0)
        for k in keys:
            avg[k] /= len(self.recent)
        return avg

    def print_summary(self):
        print("--- Timing Summary ---")
        for k, v in self.current.items():
            if k == "total":
                continue
            print(f"{k}: {v:.2f}s")
        print(f"total: {self.current.get('total', 0):.2f}s")
        if len(self.recent) > 1:
            avg = self.rolling_average()
            print("Rolling average (last {} runs):".format(len(self.recent)))
            for k, v in avg.items():
                print(f"{k}: {v:.2f}s")
