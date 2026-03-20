"""Timing metrics logger."""

import csv
import os
import time
from collections import deque
from contextlib import contextmanager

from app.config import MetricsConfig


class MetricsLogger:
    def __init__(self, config: MetricsConfig | None = None):
        self.config = config or MetricsConfig()
        self.recent: deque[dict] = deque(maxlen=self.config.rolling_window)
        self.current: dict[str, float] = {}

        if self.config.enabled and not os.path.exists(self.config.path):
            with open(self.config.path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "record_audio", "transcribe",
                    "generate_response", "speak", "total",
                ])

    @contextmanager
    def time(self, key: str):
        start = time.perf_counter()
        yield
        self.current[key] = time.perf_counter() - start

    def log_run(self):
        total = sum(self.current.values())
        self.current["total"] = total

        if self.config.enabled:
            with open(self.config.path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                    self.current.get("record_audio", 0),
                    self.current.get("transcribe", 0),
                    self.current.get("generate_response", 0),
                    self.current.get("speak", 0),
                    total,
                ])

        self.recent.append(dict(self.current))
        self.print_summary()
        self.current = {}

    def rolling_average(self) -> dict[str, float]:
        if not self.recent:
            return {}
        keys = ["record_audio", "transcribe", "generate_response", "speak", "total"]
        avg = {k: 0.0 for k in keys}
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
            print(f"Rolling average (last {len(self.recent)} runs):")
            for k, v in avg.items():
                print(f"  {k}: {v:.2f}s")
