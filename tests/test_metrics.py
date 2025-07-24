import csv
import builtins
import time
from metrics import MetricsLogger


def test_time_records_duration(monkeypatch):
    logger = MetricsLogger(enabled=False)
    counter = iter([1.0, 2.0])
    monkeypatch.setattr(time, "perf_counter", lambda: next(counter))
    with logger.time("record_audio"):
        pass
    assert logger.current["record_audio"] == 1.0


def test_log_run_writes_csv(tmp_path, monkeypatch):
    path = tmp_path / "metrics.csv"
    logger = MetricsLogger(path=path, enabled=True)
    logger.current = {
        "record_audio": 1.0,
        "transcribe": 2.0,
        "generate_response": 3.0,
        "speak": 4.0,
    }
    monkeypatch.setattr(time, "strftime", lambda *_: "2025-01-01 00:00:00")
    logger.log_run()
    with open(path) as f:
        rows = list(csv.reader(f))
    assert rows[0] == [
        "timestamp",
        "record_audio",
        "transcribe",
        "generate_response",
        "speak",
        "total",
    ]
    assert len(rows) == 2
    assert float(rows[1][1]) == 1.0
    assert float(rows[1][2]) == 2.0
    assert float(rows[1][3]) == 3.0
    assert float(rows[1][4]) == 4.0
    assert float(rows[1][5]) == 10.0


def test_rolling_average(monkeypatch):
    logger = MetricsLogger(enabled=False, rolling=5)
    monkeypatch.setattr(time, "strftime", lambda *_: "2025-01-01 00:00:00")
    for i in range(3):
        logger.current = {
            "record_audio": i + 1,
            "transcribe": i + 1,
            "generate_response": i + 1,
            "speak": i + 1,
        }
        logger.log_run()
    avg = logger.rolling_average()
    assert avg["record_audio"] == 2.0
    assert avg["total"] == 8.0

