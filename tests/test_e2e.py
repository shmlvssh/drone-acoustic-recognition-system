import json
import os
import random
import sys
import tracemalloc
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import main  # noqa: E402


def _real_tflite() -> bool:
    mod = sys.modules.get("tflite_runtime.interpreter")
    return mod is not None and not getattr(mod.Interpreter, "_is_stub", False)


def _models_exist() -> bool:
    c = main.Config()
    return os.path.exists(c.model_1d_path) and os.path.exists(c.model_2d_path)


pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not _real_tflite(), reason="tflite_runtime недоступен"),
    pytest.mark.skipif(not _models_exist(), reason="tflite модели не найдены"),
]


MIN_ACCURACY = 0.70
MIN_F1 = 0.65
MAX_PREPROCESS_MS = 100.0
MAX_1D_MS = 200.0
MAX_2D_MS = 500.0
MAX_E2E_MS = 800.0
P95_MULTIPLIER = 3.0
MAX_PREPROCESS_MEMORY_KB = 200
MAX_DETECTOR_MEMORY_MB = 1
SAMPLE_PER_CLASS = 40
WARMUP_CHUNKS = 5
PERF_CHUNKS = 50
PREPROCESS_ITERS = 1000
DETECTOR_ITERS = 500
SEED = 42


def _sample_files(dir_path: Path, n: int, seed: int):
    files = sorted(str(p) for p in dir_path.glob("*.wav"))
    if not files:
        return []
    rng = random.Random(seed)
    if len(files) <= n:
        return files
    return rng.sample(files, n)


def _load_first_chunk(path: str, chunk_samples: int) -> np.ndarray:
    with sf.SoundFile(path) as f:
        audio = f.read(chunk_samples, dtype="float32")
    return main._normalize_and_fit(audio, chunk_samples)


def test_accuracy_on_random_sample(detector, cfg, yes_dir, no_dir):
    pos = _sample_files(yes_dir, SAMPLE_PER_CLASS, SEED)
    neg = _sample_files(no_dir, SAMPLE_PER_CLASS, SEED + 1)
    if not pos or not neg:
        pytest.skip("недостаточно данных в yes_drone/no_drone")

    tp = fp = tn = fn = 0
    for p in pos:
        audio = _load_first_chunk(p, cfg.chunk_samples)
        pred = detector.predict(audio)["prediction"]
        if pred == 1:
            tp += 1
        else:
            fn += 1
    for p in neg:
        audio = _load_first_chunk(p, cfg.chunk_samples)
        pred = detector.predict(audio)["prediction"]
        if pred == 1:
            fp += 1
        else:
            tn += 1

    total = tp + fp + tn + fn
    acc = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    assert acc >= MIN_ACCURACY, f"Accuracy {acc:.3f} < {MIN_ACCURACY}"
    assert f1 >= MIN_F1, f"F1 {f1:.3f} < {MIN_F1}"


def test_performance_mean(detector, cfg, yes_dir):
    files = _sample_files(yes_dir, WARMUP_CHUNKS + PERF_CHUNKS, SEED)
    if len(files) < WARMUP_CHUNKS + PERF_CHUNKS:
        pytest.skip("недостаточно файлов для замера производительности")

    chunks = [_load_first_chunk(p, cfg.chunk_samples) for p in files]

    # Прогрев.
    for c in chunks[:WARMUP_CHUNKS]:
        detector.predict(c)

    pre_times = []
    t1_times = []
    t2_times = []
    e2e_times = []
    for c in chunks[WARMUP_CHUNKS : WARMUP_CHUNKS + PERF_CHUNKS]:
        r = detector.predict(c)
        pre_times.append(r["preprocess_ms"])
        t1_times.append(r["time_1d_ms"])
        if r["triggered_2d"]:
            t2_times.append(r["time_2d_ms"])
        e2e_times.append(r["e2e_ms"])

    assert np.mean(pre_times) < MAX_PREPROCESS_MS, f"preprocess mean {np.mean(pre_times):.1f}"
    assert np.mean(t1_times) < MAX_1D_MS, f"1d mean {np.mean(t1_times):.1f}"
    if t2_times:
        assert np.mean(t2_times) < MAX_2D_MS, f"2d mean {np.mean(t2_times):.1f}"
    assert np.mean(e2e_times) < MAX_E2E_MS, f"e2e mean {np.mean(e2e_times):.1f}"
    # Все тайминги должны быть неотрицательные.
    for arr in (pre_times, t1_times, e2e_times):
        assert all(x >= 0.0 for x in arr)


def test_performance_p95_vs_mean(detector, cfg, yes_dir):
    files = _sample_files(yes_dir, WARMUP_CHUNKS + PERF_CHUNKS, SEED)
    if len(files) < WARMUP_CHUNKS + PERF_CHUNKS:
        pytest.skip("недостаточно файлов для p95")

    chunks = [_load_first_chunk(p, cfg.chunk_samples) for p in files]
    for c in chunks[:WARMUP_CHUNKS]:
        detector.predict(c)

    e2e = []
    for c in chunks[WARMUP_CHUNKS:]:
        e2e.append(detector.predict(c)["e2e_ms"])

    mean = float(np.mean(e2e))
    p95 = float(np.percentile(e2e, 95))
    assert p95 < P95_MULTIPLIER * mean, f"p95={p95:.1f}, mean={mean:.1f}"


def test_preprocess_memory_stable(preproc, cfg):
    # Прогрев.
    zero = np.zeros(cfg.chunk_samples, dtype=np.float32)
    for _ in range(WARMUP_CHUNKS):
        preproc.preprocess(zero)

    tracemalloc.start()
    snap1 = tracemalloc.take_snapshot()
    for _ in range(PREPROCESS_ITERS):
        preproc.preprocess(zero)
    snap2 = tracemalloc.take_snapshot()
    tracemalloc.stop()

    diffs = snap2.compare_to(snap1, "filename")
    size_diff = sum(d.size_diff for d in diffs)
    assert (
        size_diff < MAX_PREPROCESS_MEMORY_KB * 1024
    ), f"preprocess memory grew by {size_diff} bytes"


def test_detector_memory_stable(detector, cfg):
    zero = np.zeros(cfg.chunk_samples, dtype=np.float32)
    for _ in range(WARMUP_CHUNKS):
        detector.predict(zero)

    tracemalloc.start()
    snap1 = tracemalloc.take_snapshot()
    for _ in range(DETECTOR_ITERS):
        detector.predict(zero)
    snap2 = tracemalloc.take_snapshot()
    tracemalloc.stop()

    diffs = snap2.compare_to(snap1, "filename")
    size_diff = sum(d.size_diff for d in diffs)
    assert (
        size_diff < MAX_DETECTOR_MEMORY_MB * 1024 * 1024
    ), f"detector memory grew by {size_diff} bytes"


def test_golden_regression(detector, preproc, cfg, golden_path, sine_chunk_factory, silence_chunk):
    inputs = {
        "sine_1khz": sine_chunk_factory(1000.0, amp=0.5),
        "silence": silence_chunk,
    }

    current = {}
    for name, audio in inputs.items():
        mfcc, mel = preproc.preprocess(audio)
        # Копируем, т.к. предсказание использует те же буферы.
        sum_mfcc = float(np.sum(mfcc))
        sum_mel = float(np.sum(mel))
        res = detector.predict(audio)
        current[name] = {
            "prob_1d": float(res["prob_1d"]),
            "prob_2d": float(res["prob_2d"]) if res["prob_2d"] is not None else None,
            "sum_mfcc": sum_mfcc,
            "sum_mel": sum_mel,
        }

    if not golden_path.exists():
        golden_path.parent.mkdir(parents=True, exist_ok=True)
        with open(golden_path, "w", encoding="utf-8") as f:
            json.dump(current, f, indent=2)
        pytest.fail("Золотые значения созданы, перезапустите тест")

    with open(golden_path, "r", encoding="utf-8") as f:
        golden = json.load(f)

    atol = 1e-4
    for name, expected in golden.items():
        got = current[name]
        for key in ("prob_1d", "sum_mfcc", "sum_mel"):
            assert got[key] == pytest.approx(expected[key], abs=atol), (
                f"{name}/{key}: got {got[key]} exp {expected[key]}"
            )
        if expected.get("prob_2d") is None:
            assert got["prob_2d"] is None
        else:
            assert got["prob_2d"] == pytest.approx(expected["prob_2d"], abs=atol)
