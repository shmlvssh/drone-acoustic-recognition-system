import csv
import os
import sys

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
    pytest.mark.skipif(not _real_tflite(), reason="tflite_runtime недоступен"),
    pytest.mark.skipif(not _models_exist(), reason="tflite модели не найдены"),
]


# --- Schema ------------------------------------------------------------------


def test_result_schema_keys(detector, silence_chunk):
    res = detector.predict(silence_chunk)
    expected = {
        "prob_1d",
        "prob_2d",
        "prediction",
        "preprocess_ms",
        "time_1d_ms",
        "time_2d_ms",
        "e2e_ms",
        "triggered_2d",
    }
    assert set(res.keys()) == expected


def test_prob_1d_in_unit_interval(detector, cfg):
    rng = np.random.default_rng(0)
    for _ in range(5):
        chunk = rng.uniform(-0.5, 0.5, cfg.chunk_samples).astype(np.float32)
        res = detector.predict(chunk)
        assert 0.0 <= res["prob_1d"] <= 1.0


def test_prob_2d_in_unit_interval_when_triggered(cfg, preproc, silence_chunk):
    det = main.CascadeDetector(cfg, preproc, tau1=-0.1, tau2=0.5)
    res = det.predict(silence_chunk)
    assert res["triggered_2d"]
    assert res["prob_2d"] is not None
    assert 0.0 <= res["prob_2d"] <= 1.0


def test_result_dict_reused(detector, silence_chunk):
    res1 = detector.predict(silence_chunk)
    res2 = detector.predict(silence_chunk)
    assert id(res1) == id(res2)


def test_trigger_off_tau1_above_one(cfg, preproc, silence_chunk):
    det = main.CascadeDetector(cfg, preproc, tau1=1.1)
    res = det.predict(silence_chunk)
    assert res["triggered_2d"] is False
    assert res["prob_2d"] is None
    assert res["time_2d_ms"] == 0.0
    assert res["prediction"] == 0


def test_trigger_on_tau1_below_zero(cfg, preproc, silence_chunk):
    det = main.CascadeDetector(cfg, preproc, tau1=-0.1, tau2=-0.1)
    res = det.predict(silence_chunk)
    assert res["triggered_2d"] is True
    assert res["prob_2d"] is not None
    assert res["time_2d_ms"] > 0.0
    assert res["prediction"] == 1


def test_predict_determinism(detector, sine_chunk_factory):
    audio = sine_chunk_factory(2000.0, amp=0.5)
    r1 = dict(detector.predict(audio))
    r2 = dict(detector.predict(audio))
    assert r1["prob_1d"] == pytest.approx(r2["prob_1d"], abs=1e-7)
    if r1["prob_2d"] is not None and r2["prob_2d"] is not None:
        assert r1["prob_2d"] == pytest.approx(r2["prob_2d"], abs=1e-7)


def test_e2e_approx_sum_of_components(detector, silence_chunk):
    res = detector.predict(silence_chunk)
    parts = res["preprocess_ms"] + res["time_1d_ms"] + res["time_2d_ms"]
    # e2e включает также оверхед на копирование/контроль: ±20% разумно.
    assert res["e2e_ms"] >= parts - 1e-6
    assert res["e2e_ms"] <= parts + max(parts * 0.5, 5.0)


def test_no_trigger_time_2d_zero_prediction_zero(cfg, preproc, silence_chunk):
    det = main.CascadeDetector(cfg, preproc, tau1=1.1)
    res = det.predict(silence_chunk)
    assert res["time_2d_ms"] == 0.0
    assert res["prediction"] == 0


def test_prediction_is_int(detector, silence_chunk):
    res = detector.predict(silence_chunk)
    assert isinstance(res["prediction"], int)


# --- run_evaluation ----------------------------------------------------------


def test_run_evaluation_on_tmp_dataset_prints_results(detector, cfg, tmp_labeled_dataset, capsys):
    source = main.LabeledDirectorySource(cfg, str(tmp_labeled_dataset))
    main.run_evaluation(detector, source)
    out = capsys.readouterr().out
    assert "ИТОГОВЫЕ РЕЗУЛЬТАТЫ" in out
    assert "Accuracy" in out
    assert "Precision" in out
    assert "Recall" in out
    assert "F1-Score" in out
    assert "Матрица ошибок" in out


def test_run_evaluation_empty_dataset_no_crash(detector, cfg, tmp_path):
    source = main.LabeledDirectorySource(cfg, str(tmp_path))
    main.run_evaluation(detector, source)  # не падает


# --- run_long_recording ------------------------------------------------------


def test_run_long_recording_csv_schema(detector, cfg, tmp_path):
    data = np.zeros(3 * 32000, dtype=np.float32)
    wav = tmp_path / "in.wav"
    sf.write(str(wav), data, 32000)
    out_csv = tmp_path / "out.csv"

    source = main.LongRecordingSource(cfg, str(wav))
    main.run_long_recording(detector, source, str(out_csv))

    with open(out_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    expected_header = [
        "chunk_index",
        "start_sec",
        "prob_1d",
        "prob_2d",
        "triggered_2d",
        "prediction",
        "preprocess_ms",
        "time_1d_ms",
        "time_2d_ms",
        "e2e_ms",
    ]
    assert rows[0] == expected_header
    assert len(rows) - 1 == 3


def test_run_long_recording_prob_2d_empty_when_not_triggered(cfg, preproc, tmp_path):
    det = main.CascadeDetector(cfg, preproc, tau1=1.1)
    sf.write(str(tmp_path / "in.wav"), np.zeros(32000, dtype=np.float32), 32000)
    out_csv = tmp_path / "out.csv"
    source = main.LongRecordingSource(cfg, str(tmp_path / "in.wav"))
    main.run_long_recording(det, source, str(out_csv))

    with open(out_csv, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    data_row = rows[1]
    header = rows[0]
    idx_p2d = header.index("prob_2d")
    idx_trig = header.index("triggered_2d")
    assert data_row[idx_p2d] == ""
    assert data_row[idx_trig] == "0"
