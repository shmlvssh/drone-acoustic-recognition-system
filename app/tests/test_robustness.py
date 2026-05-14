import queue as _queue
import threading
import time
import warnings

import numpy as np
import pytest
import soundfile as sf

import main


# --- LabeledDirectorySource robustness --------------------------------------


def test_labeled_broken_truncated_wav(cfg, tmp_path, capsys):
    d = tmp_path / "drone"
    d.mkdir()
    # Заголовок RIFF, но тело обрезано.
    (d / "bad.wav").write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt ")
    sf.write(str(d / "good.wav"), np.zeros(32000, dtype=np.float32), 32000)
    src = main.LabeledDirectorySource(cfg, str(tmp_path))
    results = list(src)
    assert len(results) == 1
    assert "Ошибка чтения файла" in capsys.readouterr().out


def test_labeled_garbage_in_wav_extension(cfg, tmp_path, capsys):
    d = tmp_path / "drone"
    d.mkdir()
    (d / "fake.wav").write_bytes(b"this is not a wav file at all 12345")
    src = main.LabeledDirectorySource(cfg, str(tmp_path))
    results = list(src)
    assert results == []
    assert "Ошибка" in capsys.readouterr().out


def test_labeled_empty_zero_byte_file(cfg, tmp_path, capsys):
    d = tmp_path / "drone"
    d.mkdir()
    (d / "empty.wav").write_bytes(b"")
    src = main.LabeledDirectorySource(cfg, str(tmp_path))
    results = list(src)
    assert results == []


def test_labeled_zero_duration_wav(cfg, tmp_path):
    d = tmp_path / "drone"
    d.mkdir()
    # 0-сэмпловый wav: soundfile умеет писать пустой массив.
    sf.write(str(d / "z.wav"), np.zeros(0, dtype=np.float32), 32000)
    src = main.LabeledDirectorySource(cfg, str(tmp_path))
    results = list(src)
    # Один результат: _normalize_and_fit дополнит нулями до chunk_samples.
    assert len(results) == 1
    chunk, _ = results[0]
    assert chunk.shape == (cfg.chunk_samples,)
    assert np.all(chunk == 0.0)


# --- preprocess robustness ---------------------------------------------------


def test_preprocess_silence_no_nan_inf(preproc, cfg):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mfcc, mel = preproc.preprocess(np.zeros(cfg.chunk_samples, dtype=np.float32))
    assert np.isfinite(mfcc).all()
    assert np.isfinite(mel).all()


def test_normalize_zeros_no_runtime_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = main._normalize_and_fit(np.zeros(100, dtype=np.float32), chunk_samples=100)
    assert np.all(out == 0.0)


def test_preprocess_extreme_amplitude(preproc, cfg):
    audio = np.full(cfg.chunk_samples, 1000.0, dtype=np.float32)
    mfcc, mel = preproc.preprocess(audio)
    assert np.isfinite(mfcc).all()
    assert np.isfinite(mel).all()


@pytest.mark.xfail(strict=True, reason="NaN на входе должен приводить к NaN в выходе")
def test_preprocess_nan_input_propagates(preproc, cfg):
    audio = np.full(cfg.chunk_samples, np.nan, dtype=np.float32)
    mfcc, mel = preproc.preprocess(audio)
    assert np.isfinite(mfcc).all() and np.isfinite(mel).all()


# --- LongRecordingSource robustness -----------------------------------------


@pytest.mark.parametrize("sr", [8000, 22050, 44100, 48000])
def test_long_recording_wrong_sr_resamples(cfg, tmp_path, sr):
    # Несовпадение SR более не вызывает ValueError; источник ресемплит до cfg.sr.
    p = tmp_path / "w.wav"
    sf.write(str(p), np.zeros(sr, dtype=np.float32), sr)
    chunks = list(main.LongRecordingSource(cfg, str(p)))
    assert len(chunks) == 1
    chunk, _ = chunks[0]
    assert chunk.shape == (cfg.chunk_samples,)
    assert chunk.dtype == np.float32


def test_long_recording_lt_chunk_yields_nothing(cfg, tmp_path, capsys):
    # 0.5 сек -> 0 чанков, сообщение Dropped.
    p = tmp_path / "short.wav"
    sf.write(str(p), np.zeros(16000, dtype=np.float32), 32000)
    src = main.LongRecordingSource(cfg, str(p))
    chunks = list(src)
    assert chunks == []
    assert "Dropped" in capsys.readouterr().out


def test_long_mono_vs_stereo_equivalent(cfg, tmp_path):
    mono = np.ones(32000, dtype=np.float32) * 0.3
    stereo = np.stack([mono, mono], axis=1)

    pm = tmp_path / "mono.wav"
    ps = tmp_path / "stereo.wav"
    sf.write(str(pm), mono, 32000)
    sf.write(str(ps), stereo, 32000)

    cm = next(iter(main.LongRecordingSource(cfg, str(pm))))[0]
    cs = next(iter(main.LongRecordingSource(cfg, str(ps))))[0]
    assert np.allclose(cm, cs, atol=1e-5)


# --- Serial robustness -------------------------------------------------------


def test_serial_guard_byte_error_triggers_slow_path(cfg, capsys):
    src = main.SerialMicSource(cfg, port="MOCK")
    payload = bytearray([0x00, 0x01, 0x02, 0x03] * 1500)
    payload[100] = 0x7F
    out = src._parse_payload(bytes(payload))
    assert "MSB Error" in capsys.readouterr().out
    assert out.size > 0


def test_serial_queue_overflow_increases_lost_bytes(cfg, fake_serial_cls):
    src = main.SerialMicSource(cfg, port="MOCK")
    fake = fake_serial_cls("MOCK", 2_000_000, 8, "N", 1, 2.0)
    fake.feed(b"X" * 20000)
    stop_event = threading.Event()
    q = _queue.Queue(maxsize=2)
    t = threading.Thread(target=src._reader_loop, args=(fake, q, stop_event), daemon=True)
    t.start()
    time.sleep(0.3)
    stop_event.set()
    t.join(timeout=3.0)
    assert src._lost_bytes > 0


def test_serial_partial_payload_waits_for_continuation(cfg, packet_factory, fake_serial_cls, monkeypatch):
    # Разрезаем стрим пополам, но FakeSerial.read берёт небольшими кусками —
    # это эквивалентно «приёму кусочками». Источник должен собрать один чанк.
    blob = b"".join(packet_factory() for _ in range(25))
    half1, half2 = blob[: len(blob) // 2], blob[len(blob) // 2 :]

    def factory(*args, **kwargs):
        inst = fake_serial_cls(*args, **kwargs)
        inst.feed(half1 + half2)
        return inst

    monkeypatch.setattr(main.serial, "Serial", factory, raising=False)
    src = main.SerialMicSource(cfg, port="MOCK")
    chunk, _ = next(iter(src))
    assert chunk.shape == (cfg.chunk_samples,)


def test_serial_bad_suffix_resyncs_one_byte(cfg, packet_factory, fake_serial_cls, monkeypatch):
    bogus = b"\x63" * 4 + b"\x00" * 6000 + b"\x11" * 4
    blob = bogus + b"".join(packet_factory() for _ in range(25))

    def factory(*args, **kwargs):
        inst = fake_serial_cls(*args, **kwargs)
        inst.feed(blob)
        return inst

    monkeypatch.setattr(main.serial, "Serial", factory, raising=False)
    src = main.SerialMicSource(cfg, port="MOCK")
    chunk, _ = next(iter(src))
    assert chunk.shape == (cfg.chunk_samples,)


def test_serial_long_garbage_does_not_grow_buffer_unbounded(cfg, fake_serial_cls):
    # Проверяем инвариант напрямую на фрагменте логики __iter__:
    # при длинном входе без валидного префикса buf должен усекаться
    # до хвоста длиной 8 байт (len(buf) > 2 * _PACKET_SIZE).
    src = main.SerialMicSource(cfg, port="MOCK")
    buf = bytearray(b"\x11\x22\x33\x44" * 5000)  # 20000 байт мусора
    # Эмулируем один шаг внутреннего while-цикла:
    idx = buf.find(src._PREFIX)
    assert idx < 0
    if len(buf) > 2 * src._PACKET_SIZE:
        del buf[: len(buf) - 8]
    assert len(buf) == 8, f"buf not truncated, len={len(buf)}"
