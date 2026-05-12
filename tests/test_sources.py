import queue as _queue
import threading
import time

import numpy as np
import pytest
import soundfile as sf

import main


# --- LabeledDirectorySource --------------------------------------------------


def test_labeled_empty_dir(cfg, tmp_path):
    src = main.LabeledDirectorySource(cfg, str(tmp_path))
    assert list(src) == []


def test_labeled_lists_sorted(cfg, tmp_labeled_dataset):
    src = main.LabeledDirectorySource(cfg, str(tmp_labeled_dataset))
    assert src.files == sorted(src.files)


def test_labeled_label_from_path_drone(cfg, tmp_path):
    d = tmp_path / "drone"
    d.mkdir()
    sf.write(str(d / "x.wav"), np.zeros(32000, dtype=np.float32), 32000)
    src = main.LabeledDirectorySource(cfg, str(tmp_path))
    _, meta = next(iter(src))
    assert meta["true_label"] == 1


def test_labeled_label_from_path_noise(cfg, tmp_path):
    n = tmp_path / "noise"
    n.mkdir()
    sf.write(str(n / "x.wav"), np.zeros(32000, dtype=np.float32), 32000)
    src = main.LabeledDirectorySource(cfg, str(tmp_path))
    _, meta = next(iter(src))
    assert meta["true_label"] == 0


def test_labeled_label_uppercase_drone(cfg, tmp_path):
    d = tmp_path / "DRONE_AUDIO"
    d.mkdir()
    sf.write(str(d / "x.wav"), np.zeros(32000, dtype=np.float32), 32000)
    src = main.LabeledDirectorySource(cfg, str(tmp_path))
    _, meta = next(iter(src))
    assert meta["true_label"] == 1


def test_labeled_chunk_shape_and_dtype(cfg, tmp_labeled_dataset):
    src = main.LabeledDirectorySource(cfg, str(tmp_labeled_dataset))
    for chunk, _ in src:
        assert chunk.shape == (cfg.chunk_samples,)
        assert chunk.dtype == np.float32


def test_labeled_reads_first_second(cfg, tmp_path):
    d = tmp_path / "drone"
    d.mkdir()
    t = np.arange(3 * 32000, dtype=np.float32) / 32000.0
    data = (0.5 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
    sf.write(str(d / "x.wav"), data, 32000)
    src = main.LabeledDirectorySource(cfg, str(tmp_path))
    chunk, _ = next(iter(src))
    assert chunk.shape == (32000,)


def test_labeled_short_wav_padded(cfg, tmp_path):
    d = tmp_path / "drone"
    d.mkdir()
    short = np.ones(16000, dtype=np.float32) * 0.5
    sf.write(str(d / "s.wav"), short, 32000)
    src = main.LabeledDirectorySource(cfg, str(tmp_path))
    chunk, _ = next(iter(src))
    assert chunk.shape == (32000,)
    assert np.all(chunk[16000:] == 0.0)


def test_labeled_continues_on_broken_file(cfg, tmp_path, capsys):
    d = tmp_path / "drone"
    d.mkdir()
    (d / "broken.wav").write_bytes(b"not-a-wav")
    sf.write(str(d / "good.wav"), np.zeros(32000, dtype=np.float32), 32000)
    src = main.LabeledDirectorySource(cfg, str(tmp_path))
    results = list(src)
    assert len(results) == 1
    assert "Ошибка чтения файла" in capsys.readouterr().out


def test_labeled_stereo_averaged(cfg, tmp_path):
    d = tmp_path / "drone"
    d.mkdir()
    stereo = np.zeros((32000, 2), dtype=np.float32)
    stereo[:, 0] = 0.5
    stereo[:, 1] = -0.5
    sf.write(str(d / "s.wav"), stereo, 32000)
    src = main.LabeledDirectorySource(cfg, str(tmp_path))
    chunk, _ = next(iter(src))
    assert chunk.ndim == 1
    assert np.allclose(chunk, 0.0)


@pytest.mark.parametrize("src_sr", [8000, 16000, 22050, 44100, 48000])
def test_labeled_resamples_various_src_sr(cfg, tmp_path, src_sr):
    d = tmp_path / "drone"
    d.mkdir()
    sf.write(str(d / "x.wav"), np.zeros(src_sr, dtype=np.float32), src_sr)
    src = main.LabeledDirectorySource(cfg, str(tmp_path))
    chunk, _ = next(iter(src))
    assert chunk.shape == (cfg.chunk_samples,)
    assert chunk.dtype == np.float32


def test_labeled_resamples_short_wav_padded(cfg, tmp_path):
    # 0.5 сек @ 16000 = 8000 src сэмплов → ~16000 target после resample → pad до 32000.
    d = tmp_path / "drone"
    d.mkdir()
    sf.write(str(d / "x.wav"), np.full(8000, 0.4, dtype=np.float32), 16000)
    chunk, _ = next(iter(main.LabeledDirectorySource(cfg, str(tmp_path))))
    assert chunk.shape == (cfg.chunk_samples,)
    # Хвост — нули, ресемплированная часть нормализована (peak 0.4 > 0.15 → /0.4).
    assert np.all(chunk[20000:] == 0.0)
    assert chunk[8000] > 0.5


def test_labeled_resamples_stereo_wav(cfg, tmp_path):
    # Стерео + другая частота: моно-фолд должен пройти до resample.
    d = tmp_path / "drone"
    d.mkdir()
    stereo = np.zeros((16000, 2), dtype=np.float32)
    stereo[:, 0] = 0.3
    stereo[:, 1] = 0.3
    sf.write(str(d / "x.wav"), stereo, 16000)
    chunk, _ = next(iter(main.LabeledDirectorySource(cfg, str(tmp_path))))
    assert chunk.ndim == 1
    assert chunk.shape == (cfg.chunk_samples,)


# --- LongRecordingSource -----------------------------------------------------


def test_long_resamples_wrong_sr_yields_chunk(cfg, tmp_path):
    # 1 сек при src_sr=16000 → 1 чанк формы (chunk_samples,) при cfg.sr=32000.
    p = tmp_path / "wrong.wav"
    sf.write(str(p), np.zeros(16000, dtype=np.float32), 16000)
    src = main.LongRecordingSource(cfg, str(p))
    chunks = list(src)
    assert len(chunks) == 1
    chunk, _ = chunks[0]
    assert chunk.shape == (cfg.chunk_samples,)
    assert chunk.dtype == np.float32


@pytest.mark.parametrize("src_sr", [8000, 16000, 22050, 44100, 48000])
def test_long_resamples_various_src_sr(cfg, tmp_path, src_sr):
    p = tmp_path / f"r_{src_sr}.wav"
    sf.write(str(p), np.zeros(src_sr, dtype=np.float32), src_sr)
    chunks = list(main.LongRecordingSource(cfg, str(p)))
    assert len(chunks) == 1
    chunk, _ = chunks[0]
    assert chunk.shape == (cfg.chunk_samples,)
    assert chunk.dtype == np.float32


def test_long_resample_start_sec_in_target_rate(cfg, tmp_path):
    # 2 сек при src_sr=16000 → 2 чанка с start_sec 0.0, 1.0 в target rate.
    p = tmp_path / "two_sec.wav"
    sf.write(str(p), np.zeros(2 * 16000, dtype=np.float32), 16000)
    metas = [m for _, m in main.LongRecordingSource(cfg, str(p))]
    assert [m["chunk_index"] for m in metas] == [0, 1]
    assert metas[0]["start_sec"] == 0.0
    assert metas[1]["start_sec"] == pytest.approx(1.0)


def test_long_resample_ramp_preserves_monotonicity(cfg, tmp_path):
    # Линейная интерполяция на рампе — монотонна; концы около ±1 после peak-нормализации.
    src_sr = 16000
    p = tmp_path / "ramp.wav"
    data = np.linspace(-0.5, 0.5, src_sr, dtype=np.float32)
    sf.write(str(p), data, src_sr)
    chunk, _ = next(iter(main.LongRecordingSource(cfg, str(p))))
    assert chunk.shape == (cfg.chunk_samples,)
    assert (np.diff(chunk) >= -1e-5).all()
    assert chunk[0] < -0.95
    assert chunk[-1] > 0.95


def test_long_chunk_count_exact(cfg, tmp_path):
    sf.write(str(tmp_path / "l.wav"), np.zeros(3 * 32000, dtype=np.float32), 32000)
    src = main.LongRecordingSource(cfg, str(tmp_path / "l.wav"))
    assert len(list(src)) == 3


def test_long_drops_tail_24sec(cfg, tmp_path, capsys):
    # 2.4 сек -> 2 чанка, хвост 12800.
    sf.write(str(tmp_path / "l.wav"), np.zeros(int(2.4 * 32000), dtype=np.float32), 32000)
    src = main.LongRecordingSource(cfg, str(tmp_path / "l.wav"))
    chunks = list(src)
    assert len(chunks) == 2
    assert "Dropped 12800 tail samples" in capsys.readouterr().out


def test_long_exactly_chunk_samples_no_dropped(cfg, tmp_path, capsys):
    sf.write(str(tmp_path / "l.wav"), np.zeros(32000, dtype=np.float32), 32000)
    src = main.LongRecordingSource(cfg, str(tmp_path / "l.wav"))
    chunks = list(src)
    assert len(chunks) == 1
    assert "Dropped" not in capsys.readouterr().out


def test_long_per_chunk_peak_normalization(cfg, tmp_path):
    data = np.concatenate(
        [np.ones(32000, dtype=np.float32) * 0.9, np.ones(32000, dtype=np.float32) * 0.1]
    )
    sf.write(str(tmp_path / "l.wav"), data, 32000)
    chunks = [c for c, _ in main.LongRecordingSource(cfg, str(tmp_path / "l.wav"))]
    assert np.isclose(np.max(np.abs(chunks[0])), 1.0, atol=1e-3)
    assert np.isclose(np.max(np.abs(chunks[1])), 1.0, atol=1e-3)


def test_long_stereo_to_mono(cfg, tmp_path):
    stereo = np.zeros((32000, 2), dtype=np.float32)
    stereo[:, 0] = 0.5
    stereo[:, 1] = 0.5
    sf.write(str(tmp_path / "l.wav"), stereo, 32000)
    chunk, _ = next(iter(main.LongRecordingSource(cfg, str(tmp_path / "l.wav"))))
    assert chunk.ndim == 1


def test_long_metadata_schema(cfg, tmp_path):
    sf.write(str(tmp_path / "l.wav"), np.zeros(2 * 32000, dtype=np.float32), 32000)
    metas = [m for _, m in main.LongRecordingSource(cfg, str(tmp_path / "l.wav"))]
    assert metas[0] == {"chunk_index": 0, "start_sec": 0.0}
    assert metas[1]["chunk_index"] == 1
    assert metas[1]["start_sec"] == pytest.approx(1.0)


# --- SerialMicSource (moked) -------------------------------------------------


def _make_stream(packet_factory, n_packets, samples=None):
    return b"".join(packet_factory(samples) for _ in range(n_packets))


def test_serial_parses_valid_stream_yields_chunk(cfg, packet_factory, fake_serial_cls, monkeypatch):
    blob = _make_stream(packet_factory, 25, np.ones(1500, dtype=np.int32) * 1000)

    orig_cls = fake_serial_cls

    def factory(*args, **kwargs):
        inst = orig_cls(*args, **kwargs)
        inst.feed(blob)
        return inst

    monkeypatch.setattr(main.serial, "Serial", factory, raising=False)
    src = main.SerialMicSource(cfg, port="MOCK")
    it = iter(src)
    chunk, meta = next(it)
    assert chunk.shape == (cfg.chunk_samples,)
    assert chunk.dtype == np.float32
    assert meta["chunk_index"] == 0
    assert meta["start_sec"] == 0.0


def test_serial_discards_first_three_packets(cfg, packet_factory, fake_serial_cls, monkeypatch):
    # Первые 3 пакета — маркер (значение != 0); остальные 22 — нули.
    marker = np.ones(1500, dtype=np.int32) * 0x400000
    zero = np.zeros(1500, dtype=np.int32)
    blob = b"".join(packet_factory(marker) for _ in range(3))
    blob += b"".join(packet_factory(zero) for _ in range(22))

    def factory(*args, **kwargs):
        inst = fake_serial_cls(*args, **kwargs)
        inst.feed(blob)
        return inst

    monkeypatch.setattr(main.serial, "Serial", factory, raising=False)
    src = main.SerialMicSource(cfg, port="MOCK")
    chunk, _ = next(iter(src))
    # Маркеры отброшены; чанк должен быть нулями (до нормализации).
    assert np.max(np.abs(chunk)) == 0.0


def test_serial_resync_after_garbage_before_prefix(cfg, packet_factory, fake_serial_cls, monkeypatch):
    garbage = b"\x00\x01\x02\x03" * 50
    blob = garbage + _make_stream(packet_factory, 25)

    def factory(*args, **kwargs):
        inst = fake_serial_cls(*args, **kwargs)
        inst.feed(blob)
        return inst

    monkeypatch.setattr(main.serial, "Serial", factory, raising=False)
    src = main.SerialMicSource(cfg, port="MOCK")
    chunk, _ = next(iter(src))
    assert chunk.shape == (cfg.chunk_samples,)


def test_serial_false_prefix_bad_suffix_shifts_one_byte(cfg, packet_factory, fake_serial_cls, monkeypatch):
    bogus = b"\x63" * 4 + b"\x00" * 6000 + b"\x00" * 4
    blob = bogus + _make_stream(packet_factory, 25)

    def factory(*args, **kwargs):
        inst = fake_serial_cls(*args, **kwargs)
        inst.feed(blob)
        return inst

    monkeypatch.setattr(main.serial, "Serial", factory, raising=False)
    src = main.SerialMicSource(cfg, port="MOCK")
    chunk, _ = next(iter(src))
    assert chunk.shape == (cfg.chunk_samples,)


def _factory_that_disconnects(fake_serial_cls):
    """Factory: fake_serial, который после опустошения буфера кидает OSError.

    Это нужно для корректного завершения _reader_loop и, как следствие,
    генератора __iter__ (через stop_event.set() в reader.finally).
    """

    def factory(*args, **kwargs):
        inst = fake_serial_cls(*args, **kwargs)
        original_read = inst.read

        def read_then_die(n, _st={"served": False}):
            if not _st["served"]:
                _st["served"] = True
                return original_read(n)
            raise OSError("disconnect")

        inst.read = read_then_die
        return inst

    return factory


def test_serial_writes_start_cmd(cfg, fake_serial_cls, monkeypatch):
    monkeypatch.setattr(main.serial, "Serial", _factory_that_disconnects(fake_serial_cls), raising=False)
    src = main.SerialMicSource(cfg, port="MOCK")
    list(iter(src))
    inst = fake_serial_cls.instances[0]
    assert inst.writes[0] == b"\x53\x00"


def test_serial_writes_stop_cmd_on_close(cfg, fake_serial_cls, monkeypatch):
    monkeypatch.setattr(main.serial, "Serial", _factory_that_disconnects(fake_serial_cls), raising=False)
    src = main.SerialMicSource(cfg, port="MOCK")
    list(iter(src))
    inst = fake_serial_cls.instances[0]
    assert b"\x45\x00" in inst.writes
    assert inst.closed


@pytest.mark.parametrize("exc_cls", [AttributeError, OSError, NotImplementedError])
def test_serial_set_buffer_size_missing_no_crash(cfg, fake_serial_cls, monkeypatch, exc_cls):
    def factory(*args, **kwargs):
        inst = fake_serial_cls(*args, **kwargs)
        inst._set_buffer_size_error = exc_cls("no buf size")
        original_read = inst.read

        def read_then_die(n, _st={"served": False}):
            if not _st["served"]:
                _st["served"] = True
                return original_read(n)
            raise OSError("disconnect")

        inst.read = read_then_die
        return inst

    monkeypatch.setattr(main.serial, "Serial", factory, raising=False)
    src = main.SerialMicSource(cfg, port="MOCK")
    list(iter(src))  # Не должно падать


def test_serial_samples_scaled_to_unit_interval(cfg, packet_factory, fake_serial_cls, monkeypatch):
    max_pos = (1 << 23) - 1
    s = np.full(1500, max_pos, dtype=np.int32)
    blob = _make_stream(packet_factory, 25, s)

    def factory(*args, **kwargs):
        inst = fake_serial_cls(*args, **kwargs)
        inst.feed(blob)
        return inst

    monkeypatch.setattr(main.serial, "Serial", factory, raising=False)
    src = main.SerialMicSource(cfg, port="MOCK")
    chunk, _ = next(iter(src))
    assert np.max(np.abs(chunk)) <= 1.0 + 1e-6
    assert np.max(np.abs(chunk)) > 0.9


def test_serial_chunk_index_increments(cfg, packet_factory, fake_serial_cls, monkeypatch):
    blob = _make_stream(packet_factory, 60)

    def factory(*args, **kwargs):
        inst = fake_serial_cls(*args, **kwargs)
        inst.feed(blob)
        original_read = inst.read

        def read_then_die(n):
            chunk = original_read(n)
            if not chunk and inst._pos >= len(inst._data):
                raise OSError("end of stream")
            return chunk

        inst.read = read_then_die
        return inst

    monkeypatch.setattr(main.serial, "Serial", factory, raising=False)
    src = main.SerialMicSource(cfg, port="MOCK")
    metas = [m for _, m in src]
    assert len(metas) >= 2
    assert metas[1]["chunk_index"] == 1
    assert metas[1]["start_sec"] > metas[0]["start_sec"]


def test_serial_reader_thread_is_daemon(cfg, fake_serial_cls, monkeypatch):
    captured = {}
    real_thread_cls = threading.Thread

    def capturing_thread(*args, **kwargs):
        th = real_thread_cls(*args, **kwargs)
        captured["thread"] = th
        return th

    monkeypatch.setattr(main.serial, "Serial", _factory_that_disconnects(fake_serial_cls), raising=False)
    monkeypatch.setattr(main.threading, "Thread", capturing_thread)
    src = main.SerialMicSource(cfg, port="MOCK")
    list(iter(src))
    assert captured["thread"].daemon is True


# --- _reader_loop ------------------------------------------------------------


def test_reader_loop_stops_on_stop_event(cfg, fake_serial_cls):
    src = main.SerialMicSource(cfg, port="MOCK")
    fake = fake_serial_cls("MOCK", 2_000_000, 8, "N", 1, 2.0)
    fake.feed(b"\x00" * 6008 * 100)
    stop_event = threading.Event()
    q = _queue.Queue(maxsize=4)
    t = threading.Thread(target=src._reader_loop, args=(fake, q, stop_event), daemon=True)
    t.start()
    stop_event.set()
    t.join(timeout=3.0)
    assert not t.is_alive()


def test_reader_loop_exits_on_read_exception(cfg, fake_serial_cls):
    src = main.SerialMicSource(cfg, port="MOCK")
    fake = fake_serial_cls("MOCK", 2_000_000, 8, "N", 1, 2.0)
    fake._read_exception = IOError("disconnect")
    stop_event = threading.Event()
    q = _queue.Queue(maxsize=4)
    t = threading.Thread(target=src._reader_loop, args=(fake, q, stop_event), daemon=True)
    t.start()
    t.join(timeout=3.0)
    assert not t.is_alive()
    assert stop_event.is_set()


def test_reader_loop_drop_oldest_on_full_queue(cfg, fake_serial_cls):
    src = main.SerialMicSource(cfg, port="MOCK")
    fake = fake_serial_cls("MOCK", 2_000_000, 8, "N", 1, 2.0)
    # Много мелких чтений; очередь маленькая.
    fake.feed(b"X" * 10000)
    stop_event = threading.Event()
    q = _queue.Queue(maxsize=2)
    t = threading.Thread(target=src._reader_loop, args=(fake, q, stop_event), daemon=True)
    t.start()
    # Подождём пока reader заполнит очередь и начнёт дропать.
    time.sleep(0.3)
    stop_event.set()
    t.join(timeout=3.0)
    assert src._lost_bytes > 0


def test_reader_loop_overflow_warning_rate_limited(cfg, fake_serial_cls, capsys, monkeypatch):
    src = main.SerialMicSource(cfg, port="MOCK")
    fake = fake_serial_cls("MOCK", 2_000_000, 8, "N", 1, 2.0)
    fake.feed(b"X" * 20000)

    # Фиксируем monotonic на значение > 1.0: первый warning должен пройти
    # (now - _last_warn_time(=0.0) >= 1.0), последующие — подавиться (same now).
    monkeypatch.setattr(main.time, "monotonic", lambda: 100.0)
    stop_event = threading.Event()
    q = _queue.Queue(maxsize=2)
    t = threading.Thread(target=src._reader_loop, args=(fake, q, stop_event), daemon=True)
    t.start()
    time.sleep(0.3)
    stop_event.set()
    t.join(timeout=3.0)
    out = capsys.readouterr().out
    # Должен быть ровно один warning (rate-limit срабатывает на повторных событиях).
    assert out.count("buffer overflow") == 1, out
