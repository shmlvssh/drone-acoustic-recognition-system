import argparse
import dataclasses
import warnings

import numpy as np
import pytest

import main


# --- Mel filterbank ----------------------------------------------------------


def test_mel_shape(cfg, preproc):
    assert preproc._mel_weights.shape == (cfg.fft_size // 2 + 1, cfg.n_mels)


def test_mel_non_negative(preproc):
    assert np.all(preproc._mel_weights >= 0.0)


def test_mel_dtype_float32(preproc):
    assert preproc._mel_weights.dtype == np.float32


def test_mel_triangular_peak_eq_one(preproc):
    peaks = preproc._mel_weights.max(axis=0)
    assert np.allclose(peaks, 1.0, atol=1e-6)


def test_mel_triangular_shape_monotone(preproc):
    for i in range(preproc._mel_weights.shape[1]):
        col = preproc._mel_weights[:, i]
        nz = np.where(col > 0)[0]
        if len(nz) < 2:
            continue
        seq = col[nz[0] : nz[-1] + 1]
        peak = seq.argmax()
        assert np.all(np.diff(seq[: peak + 1]) >= -1e-7), f"col {i} rising"
        assert np.all(np.diff(seq[peak:]) <= 1e-7), f"col {i} falling"


def test_mel_centers_monotone(preproc):
    centers = preproc._mel_weights.argmax(axis=0)
    assert np.all(np.diff(centers) >= 0)


def test_mel_coverage_low_high(preproc):
    sums = preproc._mel_weights.sum(axis=0)
    assert np.all(sums > 0)


def test_mel_zero_outside_range(preproc):
    # Верхний край (Nyquist) не должен быть частью средних фильтров.
    nmels = preproc._mel_weights.shape[1]
    assert preproc._mel_weights[0, nmels // 2] == 0.0
    assert preproc._mel_weights[-1, nmels // 2] == 0.0


# --- DCT matrix --------------------------------------------------------------


def test_dct_shape(cfg, preproc):
    assert preproc._dct_matrix.shape == (cfg.n_mels, cfg.n_mfcc)


def test_dct_dtype(preproc):
    assert preproc._dct_matrix.dtype == np.float32


def test_dct_matches_manual_formula(cfg, preproc):
    n = np.arange(cfg.n_mels)
    k = np.arange(cfg.n_mfcc)
    expected = np.cos(np.pi * k[np.newaxis, :] * (2 * n[:, np.newaxis] + 1) / (2 * cfg.n_mels))
    expected = (expected * np.sqrt(2.0 / cfg.n_mels)).astype(np.float32)
    assert np.allclose(preproc._dct_matrix, expected, atol=1e-6)


def test_dct_columns_orthogonal_k_ge_1(cfg, preproc):
    d = preproc._dct_matrix[:, 1:]
    g = d.T @ d
    # G должна быть ≈ I (с точностью до нормы); проверяем единичную диагональ и малые off-diag.
    eye = np.eye(cfg.n_mfcc - 1, dtype=np.float32)
    assert np.allclose(g, eye, atol=1e-5), f"max dev {np.abs(g - eye).max()}"


def test_dct_k0_column_constant(cfg, preproc):
    expected = np.sqrt(2.0 / cfg.n_mels)
    assert np.allclose(preproc._dct_matrix[:, 0], expected, atol=1e-6)


# --- _frame ------------------------------------------------------------------


def test_frame_shape(cfg, preproc):
    audio = np.arange(cfg.chunk_samples, dtype=np.float32)
    frames = preproc._frame(audio)
    assert frames.shape == (cfg.num_frames, cfg.frame_length)


def test_frame_is_view(cfg, preproc):
    audio = np.arange(cfg.chunk_samples, dtype=np.float32)
    frames = preproc._frame(audio)
    assert frames.base is not None
    assert not frames.flags["OWNDATA"]


def test_frame_contents_match_manual_slices(cfg, preproc):
    audio = np.arange(cfg.chunk_samples, dtype=np.float32)
    frames = preproc._frame(audio)
    for i in (0, 10, 30, 60):
        start = i * cfg.frame_step
        expected = audio[start : start + cfg.frame_length]
        assert np.array_equal(frames[i], expected), f"frame {i}"


def test_frame_requires_contiguous_input(cfg, preproc):
    # Не-contig вход не должен падать; содержимое равно ascontiguousarray-варианту.
    src = np.arange(cfg.chunk_samples * 2, dtype=np.float32)[::2]
    assert not src.flags["C_CONTIGUOUS"]
    frames = preproc._frame(src)
    expected = preproc._frame(np.ascontiguousarray(src))
    assert np.array_equal(frames, expected)


# --- preprocess --------------------------------------------------------------


def test_preprocess_shapes(cfg, preproc, silence_chunk):
    mfcc, mel = preproc.preprocess(silence_chunk)
    assert mfcc.shape == (1, cfg.num_frames, cfg.n_mfcc)
    assert mel.shape == (1, cfg.num_frames, cfg.n_mels, 1)


def test_preprocess_returns_internal_buffers_same_id(preproc, silence_chunk):
    mfcc1, mel1 = preproc.preprocess(silence_chunk)
    mfcc2, mel2 = preproc.preprocess(silence_chunk)
    assert mfcc1 is mfcc2
    assert mel1 is mel2


def test_preprocess_silence_no_nan_no_inf(preproc, silence_chunk):
    mfcc, mel = preproc.preprocess(silence_chunk)
    assert np.isfinite(mfcc).all()
    assert np.isfinite(mel).all()
    # log(1e-6) ≈ -13.8155
    assert np.allclose(mel, np.log(1e-6), atol=1e-3)


def test_preprocess_determinism(preproc, sine_chunk_factory):
    audio = sine_chunk_factory(1000.0, amp=0.5)
    mfcc1, mel1 = preproc.preprocess(audio)
    mfcc1 = mfcc1.copy()
    mel1 = mel1.copy()
    mfcc2, mel2 = preproc.preprocess(audio)
    assert np.array_equal(mfcc1, mfcc2)
    assert np.array_equal(mel1, mel2)


def test_preprocess_sine_peak_in_expected_mel_bin(cfg, preproc, sine_chunk_factory):
    freq = 1000.0
    audio = sine_chunk_factory(freq, amp=0.5)
    _, mel = preproc.preprocess(audio)
    avg = mel[0, :, :, 0].mean(axis=0)

    # Ожидаемый fft-бин для 1 кГц: round(1000 * fft_size / sr)
    fft_bin = int(round(freq * cfg.fft_size / cfg.sr))
    # Найдём мел-фильтр, чей максимум приходится на fft_bin.
    expected_mel_bin = int(np.argmax(preproc._mel_weights[fft_bin, :]))
    got = int(np.argmax(avg))
    assert abs(got - expected_mel_bin) <= 1, f"got {got}, exp {expected_mel_bin}"


def test_preprocess_matches_reference_numpy_pipeline(cfg, preproc, sine_chunk_factory):
    audio = sine_chunk_factory(1500.0, amp=0.5)

    # Ручной эталон чистым numpy.
    window = np.hanning(cfg.frame_length).astype(np.float32)
    frames = np.stack(
        [
            audio[i * cfg.frame_step : i * cfg.frame_step + cfg.frame_length] * window
            for i in range(cfg.num_frames)
        ]
    )
    spec = np.abs(np.fft.rfft(frames, n=cfg.fft_size))
    mel_ref = np.log(spec @ preproc._mel_weights + 1e-6)
    mfcc_ref = mel_ref @ preproc._dct_matrix

    mfcc, mel = preproc.preprocess(audio)
    assert np.allclose(mel[0, :, :, 0], mel_ref, atol=1e-4, rtol=1e-4)
    assert np.allclose(mfcc[0], mfcc_ref, atol=1e-4, rtol=1e-4)


def test_preprocess_accepts_non_float32_input(cfg, preproc):
    # int16, масштабируется до ~0.5 по амплитуде.
    t = np.arange(cfg.chunk_samples) / cfg.sr
    audio = (16000 * np.sin(2 * np.pi * 1000.0 * t)).astype(np.int16)
    mfcc, mel = preproc.preprocess(audio)
    assert mfcc.dtype == np.float32
    assert mel.dtype == np.float32
    assert np.isfinite(mfcc).all()
    assert np.isfinite(mel).all()


# --- _normalize_and_fit ------------------------------------------------------


def test_normalize_mono_peak_eq_one():
    x = np.array([0.0, 0.1, -0.5, 0.3], dtype=np.float32)
    out = main._normalize_and_fit(x, chunk_samples=10)
    assert np.isclose(np.max(np.abs(out[:4])), 1.0)


def test_normalize_stereo_becomes_mono():
    x = np.zeros((100, 2), dtype=np.float32)
    x[:, 0] = 0.5
    x[:, 1] = -0.5
    out = main._normalize_and_fit(x, chunk_samples=100)
    assert out.ndim == 1
    assert np.allclose(out, 0.0)


def test_normalize_silence_no_div_by_zero():
    x = np.zeros(100, dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = main._normalize_and_fit(x, chunk_samples=100)
    assert np.all(out == 0.0)


def test_normalize_pad_short():
    x = np.ones(10, dtype=np.float32)
    out = main._normalize_and_fit(x, chunk_samples=50)
    assert out.shape == (50,)
    assert np.all(out[:10] == 1.0)
    assert np.all(out[10:] == 0.0)


def test_normalize_trim_long():
    x = np.ones(100, dtype=np.float32)
    out = main._normalize_and_fit(x, chunk_samples=50)
    assert out.shape == (50,)


def test_normalize_dtype_float32():
    x = np.ones(10, dtype=np.float64)
    out = main._normalize_and_fit(x, chunk_samples=20)
    assert out.dtype == np.float32


def test_normalize_extreme_amplitude_full_1000():
    x = np.full(100, 1000.0, dtype=np.float32)
    out = main._normalize_and_fit(x, chunk_samples=100)
    assert np.all(np.isfinite(out))
    assert np.isclose(np.max(np.abs(out)), 1.0, atol=1e-6)


# --- _resample_linear --------------------------------------------------------


def test_resample_identity_returns_same_data():
    x = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    out = main._resample_linear(x, 32000, 32000)
    assert out.dtype == np.float32
    assert np.array_equal(out, x)


def test_resample_empty_input():
    out = main._resample_linear(np.zeros(0, dtype=np.float32), 16000, 32000)
    assert out.shape == (0,)
    assert out.dtype == np.float32


def test_resample_upsample_length_doubles():
    out = main._resample_linear(np.zeros(16000, dtype=np.float32), 16000, 32000)
    assert out.shape == (32000,)
    assert out.dtype == np.float32


def test_resample_downsample_length_halves():
    out = main._resample_linear(np.zeros(44100, dtype=np.float32), 44100, 22050)
    assert out.shape == (22050,)


def test_resample_constant_signal_stays_constant():
    x = np.full(1000, 0.5, dtype=np.float32)
    out = main._resample_linear(x, 16000, 32000)
    assert out.shape == (2000,)
    assert np.allclose(out, 0.5, atol=1e-6)


def test_resample_ramp_linear_interpolation():
    # На рампе np.interp совпадает с np.linspace по эндпоинтам и шагу.
    x = np.arange(10, dtype=np.float32)
    out = main._resample_linear(x, 10, 20)
    expected = np.linspace(0.0, 9.0, 20).astype(np.float32)
    assert out.shape == (20,)
    assert np.allclose(out, expected, atol=1e-5)


def test_resample_float64_input_returns_float32():
    out = main._resample_linear(np.zeros(100, dtype=np.float64), 16000, 32000)
    assert out.dtype == np.float32


# --- _parse_payload fast path ------------------------------------------------


def test_parse_payload_simple(cfg):
    src = main.SerialMicSource(cfg, port="MOCK")
    # Три известных сэмпла, остальные нули.
    payload = bytearray(6000)
    samples = [(0x00, 0x56, 0x34, 0x12), (0x00, 0x01, 0x00, 0x00), (0x00, 0xFF, 0xFF, 0xFF)]
    for i, s in enumerate(samples):
        payload[i * 4 : (i + 1) * 4] = s
    out = src._parse_payload(bytes(payload))
    assert out[0] == 0x123456
    assert out[1] == 1
    assert out[2] == -1


def test_parse_payload_max_positive(cfg):
    src = main.SerialMicSource(cfg, port="MOCK")
    payload = bytes([0x00, 0xFF, 0xFF, 0x7F] * 1500)
    out = src._parse_payload(payload)
    assert out[0] == 0x7FFFFF


def test_parse_payload_min_negative(cfg):
    src = main.SerialMicSource(cfg, port="MOCK")
    payload = bytes([0x00, 0x00, 0x00, 0x80] * 1500)
    out = src._parse_payload(payload)
    assert out[0] == -0x800000


def test_parse_payload_negative_minus_one(cfg):
    src = main.SerialMicSource(cfg, port="MOCK")
    payload = bytes([0x00, 0xFF, 0xFF, 0xFF] * 1500)
    out = src._parse_payload(payload)
    assert out[0] == -1


def test_parse_payload_fallback_to_slow_on_nonzero_msb(cfg, capsys):
    src = main.SerialMicSource(cfg, port="MOCK")
    payload = bytearray([0x00, 0x00, 0x00, 0x00] * 1500)
    payload[0] = 0xFF
    out = src._parse_payload(bytes(payload))
    captured = capsys.readouterr()
    assert "MSB Error" in captured.out
    assert out.size > 0


def test_parse_payload_length_matches_1500(cfg):
    src = main.SerialMicSource(cfg, port="MOCK")
    payload = bytes(6000)
    out = src._parse_payload(payload)
    assert out.size == 1500


def test_parse_payload_dtype_int32(cfg):
    src = main.SerialMicSource(cfg, port="MOCK")
    payload = bytes(6000)
    out = src._parse_payload(payload)
    assert out.dtype == np.int32


# --- _parse_payload_with_msb_errors -----------------------------------------


def test_parse_payload_with_msb_errors_resync(cfg):
    src = main.SerialMicSource(cfg, port="MOCK")
    # Формируем payload где первый guard «съеден»: байты идут без ведущего 0,
    # дальше нормальные сэмплы с guard=0. Поведение медленного пути:
    # первый b0 != 0 -> b1=b0, b2=payload[1], b3=payload[2], advance=3.
    # Готовим: первый «съеденный» сэмпл: [0x11, 0x22, 0x33] (advance=3)
    # затем N нормальных: [0x00, 0x11, 0x22, 0x33] (advance=4)
    body = bytearray()
    body.extend([0x11, 0x22, 0x33])  # 3 байта, advance=3
    for _ in range(10):
        body.extend([0x00, 0x44, 0x55, 0x66])
    out = src._parse_payload_with_msb_errors(bytes(body))
    # Первый сэмпл: значение = (0x33 << 16) | (0x22 << 8) | 0x11 = 0x332211
    assert out[0] == 0x332211
    # Второй — уже с нормальным guard: 0x665544
    assert out[1] == 0x665544


def test_parse_payload_with_msb_errors_prints_recovery(cfg, capsys):
    src = main.SerialMicSource(cfg, port="MOCK")
    out = src._parse_payload_with_msb_errors(bytes([0x00, 0x01, 0x02, 0x03] * 10))
    captured = capsys.readouterr()
    assert "MSB Error" in captured.out
    assert out.size > 0


def test_parse_payload_with_msb_errors_empty_payload(cfg):
    src = main.SerialMicSource(cfg, port="MOCK")
    out = src._parse_payload_with_msb_errors(b"")
    assert out.size == 0


# --- _unit_interval ----------------------------------------------------------


def test_unit_interval_zero_ok():
    assert main._unit_interval("0") == 0.0


def test_unit_interval_one_ok():
    assert main._unit_interval("1") == 1.0


def test_unit_interval_half_ok():
    assert main._unit_interval("0.5") == 0.5


def test_unit_interval_below_zero_raises():
    with pytest.raises(argparse.ArgumentTypeError):
        main._unit_interval("-0.1")


def test_unit_interval_above_one_raises():
    with pytest.raises(argparse.ArgumentTypeError):
        main._unit_interval("1.01")


def test_unit_interval_non_float_raises():
    with pytest.raises(ValueError):
        main._unit_interval("abc")


# --- Config ------------------------------------------------------------------


def test_config_frozen():
    c = main.Config()
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.sr = 16000  # type: ignore[misc]


def test_config_chunk_samples_eq_sr_times_chunk_sec():
    c = main.Config()
    assert c.chunk_samples == int(c.sr * c.chunk_sec)


def test_config_frames_fit_in_chunk():
    c = main.Config()
    span = (c.num_frames - 1) * c.frame_step + c.frame_length
    assert span <= c.chunk_samples, f"span {span} > chunk {c.chunk_samples}"


def test_config_paths_under_app_models():
    c = main.Config()
    assert c.model_1d_path.endswith("model_1d.tflite")
    assert c.model_2d_path.endswith("model_2d.tflite")
    # Пути должны содержать "models".
    assert "models" in c.model_1d_path.replace("\\", "/")
    assert "models" in c.model_2d_path.replace("\\", "/")
