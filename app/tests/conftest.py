# Установка на целевом устройстве (RPi):
#   pip install pytest numpy soundfile
#   pip install pyserial         # опц., только если тестируется Serial-режим
#   pip install tflite_runtime   # опц., только если тестируется детектор
#
# Без pyserial / tflite_runtime коллекция тестов не падает: модули
# подменяются заглушками в sys.modules ДО импорта main. Тесты, реально
# использующие эти библиотеки, защищены pytest.importorskip / skipif.

import sys
import time
import types
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


_ROOT = Path(__file__).resolve().parents[1]
_APP = _ROOT / "app"
if str(_APP) not in sys.path:
    sys.path.insert(0, str(_APP))


def _install_stub(name: str, module: types.ModuleType) -> None:
    sys.modules.setdefault(name, module)


# --- pyserial stub -----------------------------------------------------------
try:  # pragma: no cover - environment-dependent
    import serial as _real_serial  # noqa: F401
except Exception:
    _serial_stub = types.ModuleType("serial")

    class _StubSerial:
        def __init__(self, *args, **kwargs):
            raise ImportError("pyserial not available")

    _serial_stub.Serial = _StubSerial
    _serial_stub.PARITY_NONE = "N"
    _install_stub("serial", _serial_stub)


# --- tflite_runtime stub -----------------------------------------------------
try:  # pragma: no cover - environment-dependent
    import tflite_runtime.interpreter as _real_tflite  # noqa: F401
except Exception:
    _tflite_pkg = types.ModuleType("tflite_runtime")
    _tflite_interp = types.ModuleType("tflite_runtime.interpreter")

    class _StubInterpreter:
        _is_stub = True

        def __init__(self, *args, **kwargs):
            # Реальная работа требует настоящий tflite_runtime; тесты,
            # которые хотят Interpreter, должны быть защищены importorskip.
            pass

        def allocate_tensors(self):
            pass

        def get_input_details(self):
            return [{"index": 0}]

        def get_output_details(self):
            return [{"index": 0}]

        def set_tensor(self, *_a, **_kw):
            pass

        def invoke(self):
            pass

        def get_tensor(self, _idx):
            return np.zeros((1, 1), dtype=np.float32)

    _tflite_interp.Interpreter = _StubInterpreter
    _tflite_pkg.interpreter = _tflite_interp
    _install_stub("tflite_runtime", _tflite_pkg)
    _install_stub("tflite_runtime.interpreter", _tflite_interp)


# --- tqdm stub (пассивный, только если отсутствует) --------------------------
try:  # pragma: no cover
    import tqdm as _real_tqdm  # noqa: F401
except Exception:
    _tqdm_mod = types.ModuleType("tqdm")

    def _passthrough(iterable=None, *args, **kwargs):
        return iterable if iterable is not None else iter(())

    _tqdm_mod.tqdm = _passthrough
    _install_stub("tqdm", _tqdm_mod)


import main  # noqa: E402


# --- Session paths -----------------------------------------------------------


@pytest.fixture(scope="session")
def root_dir():
    return _ROOT


@pytest.fixture(scope="session")
def app_dir():
    return _APP


@pytest.fixture(scope="session")
def dataset_dir(app_dir):
    p = app_dir / "dataset"
    if not p.exists() or not any(p.iterdir()):
        pytest.skip("app/dataset не найден или пуст")
    return p


@pytest.fixture(scope="session")
def yes_dir(dataset_dir):
    for name in ("yes_drone", "drone", "yes"):
        cand = dataset_dir / name
        if cand.exists() and any(cand.glob("*.wav")):
            return cand
    pytest.skip("yes_drone/ пуст или не найден")


@pytest.fixture(scope="session")
def no_dir(dataset_dir):
    for name in ("no_drone", "noise", "no"):
        cand = dataset_dir / name
        if cand.exists() and any(cand.glob("*.wav")):
            return cand
    pytest.skip("no_drone/ пуст или не найден")


@pytest.fixture(scope="session")
def models_available(app_dir):
    return (app_dir / "models" / "model_1d.tflite").exists() and (
        app_dir / "models" / "model_2d.tflite"
    ).exists()


@pytest.fixture(scope="session")
def golden_path(root_dir):
    return root_dir / "tests" / "golden.json"


# --- Core fixtures -----------------------------------------------------------


@pytest.fixture(scope="session")
def cfg():
    return main.Config()


@pytest.fixture(scope="session")
def preproc(cfg):
    return main.DataPreprocessor(cfg)


def _real_tflite_available() -> bool:
    mod = sys.modules.get("tflite_runtime.interpreter")
    # stub-модули мы маркируем: у них Interpreter — наш _StubInterpreter класс.
    return mod is not None and not getattr(mod.Interpreter, "_is_stub", False)


@pytest.fixture(scope="session")
def detector(cfg, preproc, models_available):
    if not _real_tflite_available():
        pytest.skip("tflite_runtime недоступен (стаб)")
    if not models_available:
        pytest.skip("tflite-модели отсутствуют")
    return main.CascadeDetector(cfg, preproc)


# --- Audio helpers -----------------------------------------------------------


@pytest.fixture
def silence_chunk(cfg):
    return np.zeros(cfg.chunk_samples, dtype=np.float32)


@pytest.fixture
def sine_chunk_factory(cfg):
    def _make(freq, amp=0.5):
        t = np.arange(cfg.chunk_samples, dtype=np.float32) / cfg.sr
        return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)

    return _make


@pytest.fixture
def tmp_wav_factory(tmp_path):
    def _make(samples, sr=32000, channels=1, name="test.wav"):
        path = tmp_path / name
        data = np.asarray(samples)
        if channels > 1 and data.ndim == 1:
            data = np.stack([data] * channels, axis=1)
        sf.write(str(path), data.astype(np.float32, copy=False), sr)
        return path

    return _make


@pytest.fixture
def tmp_labeled_dataset(tmp_path):
    """tmp_path/drone/{a,b}.wav (sine 2kHz) + tmp_path/noise/{c,d}.wav (seed=42)."""
    sr = 32000
    t = np.arange(sr, dtype=np.float32) / sr
    sine = (0.5 * np.sin(2 * np.pi * 2000.0 * t)).astype(np.float32)
    rng = np.random.default_rng(42)
    noise = rng.uniform(-0.3, 0.3, sr).astype(np.float32)

    drone_dir = tmp_path / "drone"
    noise_dir = tmp_path / "noise"
    drone_dir.mkdir()
    noise_dir.mkdir()

    sf.write(str(drone_dir / "a.wav"), sine, sr)
    sf.write(str(drone_dir / "b.wav"), sine * 0.7, sr)
    sf.write(str(noise_dir / "c.wav"), noise, sr)
    sf.write(str(noise_dir / "d.wav"), noise * 0.8, sr)
    return tmp_path


# --- FakeSerial --------------------------------------------------------------


class FakeSerial:
    """Пишется тестами монкипатчем main.serial.Serial на этот класс."""

    instances = []

    def __init__(self, port, baudrate, bytesize, parity, stopbits, timeout):
        self.port = port
        self.baudrate = baudrate
        self.bytesize = bytesize
        self.parity = parity
        self.stopbits = stopbits
        self.timeout = timeout
        self.writes = []
        self._data = bytearray()
        self._pos = 0
        self._read_exception = None
        self._missing_set_buffer_size = False
        self._set_buffer_size_error = None
        self.closed = False
        FakeSerial.instances.append(self)

    def feed(self, blob: bytes) -> None:
        self._data.extend(blob)

    def read(self, n):
        if self._read_exception:
            exc = self._read_exception
            self._read_exception = None
            raise exc
        if self._pos >= len(self._data):
            # Имитируем таймаут.
            time.sleep(0.01)
            return b""
        end = min(self._pos + n, len(self._data))
        chunk = bytes(self._data[self._pos : end])
        self._pos = end
        return chunk

    def write(self, data):
        self.writes.append(bytes(data))
        return len(data)

    def flush(self):
        pass

    def close(self):
        self.closed = True

    def set_buffer_size(self, rx_size, tx_size):
        if self._missing_set_buffer_size:
            raise AttributeError("no set_buffer_size")
        if self._set_buffer_size_error:
            raise self._set_buffer_size_error


@pytest.fixture
def fake_serial_cls(monkeypatch):
    FakeSerial.instances = []
    monkeypatch.setattr(main.serial, "Serial", FakeSerial, raising=False)
    return FakeSerial


# --- Serial packet factory ---------------------------------------------------


@pytest.fixture
def packet_factory():
    """Пакет: 4*0x63 + 6000 байт payload + 4*0x49.

    payload — 1500 сэмплов int24 LE: [MSB-guard=0x00, lo, mid, hi].
    """

    def _make(samples=None):
        if samples is None:
            samples = np.zeros(1500, dtype=np.int32)
        samples = np.asarray(samples, dtype=np.int32)
        assert samples.size == 1500, "pack requires 1500 samples"
        payload = bytearray(6000)
        for i, s in enumerate(samples):
            v = int(s) & 0xFFFFFF
            payload[i * 4 + 0] = 0x00
            payload[i * 4 + 1] = v & 0xFF
            payload[i * 4 + 2] = (v >> 8) & 0xFF
            payload[i * 4 + 3] = (v >> 16) & 0xFF
        return b"\x63" * 4 + bytes(payload) + b"\x49" * 4

    return _make
