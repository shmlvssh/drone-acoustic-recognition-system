import argparse
import csv
import glob
import os
import queue
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

import numpy as np
import serial
import soundfile as sf
from numpy.lib.stride_tricks import as_strided
from tflite_runtime.interpreter import Interpreter
from tqdm import tqdm

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@dataclass(frozen=True)
class Config:
    # Параметры аудио
    sr: int = 32000
    chunk_sec: float = 1.0
    chunk_samples: int = 32000
    # Шаг между соседними чанками (overlap внутри файла/потока).
    # 50% overlap: chunk_samples // 2 — как в обучающем пайплайне (tf.signal.frame
    # с frame_step=CHUNK_SAMPLES//2). Позволяет ловить дрон, появляющийся
    # на стыке секунд.
    chunk_step: int = 16000

    # Порог активации верификатора
    tau1: float = 0.5
    # Порог подтверждения ответа верификатора
    tau2: float = 0.5

    # Параметры предобработки данных
    frame_length: int = 1024
    frame_step: int = 512
    num_frames: int = 61

    fft_size: int = 1024
    n_mels: int = 96
    n_mfcc: int = 20
    fmin: float = 0.0
    fmax: float = 16000.0

    # Пути к моделям
    model_1d_path: str = os.path.join(_BASE_DIR, "models", "model_1d.tflite")
    model_2d_path: str = os.path.join(_BASE_DIR, "models", "model_2d.tflite")


class DataPreprocessor:
    """DSP-пайплайн: аудио -> (MFCC, log-mel-спектрограмма).

    Все константы (окно, мел-фильтры, матрица DCT) и выходные буферы
    преаллоцируются в конструкторе и переиспользуются на каждом вызове
    preprocess(), чтобы не плодить временные массивы.
    """

    __slots__ = (
        "cfg",
        "_window",
        "_mel_weights",
        "_dct_matrix",
        "_mfcc_buf",
        "_mel_buf",
        "_rms_buf",
        "_frames_buf",
        "_spectrum_buf",
    )

    def __init__(self, cfg: Config):
        self.cfg = cfg

        self._window = np.hanning(cfg.frame_length).astype(np.float32)
        self._mel_weights = self._build_mel_filterbank(cfg)  # (n_fft/2+1, n_mels)
        self._dct_matrix = self._build_dct_matrix(cfg)  # (n_mels, n_mfcc)

        # Преаллоцированные входные тензоры TFLite (будут заполняться in-place).
        self._mfcc_buf = np.empty((1, cfg.num_frames, cfg.n_mfcc), dtype=np.float32)
        self._mel_buf = np.empty((1, cfg.num_frames, cfg.n_mels, 1), dtype=np.float32)
        # log-RMS чанка как доп.вход 1D-CNN: возвращает сети абсолютный
        # уровень сигнала, который MFCC отбрасывают через DCT.
        self._rms_buf = np.empty((1, 1), dtype=np.float32)

        # Преаллоцированные рабочие буферы hot-loop'а preprocess().
        self._frames_buf = np.empty(
            (cfg.num_frames, cfg.frame_length), dtype=np.float32
        )
        self._spectrum_buf = np.empty(
            (cfg.num_frames, cfg.fft_size // 2 + 1), dtype=np.float32
        )

    @staticmethod
    def _build_mel_filterbank(cfg: Config) -> np.ndarray:
        """Матрица мел-фильтров (n_fft/2+1, n_mels). Совместимо с tf.signal."""
        hz_to_mel = lambda hz: 2595.0 * np.log10(1.0 + hz / 700.0)
        mel_to_hz = lambda mel: 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

        num_bins = cfg.fft_size // 2 + 1
        mel_points = np.linspace(hz_to_mel(cfg.fmin), hz_to_mel(cfg.fmax), cfg.n_mels + 2)
        hz_points = mel_to_hz(mel_points)
        bins = np.floor((cfg.fft_size + 1) * hz_points / cfg.sr).astype(int)

        filters = np.zeros((num_bins, cfg.n_mels), dtype=np.float32)
        for i in range(cfg.n_mels):
            left, center, right = bins[i], bins[i + 1], bins[i + 2]
            if center > left:
                filters[left:center, i] = np.arange(center - left) / (center - left)
            if right > center:
                filters[center:right, i] = np.arange(right - center, 0, -1) / (right - center)
        return filters

    @staticmethod
    def _build_dct_matrix(cfg: Config) -> np.ndarray:
        """Матрица DCT-II (n_mels, n_mfcc) для перехода log-mel -> MFCC."""
        n = np.arange(cfg.n_mels)
        k = np.arange(cfg.n_mfcc)
        dct = np.cos(np.pi * k[np.newaxis, :] * (2 * n[:, np.newaxis] + 1) / (2 * cfg.n_mels))
        dct *= np.sqrt(2.0 / cfg.n_mels)
        return dct.astype(np.float32)

    def _frame(self, audio: np.ndarray) -> np.ndarray:
        """Zero-copy фрейминг через as_strided -> view (num_frames, frame_length)."""
        audio = np.ascontiguousarray(audio)
        byte_step = audio.strides[0]
        return as_strided(
            audio,
            shape=(self.cfg.num_frames, self.cfg.frame_length),
            strides=(self.cfg.frame_step * byte_step, byte_step),
        )

    def preprocess(self, audio: np.ndarray):
        """Готовит входы обеих моделей из одного блока аудио.

        Возвращает (mfcc_input, mel_input, rms_input) — ссылки на внутренние
        буферы формы (1, num_frames, n_mfcc), (1, num_frames, n_mels, 1) и (1, 1).
        Буферы переиспользуются: результаты валидны до следующего вызова.
        """
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32, copy=False)

        # log-RMS чанка считается ДО матричных операций. log compress'ит
        # большой динамический диапазон (1e-5..1.0) в линейный (-5..0),
        # удобный для Dense-слоя в 1D-CNN.
        rms_raw = float(np.sqrt(np.mean(audio * audio)))
        self._rms_buf[0, 0] = np.log(rms_raw + 1e-7)

        # Фрейминг без копии; умножение на окно пишет в преаллоцированный
        # _frames_buf — FFT получает contiguous-вход без временных массивов.
        np.multiply(self._frame(audio), self._window, out=self._frames_buf)

        # Амплитудный спектр (num_frames, fft_size/2+1). rfft не поддерживает out=,
        # а np.abs с out= падает на старых numpy, где rfft возвращает complex128.
        rfft_out = np.fft.rfft(self._frames_buf, n=self.cfg.fft_size)
        # |z|^2 = re^2 + im^2 — пишем в float32-буфер напрямую, минуя промежуточный массив от np.abs.
        np.multiply(rfft_out.real, rfft_out.real, out=self._spectrum_buf)
        self._spectrum_buf += rfft_out.imag * rfft_out.imag
        np.sqrt(self._spectrum_buf, out=self._spectrum_buf)

        # View на (num_frames, n_mels)-срез внутри преаллоцированного mel-буфера.
        # Это не копия: операции через mel_view пишут напрямую в self._mel_buf.
        mel_view = self._mel_buf[0, :, :, 0]

        # mel-power сразу в buffer; затем +eps и log — всё in-place.
        np.matmul(self._spectrum_buf, self._mel_weights, out=mel_view)
        mel_view += 1e-6
        np.log(mel_view, out=mel_view)

        # MFCC = log_mel @ DCT; input (mel_view) и output (_mfcc_buf[0]) — разные буферы.
        np.matmul(mel_view, self._dct_matrix, out=self._mfcc_buf[0])

        return self._mfcc_buf, self._mel_buf, self._rms_buf


class CascadeDetector:
    """Каскадный детектор: быстрый 1D-CNN + точный 2D-CNN верификатор."""

    __slots__ = (
        "cfg",
        "preproc",
        "tau1",
        "tau2",
        "_interp_1d",
        "_interp_2d",
        "_idx_in_1d_mfcc",
        "_idx_in_1d_rms",
        "_idx_out_1d",
        "_idx_in_2d",
        "_idx_out_2d",
        "_result",
    )

    def __init__(
        self,
        cfg: Config,
        preproc: DataPreprocessor,
        tau1: Optional[float] = None,
        tau2: Optional[float] = None,
    ):
        self.cfg = cfg
        self.preproc = preproc
        self.tau1 = cfg.tau1 if tau1 is None else tau1
        self.tau2 = cfg.tau2 if tau2 is None else tau2

        self._interp_1d = self._load_interpreter(cfg.model_1d_path)
        # 1D-модель двухвходовая (mfcc + rms). Индексы тензоров ищем по имени,
        # т.к. порядок входов в TFLite не гарантирован.
        details_1d = self._interp_1d.get_input_details()
        self._idx_in_1d_mfcc = next(d["index"] for d in details_1d if "mfcc" in d["name"])
        self._idx_in_1d_rms = next(d["index"] for d in details_1d if "rms" in d["name"])
        self._idx_out_1d = self._interp_1d.get_output_details()[0]["index"]

        self._interp_2d = self._load_interpreter(cfg.model_2d_path)
        self._idx_in_2d = self._interp_2d.get_input_details()[0]["index"]
        self._idx_out_2d = self._interp_2d.get_output_details()[0]["index"]

        # Фиксированный словарь-результат: переиспользуется между вызовами,
        # чтобы не плодить аллокации на каждом чанке.
        self._result = {
            "prob_1d": 0.0,
            "prob_2d": None,
            "prediction": 0,
            "preprocess_ms": 0.0,
            "time_1d_ms": 0.0,
            "time_2d_ms": 0.0,
            "e2e_ms": 0.0,
            "triggered_2d": False,
        }

    @staticmethod
    def _load_interpreter(model_path: str) -> Interpreter:
        interpreter = Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter

    def predict(self, audio: np.ndarray) -> dict:
        """Каскадная инференция.

        1. DSP считается один раз; MFCC и log-mel формируются из одного STFT.
        2. 1D-CNN даёт prob_1d; если prob_1d > tau1, запускается 2D-CNN.
        3. prediction = 1 только если 2D подтвердил (prob_2d > tau2).

        Тайминги в результате разнесены по стадиям:
        preprocess_ms — DSP; time_1d_ms — только инференция 1D (без DSP);
        time_2d_ms — только инференция 2D (0.0 если не сработал триггер);
        e2e_ms — полное время predict() по стенным часам.
        """
        t0 = time.perf_counter()
        mfcc_input, mel_input, rms_input = self.preproc.preprocess(audio)
        t_pre = time.perf_counter()

        self._interp_1d.set_tensor(self._idx_in_1d_mfcc, mfcc_input)
        self._interp_1d.set_tensor(self._idx_in_1d_rms, rms_input)
        self._interp_1d.invoke()
        prob_1d = float(self._interp_1d.get_tensor(self._idx_out_1d).flat[0])
        t_1d = time.perf_counter()

        preprocess_ms = (t_pre - t0) * 1000.0
        time_1d_ms = (t_1d - t_pre) * 1000.0

        triggered_2d = prob_1d > self.tau1
        res = self._result
        res["prob_1d"] = prob_1d
        res["preprocess_ms"] = preprocess_ms
        res["time_1d_ms"] = time_1d_ms
        res["triggered_2d"] = triggered_2d

        if triggered_2d:
            self._interp_2d.set_tensor(self._idx_in_2d, mel_input)
            self._interp_2d.invoke()
            prob_2d = float(self._interp_2d.get_tensor(self._idx_out_2d).flat[0])
            t_2d = time.perf_counter()
            res["prob_2d"] = prob_2d
            res["time_2d_ms"] = (t_2d - t_1d) * 1000.0
            res["prediction"] = 1 if prob_2d > self.tau2 else 0
        else:
            res["prob_2d"] = None
            res["time_2d_ms"] = 0.0
            res["prediction"] = 0

        res["e2e_ms"] = (time.perf_counter() - t0) * 1000.0
        return res


# --- Источники аудио ---------------------------------------------------------


def _resample_linear(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Линейная ресемплизация 1-D float32 аудио средствами numpy.

    Линейной интерполяции достаточно для mel-фичей: aliasing при downsample
    усредняется треугольными мел-фильтрами и не влияет на классификацию.
    """
    if src_sr == dst_sr:
        return audio.astype(np.float32, copy=False)
    n_src = len(audio)
    if n_src == 0:
        return np.zeros(0, dtype=np.float32)
    n_dst = int(round(n_src * dst_sr / src_sr))
    if n_dst <= 0:
        return np.zeros(0, dtype=np.float32)
    src_t = np.arange(n_src, dtype=np.float64)
    dst_t = np.linspace(0.0, n_src - 1, n_dst, dtype=np.float64)
    return np.interp(dst_t, src_t, audio).astype(np.float32, copy=False)


def _to_mono_float32(audio: np.ndarray) -> np.ndarray:
    """Стерео -> моно (mean), приведение dtype к float32 (без копии при возможности)."""
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32, copy=False)
    return audio


def _loop_to_length(audio: np.ndarray, target_len: int) -> np.ndarray:
    """Зацикливает аудио (np.tile) до длины >= target_len и обрезает.

    Используется для файлов короче окна chunk_samples: вместо набивания тишиной
    повторяем содержимое — сеть видит «реальный» сигнал на всё окно,
    а не «короткий звук + тишина», что ломает распределение MFCC/RMS.
    """
    n = len(audio)
    if n == 0:
        return np.zeros(target_len, dtype=np.float32)
    if n >= target_len:
        return audio[:target_len]
    repeats = (target_len + n - 1) // n  # ceil(target_len / n)
    looped = np.tile(audio, repeats)
    return looped[:target_len]


def _peak_normalize(audio: np.ndarray) -> np.ndarray:
    """Peak-нормализация in-place. Возвращает тот же массив (для удобства цепочки)."""
    max_val = float(np.max(np.abs(audio))) if audio.size else 0.0
    if max_val > 0:
        audio /= max_val
    return audio


def _iter_windows(
    audio: np.ndarray, chunk_samples: int, chunk_step: int
) -> Iterator[Tuple[int, np.ndarray]]:
    """Итерирует (start_index, window) с шагом chunk_step.

    Хвост короче chunk_samples отбрасывается: он бы потребовал паддинга,
    а перед вызовом вызывающая сторона уже гарантирует len(audio) >= chunk_samples
    (через _loop_to_length для коротких файлов).
    """
    n = len(audio)
    if n < chunk_samples:
        return
    last_start = n - chunk_samples
    start = 0
    while start <= last_start:
        yield start, audio[start : start + chunk_samples]
        start += chunk_step


class AudioSource(ABC):
    """Интерфейс источника аудио: итерирует блоки фиксированной длины.

    Каждый элемент — кортеж (chunk, metadata), где chunk имеет форму
    (cfg.chunk_samples,) и dtype float32, а metadata содержит полезную
    для потребителя информацию (например, метку класса или индекс чанка).
    """

    cfg: Config

    @abstractmethod
    def __iter__(self) -> Iterator[Tuple[np.ndarray, dict]]:
        """Yields (chunk, metadata).

        Контракт metadata по конкретным реализациям:
          * LabeledDirectorySource -> {"path": str, "true_label": int,
                                       "chunk_index": int, "num_chunks": int,
                                       "is_last_chunk": bool, "start_sec": float}
          * LongRecordingSource    -> {"chunk_index": int, "start_sec": float}
          * SerialMicSource        -> {"chunk_index": int, "start_sec": float}
        """
        ...


class LabeledDirectorySource(AudioSource):
    """Читает wav-файлы из директории с поддиректориями drone/ и noise/.

    Проходит каждый файл целиком окнами длиной chunk_samples с шагом chunk_step
    (overlap, как в обучающем пайплайне). Файлы короче chunk_samples
    зацикливаются (np.tile), а не дополняются тишиной — иначе короткий
    дрон-файл превращался бы в «короткий сигнал + тишина», что ломает
    распределение признаков относительно train.

    Метка определяется по наличию подстроки "drone" в пути.
    Метаданные включают `chunk_index`, `num_chunks`, `is_last_chunk` —
    потребитель (run_evaluation) агрегирует предсказания на уровне файла.
    """

    def __init__(self, cfg: Config, root_dir: str, first_second_only: bool = False):
        self.cfg = cfg
        self.root_dir = root_dir
        # first_second_only=True — старое поведение: только первое окно файла
        # (первая секунда). Полезно для быстрых прогонов и сравнения с прошлыми
        # цифрами. По умолчанию проходим файл целиком окнами с overlap.
        self.first_second_only = first_second_only
        self.files = sorted(
            glob.glob(os.path.join(root_dir, "**", "*.wav"), recursive=True)
        )

    def __iter__(self) -> Iterator[Tuple[np.ndarray, dict]]:
        chunk_samples = self.cfg.chunk_samples
        chunk_step = self.cfg.chunk_step
        for path in self.files:
            try:
                audio, src_sr = sf.read(path, dtype="float32", always_2d=False)
            except Exception as e:
                print(f"Ошибка чтения файла {path}: {e}")
                continue

            audio = _to_mono_float32(audio)
            if src_sr != self.cfg.sr:
                audio = _resample_linear(audio, src_sr, self.cfg.sr)

            # Короткий файл: зацикливаем до полного окна. Для длинных
            # _loop_to_length вернёт срез [:chunk_samples] — но этого мы
            # хотим избежать, поэтому вызываем только когда реально коротко.
            if len(audio) < chunk_samples:
                audio = _loop_to_length(audio, chunk_samples)

            # ВАЖНО: НЕ нормализуем файл целиком. В обучении (augment_and_extract)
            # peak-norm применяется к КАЖДОМУ чанку независимо: result_audio /
            # max(|result_audio|) при max>1e-5. File-wide norm даёт другой масштаб
            # для всех окон, кроме того, в котором лежит файловый пик, → сдвиг
            # log-mel на каждый бин, который BatchNorm не амортизирует.

            true_label = 1 if "drone" in path.lower() else 0
            windows = list(_iter_windows(audio, chunk_samples, chunk_step))
            if self.first_second_only:
                # Берём только первое окно — старое поведение «только первая секунда».
                windows = windows[:1]
            num_chunks = len(windows)
            for ci, (start, window) in enumerate(windows):
                # .copy() обязателен: соседние окна перекрываются (chunk_step <
                # chunk_samples), in-place деление испортило бы общий хвост.
                chunk = window.copy()
                m = float(np.max(np.abs(chunk)))
                # Порог 1e-5 точно как в augment_and_extract обучения.
                if m > 1e-5:
                    chunk /= m
                yield chunk, {
                    "path": path,
                    "true_label": true_label,
                    "chunk_index": ci,
                    "num_chunks": num_chunks,
                    "is_last_chunk": ci == num_chunks - 1,
                    "start_sec": start / self.cfg.sr,
                }


class LongRecordingSource(AudioSource):
    """Режет длинную wav-запись на пересекающиеся (overlap) секундные чанки.

    Шаг между чанками — cfg.chunk_step (по умолчанию 50% overlap), как в обучении.
    Запись читается потоково через скользящий буфер, чтобы не держать весь
    файл в памяти на длинных записях.
    """

    def __init__(self, cfg: Config, wav_path: str):
        self.cfg = cfg
        self.wav_path = wav_path

    def __iter__(self) -> Iterator[Tuple[np.ndarray, dict]]:
        chunk_samples = self.cfg.chunk_samples
        chunk_step = self.cfg.chunk_step
        with sf.SoundFile(self.wav_path) as f:
            src_sr = f.samplerate
            need_resample = src_sr != self.cfg.sr
            # Читаем в исходном SR блоками ~chunk_step, ресемплируем по куску
            # и накапливаем в буфере выходного SR.
            read_block_src = (
                chunk_step
                if not need_resample
                else int(round(chunk_step * src_sr / self.cfg.sr))
            )
            buf = np.zeros(0, dtype=np.float32)
            i = 0
            eof = False
            while not eof:
                # Доберём буфер до >= chunk_samples (или до EOF).
                while len(buf) < chunk_samples:
                    block = f.read(read_block_src, dtype="float32")
                    if len(block) == 0:
                        eof = True
                        break
                    block = _to_mono_float32(block)
                    if need_resample:
                        block = _resample_linear(block, src_sr, self.cfg.sr)
                    buf = np.concatenate((buf, block)) if buf.size else block

                while len(buf) >= chunk_samples:
                    chunk = buf[:chunk_samples].copy()
                    # Per-chunk peak-normalize (порог 1e-5 как в обучении —
                    # настоящую тишину не усиливаем до peak=1).
                    m = float(np.max(np.abs(chunk)))
                    if m > 1e-5:
                        chunk /= m
                    yield chunk, {
                        "chunk_index": i,
                        "start_sec": i * chunk_step / self.cfg.sr,
                    }
                    i += 1
                    # Сдвиг на chunk_step (overlap), а не на chunk_samples.
                    buf = buf[chunk_step:]
                    # Если EOF и в буфере осталось меньше chunk_samples — выходим
                    # из внутреннего while; внешний while завершится по eof.
                    if eof and len(buf) < chunk_samples:
                        break
            if buf.size:
                print(f"Dropped {len(buf)} tail samples (<1s)")


class SerialMicSource(AudioSource):
    """Поток аудио-сэмплов с МК по serial-порту.

    Протокол: пакеты 4×0x63 + 6000 байт payload + 4×0x49. Сэмплы — signed
    24-bit LE, упакованные в 4 байта (MSB guard + 3 байта значения).
    Хост стартует стрим командой 0x53 0x00 и останавливает 0x45 0x00.
    """

    _PREFIX = b"\x63" * 4
    _SUFFIX = b"\x49" * 4
    _PAYLOAD_SIZE = 6000
    _PACKET_SIZE = 6008
    _SAMPLES_PER_PACKET = 1500
    _BAUD = 2_000_000
    _STARTUP_DISCARD = 3
    _CMD_DETECT = b"\x53\x00"
    _CMD_WAIT = b"\x45\x00"
    _MSB_SCALE = float(1 << 23)
    _QUEUE_MAXSIZE = 128

    def __init__(self, cfg: Config, port: str, timeout_sec: float = 2.0):
        self.cfg = cfg
        self.port = port
        self.timeout_sec = timeout_sec
        self._lost_bytes = 0
        self._last_warn_time = 0.0
        self._last_warn_lost = 0

    def _reader_loop(self, ser, q, stop_event):
        """Drain serial port into queue in dedicated thread. Drop-oldest on full."""
        try:
            while not stop_event.is_set():
                try:
                    data = ser.read(self._PACKET_SIZE)
                except Exception as exc:
                    print(f"[SerialMicSource] reader fatal: {exc!r}")
                    return
                if not data:
                    continue
                try:
                    q.put(data, block=False)
                except queue.Full:
                    # Drop-oldest: consumer is slow, freshest bytes matter most.
                    try:
                        dropped = q.get_nowait()
                        self._lost_bytes += len(dropped)
                    except queue.Empty:
                        pass
                    try:
                        q.put(data, block=False)
                    except queue.Full:
                        # Rare race: still full — drop the new data too.
                        self._lost_bytes += len(data)

                # Rate-limited overflow warning (once per ~1s).
                if self._lost_bytes != self._last_warn_lost:
                    now = time.monotonic()
                    if now - self._last_warn_time >= 1.0:
                        delta = self._lost_bytes - self._last_warn_lost
                        print(
                            f"[SerialMicSource] buffer overflow: dropped {delta} bytes "
                            f"in last ~1s (total lost: {self._lost_bytes})"
                        )
                        self._last_warn_time = now
                        self._last_warn_lost = self._lost_bytes
        except Exception as exc:
            print(f"[SerialMicSource] reader loop exited on error: {exc!r}")
        finally:
            # Разблокируем consumer'а: без этого q.get() крутился бы вечно,
            # ожидая данные от уже мёртвого reader-треда.
            stop_event.set()

    def _parse_payload_with_msb_errors(self, payload: bytes) -> np.ndarray:
        """Медленный путь: сэмпл со «съеденным» MSB-guard сдвигает поток на 1 байт."""
        out = []
        i = 0
        n = len(payload)
        while i + 4 <= n:
            b0 = payload[i]
            if b0 != 0:
                # Съеденный guard-байт: значение начинается прямо с b0 как младший байт.
                b1 = payload[i]
                b2 = payload[i + 1]
                b3 = payload[i + 2]
                advance = 3
            else:
                b1 = payload[i + 1]
                b2 = payload[i + 2]
                b3 = payload[i + 3]
                advance = 4
            value = (b3 << 16) | (b2 << 8) | b1
            if value & 0x800000:
                value -= 1 << 24
            out.append(value)
            i += advance
        arr = np.asarray(out, dtype=np.int32)
        print(f"MSB Error at packet, recovered {arr.size} samples")
        return arr

    def _parse_payload(self, payload: bytes) -> np.ndarray:
        """Fast path (векторизованный). При нарушении MSB-guard — откат в slow path."""
        arr = np.frombuffer(payload, dtype=np.uint8).reshape(-1, 4)
        if np.any(arr[:, 0] != 0):
            return self._parse_payload_with_msb_errors(payload)
        vals = (
            arr[:, 1].astype(np.int32)
            | (arr[:, 2].astype(np.int32) << 8)
            | (arr[:, 3].astype(np.int32) << 16)
        )
        # Знаковое расширение 24-бит -> int32: если бит 23 установлен, вычесть 2^24.
        # Всё в int32, без хрупкого ~0xFFFFFF (Python int с бесконечным sign-bit).
        vals -= (vals & 0x800000) << 1
        return vals

    def __iter__(self) -> Iterator[Tuple[np.ndarray, dict]]:
        chunk_samples = self.cfg.chunk_samples
        chunk_step = self.cfg.chunk_step
        # Fresh stream: reset per-stream counters.
        self._lost_bytes = 0
        self._last_warn_time = 0.0
        self._last_warn_lost = 0

        ser = serial.Serial(
            self.port,
            baudrate=self._BAUD,
            bytesize=8,
            parity=serial.PARITY_NONE,
            stopbits=1,
            timeout=self.timeout_sec,
        )
        # Bump OS RX buffer where supported (Windows); Linux pyserial no-ops.
        try:
            ser.set_buffer_size(rx_size=65536, tx_size=4096)
        except (AttributeError, OSError, NotImplementedError):
            pass

        stop_event = threading.Event()
        q: "queue.Queue[bytes]" = queue.Queue(maxsize=self._QUEUE_MAXSIZE)
        reader = threading.Thread(
            target=self._reader_loop,
            args=(ser, q, stop_event),
            daemon=True,
        )
        reader.start()

        try:
            ser.write(self._CMD_DETECT)
            ser.flush()

            buf = bytearray()
            pending: list = []
            pending_total = 0
            packets_seen = 0
            i = 0

            while True:
                try:
                    data = q.get(timeout=1.0)
                except queue.Empty:
                    if stop_event.is_set():
                        break
                    continue
                buf.extend(data)

                while True:
                    idx = buf.find(self._PREFIX)
                    if idx < 0:
                        if len(buf) > 2 * self._PACKET_SIZE:
                            # Ресинк: оставляем хвост на случай частичного префикса.
                            del buf[: len(buf) - 8]
                        break
                    if len(buf) - idx < self._PACKET_SIZE:
                        break
                    suffix_off = idx + 4 + self._PAYLOAD_SIZE
                    if bytes(buf[suffix_off : suffix_off + 4]) != self._SUFFIX:
                        # Ложный префикс: сдвигаемся на 1 байт и ищем следующий.
                        del buf[: idx + 1]
                        continue
                    payload = bytes(buf[idx + 4 : idx + 4 + self._PAYLOAD_SIZE])
                    del buf[: idx + self._PACKET_SIZE]

                    packets_seen += 1
                    if packets_seen <= self._STARTUP_DISCARD:
                        continue

                    samples = self._parse_payload(payload)
                    pending.append(samples)
                    pending_total += samples.size

                    while pending_total >= chunk_samples:
                        raw = np.concatenate(pending) if len(pending) > 1 else pending[0]
                        chunk = raw[:chunk_samples].astype(np.float32) / self._MSB_SCALE
                        # Per-chunk peak-norm с порогом 1e-5 как в обучении.
                        m = float(np.max(np.abs(chunk)))
                        if m > 1e-5:
                            chunk /= m
                        meta = {
                            "chunk_index": i,
                            "start_sec": i * chunk_step / self.cfg.sr,
                        }
                        yield chunk, meta
                        i += 1

                        # Сдвиг на chunk_step (overlap с предыдущим окном),
                        # а не на chunk_samples — латентность падает с 1с до 0.5с.
                        leftover = raw[chunk_step:]
                        pending = [leftover] if leftover.size else []
                        pending_total = leftover.size
        finally:
            stop_event.set()
            reader.join(timeout=self.timeout_sec + 1.0)
            try:
                ser.write(self._CMD_WAIT)
                ser.flush()
            except Exception:
                pass
            try:
                ser.close()
            except Exception:
                pass


# --- Режимы запуска ----------------------------------------------------------


def run_evaluation(detector: CascadeDetector, source: LabeledDirectorySource) -> None:
    """Файл-уровневая оценка: проходим весь файл окнами с overlap, агрегируем.

    Предполагаем, что файл целиком относится к одному классу (drone/noise),
    но дрон может появиться не с самого начала — поэтому решение по файлу
    принимаем как OR по чанкам (positive если хотя бы одно окно сработало).
    """
    tp = fp = tn = fn = 0
    # Per-chunk суммарные тайминги (для среднего по чанкам).
    sum_preprocess = 0.0
    sum_1d = 0.0
    sum_2d = 0.0
    sum_e2e = 0.0
    chunks_total = 0
    count_2d = 0
    triggered_count = 0
    # Per-file: сколько файлов обработали и сколько в них суммарно сработало 2D.
    files_total = 0
    files_triggered = 0

    # Состояние по текущему файлу.
    file_pred = 0
    file_triggered_any = False

    for audio, meta in tqdm(source, desc="Инференс", unit="чанк"):
        res = detector.predict(audio)

        sum_preprocess += res["preprocess_ms"]
        sum_1d += res["time_1d_ms"]
        sum_e2e += res["e2e_ms"]
        chunks_total += 1
        if res["triggered_2d"]:
            triggered_count += 1
            sum_2d += res["time_2d_ms"]
            count_2d += 1
            file_triggered_any = True

        # OR-агрегация: одно положительное окно делает файл положительным.
        if res["prediction"] == 1:
            file_pred = 1

        if meta["is_last_chunk"]:
            true_label = meta["true_label"]
            if file_pred == 1 and true_label == 1:
                tp += 1
            elif file_pred == 1 and true_label == 0:
                fp += 1
            elif file_pred == 0 and true_label == 0:
                tn += 1
            elif file_pred == 0 and true_label == 1:
                fn += 1
            files_total += 1
            if file_triggered_any:
                files_triggered += 1
            file_pred = 0
            file_triggered_any = False

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    trigger_percent_chunks = (
        (triggered_count / chunks_total) * 100 if chunks_total > 0 else 0.0
    )
    trigger_percent_files = (
        (files_triggered / files_total) * 100 if files_total > 0 else 0.0
    )

    print("\n" + "=" * 50)
    print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ РАБОТЫ СИСТЕМЫ")
    print("=" * 50)

    print("\nПРОИЗВОДИТЕЛЬНОСТЬ:")
    if chunks_total > 0:
        print(f"Всего чанков обработано: {chunks_total} (файлов: {files_total})")
        print(f"Среднее время DSP      : {sum_preprocess / chunks_total:.2f} мс")
        print(f"Среднее время 1D-CNN   : {sum_1d / chunks_total:.2f} мс")
        print(f"Среднее время общее    : {sum_e2e / chunks_total:.2f} мс")
    if count_2d > 0:
        print(f"Среднее время 2D-CNN   : {sum_2d / count_2d:.2f} мс")
    print(
        f"Срабатываний верификатора   : {trigger_percent_chunks:.2f}% "
        f"({triggered_count} из {chunks_total})"
    )

    print("\nМЕТРИКИ КЛАССИФИКАЦИИ:")
    print("Матрица ошибок:")
    print("             Predicted: DRONE | Predicted: NOISE")
    print(f"Actual DRONE |    TP: {tp:<4}    |    FN: {fn:<4}")
    print(f"Actual NOISE |    FP: {fp:<4}    |    TN: {tn:<4}")
    print("-" * 50)
    print(f"Accuracy (Точность общая) : {accuracy:.4f}")
    print(f"Precision (Точность АЛА) : {precision:.4f}")
    print(f"Recall (Полнота АЛА)     : {recall:.4f}")
    print(f"F1-Score                  : {f1_score:.4f}")
    print("=" * 50)


def run_long_recording(
    detector: CascadeDetector,
    source: LongRecordingSource,
    csv_path: str,
) -> None:
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
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
        )
        for audio, meta in tqdm(source, desc="Обработка", unit="чанк"):
            res = detector.predict(audio)
            writer.writerow(
                [
                    meta["chunk_index"],
                    f"{meta['start_sec']:.3f}",
                    f"{res['prob_1d']:.6f}",
                    f"{res['prob_2d']:.6f}" if res["prob_2d"] is not None else "",
                    int(res["triggered_2d"]),
                    res["prediction"],
                    f"{res['preprocess_ms']:.3f}",
                    f"{res['time_1d_ms']:.3f}",
                    f"{res['time_2d_ms']:.3f}",
                    f"{res['e2e_ms']:.3f}",
                ]
            )
    print(f"Результаты записаны в {csv_path}")


# --- CLI ---------------------------------------------------------------------


def _unit_interval(s):
    x = float(s)
    if not 0.0 <= x <= 1.0:
        raise argparse.ArgumentTypeError(f"must be in [0, 1], got {x}")
    return x


def _build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--tau1",
        type=_unit_interval,
        default=None,
        help="Порог активации верификатора. По умолчанию - 0.5",
    )
    common.add_argument(
        "--tau2",
        type=_unit_interval,
        default=None,
        help="Порог подтверждения ответа верификатора. По умолчанию - 0.5",
    )

    parser = argparse.ArgumentParser(
        description="Каскадный детектор БПЛА"
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    p_eval = sub.add_parser(
        "evaluate",
        parents=[common],
        help="Оценка на размеченном наборе данных с подпапками drone/ и noise/",
    )
    p_eval.add_argument(
        "--data-dir",
        default=os.path.join(_BASE_DIR, "test"),
        help="Корневая директория тестового набора данных. По умолчанию - test",
    )
    p_eval.add_argument(
        "--first-second",
        action="store_true",
        help=(
            "Обработка только первой секунды каждого файла. По умолчанию — обрабатывается каждая секунда файла с "
            "последующей агрегацией предсказания на уровне файла"
        ),
    )

    p_long = sub.add_parser(
        "process",
        parents=[common],
        help="Обработка длинной wav-записи с выгрузкой результатов в CSV",
    )
    p_long.add_argument("--input", required=True, help="Путь к wav-файлу")
    p_long.add_argument("--output", required=True, help="Путь к выходному csv")

    p_stream = sub.add_parser(
        "stream",
        parents=[common],
        help="Онлайн-обработка потока данных с акустического детектора по serial-порту",
    )
    p_stream.add_argument(
        "--port",
        required=True,
        help="Имя serial-порта. Например, /dev/ttyUSB0",
    )
    p_stream.add_argument("--output", required=True, help="Путь к выходному csv")
    p_stream.add_argument(
        "--timeout",
        type=float,
        default=2.0,
        help="Таймаут чтения serial-порта в секундах. По умолчанию - 2.0",
    )

    return parser


def main():
    args = _build_parser().parse_args()

    cfg = Config()
    preproc = DataPreprocessor(cfg)
    detector = CascadeDetector(cfg, preproc, tau1=args.tau1, tau2=args.tau2)

    if args.mode == "evaluate":
        source = LabeledDirectorySource(
            cfg, args.data_dir, first_second_only=args.first_second
        )
        run_evaluation(detector, source)
    elif args.mode == "process":
        source = LongRecordingSource(cfg, args.input)
        run_long_recording(detector, source, args.output)
    elif args.mode == "stream":
        source = SerialMicSource(cfg, args.port, timeout_sec=args.timeout)
        run_long_recording(detector, source, args.output)


if __name__ == "__main__":
    main()
