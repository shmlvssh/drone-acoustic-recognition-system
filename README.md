# Каскадный детектор АЛА

Акустический детектор АЛА на базе каскада двух CNN: 1D-CNN фильтрует поток по MFCC, а 2D-CNN подтверждает срабатывания по log-mel-спектрограмме. Программа рассчитана на встраиваемую систему с ограничением по RAM.

## Установка

### Зависимости

```bash
pip install numpy soundfile tqdm pyserial tflite-runtime
```

Для запуска на PC с полным TensorFlow вместо `tflite-runtime` подойдёт `tensorflow` (модуль `tf.lite.Interpreter`), но тогда потребуется заменить импорт в `main.py`.

### Структура директории `app/`

```
app/
├── main.py
├── models/
│   ├── model_1d.tflite    # 1D-CNN, вход: (1, 61, 13) float32 (MFCC)
│   └── model_2d.tflite    # 2D-CNN, вход: (1, 61, 64, 1) float32 (log-mel)
├── test_dataset/                  # опционально: датасет для evaluate
│   ├── drone/*.wav
│   └── noise/*.wav
└── ...
```

Пути к моделям заданы в `Config.model_1d_path` / `Config.model_2d_path`.

### Параметры аудио

Жёстко зашиты в `Config`: **32000 Гц, моно, 1 секунда на окно** (`chunk_samples = 32000`). Любой входной wav ресемплируется по факту только для `stream`-режима через смену частоты на МК; `evaluate` / `process` ожидают уже 32 кГц.

## Пороги каскада

- `--tau1` ∈ [0, 1] — порог активации верификатора. Если `prob_1d > tau1`, запускается 2D-CNN. По умолчанию `0.5`.
- `--tau2` ∈ [0, 1] — порог подтверждения. `prediction = 1`, только если `prob_2d > tau2`. По умолчанию `0.5`.

Оба аргумента валидируются (вне диапазона — ошибка CLI).

## Режимы работы

Программа имеет три subcommand'а: `evaluate`, `process`, `stream`.

### 1. `evaluate` — оценка на размеченном датасете

Читает wav-файлы из директории с подпапками `drone/` и `noise/`, берёт **первую секунду** каждого файла, считает матрицу ошибок и тайминги.

```bash
python main.py evaluate --data-dir ./test
python main.py evaluate --data-dir ./test --tau1 0.6 --tau2 0.7
```

- `--data-dir` (default: `app/test`) — корень с `drone/` и `noise/`. Рекурсивный обход `**/*.wav`. Метка определяется по наличию подстроки `drone` в пути файла.

Вывод — матрица ошибок + средние тайминги:

```
ПРОИЗВОДИТЕЛЬНОСТЬ:
Среднее время DSP (preprocess)  : X.XX мс
Среднее время 1D-CNN (Дозорный) : X.XX мс
Среднее время 2D-CNN (Эксперт)  : X.XX мс   # только по чанкам с триггером
Среднее время predict() e2e     : X.XX мс
Срабатываний Эксперта (Trigger) : XX.XX% (N из M)

МЕТРИКИ КЛАССИФИКАЦИИ:
             Predicted: DRONE | Predicted: NOISE
Actual DRONE |    TP: N       |    FN: N
Actual NOISE |    FP: N       |    TN: N
Accuracy / Precision / Recall / F1-Score
```

### 2. `process` — обработка длинной wav-записи

Режет записанный файл на непересекающиеся секундные чанки, прогоняет через каскад, пишет результаты по каждому чанку в CSV.

```bash
python main.py process --input long_recording.wav --output results.csv
```

- `--input` — путь к wav-файлу (ожидается 32 кГц; иначе `ValueError`).
- `--output` — путь к CSV.

Стриминговое чтение через `sf.SoundFile.read(chunk_samples)` — весь файл в RAM не загружается. Остаток короче 1 секунды в конце файла печатается в stdout как `Dropped N tail samples`.

### 3. `stream` — онлайн с микроконтроллера

Принимает сэмплы с МК по serial-порту (протокол `portsservice`), накапливает секундные чанки и пишет те же CSV-строки, что и `process`.

```bash
python main.py stream --port COM5 --output live.csv
python main.py stream --port /dev/ttyUSB0 --output live.csv --timeout 3.0
```

- `--port` — имя порта (Windows: `COM5`, Linux: `/dev/ttyUSB0`).
- `--output` — путь к CSV.
- `--timeout` — таймаут чтения serial (сек, default 2.0).

Завершение — **Ctrl-C**. Источник гарантированно отправит МК команду остановки и закроет порт в блоке `finally`.

#### Протокол поверх serial (кратко)

- 2 000 000 бод, 8N1, no flow control.
- Пакет: `4×0x63` + 6000 Б payload + `4×0x49` (итого 6008 Б).
- Сэмпл: 4 Б (guard `0x00` + 24-битное signed LE значение). 1500 сэмплов на пакет.
- Команда старта (хост → МК): `0x53 0x00`. Команда останова: `0x45 0x00`.
- Первые 3 пакета после старта отбрасываются (буфер МК содержит мусор).

## Формат CSV (`process` и `stream`)

| Колонка | Тип | Описание |
|---|---|---|
| `chunk_index` | int | порядковый номер секундного чанка, с 0 |
| `start_sec` | float, 3 знака | время начала чанка от старта потока/файла |
| `prob_1d` | float, 6 знаков | вероятность от 1D-CNN |
| `prob_2d` | float, 6 знаков или пусто | вероятность от 2D-CNN (пусто, если не триггернуло) |
| `triggered_2d` | 0/1 | запускался ли 2D-CNN |
| `prediction` | 0/1 | финальное решение каскада (0 = шум, 1 = БПЛА) |
| `preprocess_ms` | float, 3 знака | время DSP (MFCC + log-mel) |
| `time_1d_ms` | float, 3 знака | время **только** инференса 1D-CNN |
| `time_2d_ms` | float, 3 знака | время инференса 2D-CNN (0 без триггера) |
| `e2e_ms` | float, 3 знака | общее время `predict()` на чанк |

## Архитектура (кратко)

- **`Config`** — все параметры DSP и каскада одним `@dataclass`.
- **`DataPreprocessor`** — один STFT → MFCC + log-mel. Все буферы преаллоцированы, hot-loop не аллоцирует.
- **`CascadeDetector`** — 1D-CNN → (опционально) 2D-CNN → решение.
- **`AudioSource`** (ABC) — интерфейс источника секундных чанков. Реализации: `LabeledDirectorySource` (датасет), `LongRecordingSource` (длинный wav стримом), `SerialMicSource` (онлайн с МК).

Добавить новый источник: унаследовать от `AudioSource`, реализовать `__iter__`, возвращающий `(chunk: np.float32[chunk_samples], metadata: dict)`.
