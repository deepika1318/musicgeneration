export type PythonFile = {
  name: string;
  language: "python" | "text" | "markdown";
  purpose: string;
  code: string;
};

const preprocess = String.raw`"""
preprocess.py
-------------
Reads every MIDI file in data/midi/, extracts notes and chords with music21,
builds a vocabulary, and turns the corpus into fixed-length training sequences
for an LSTM.

Encoding scheme (one token per event):
  * single Note  -> its pitch name with octave, e.g. "C4", "F#5"
  * Chord        -> normal-order pitch classes joined by dots, e.g. "4.7.11"
  * Rest         -> "REST"
Durations are quantised to a small set and appended: "C4_0.5", "4.7.11_1.0".
That keeps rhythm in the model without exploding the vocabulary.

Run:
    python preprocess.py --data-dir data/midi --sequence-length 64
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np

try:
    from music21 import chord, converter, instrument, note
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "music21 is required. Install dependencies with: pip install -r requirements.txt"
    ) from exc


DURATION_BUCKETS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]


class PreprocessingError(RuntimeError):
    """Raised when the dataset cannot produce usable training data."""


def quantise_duration(value: float) -> float:
    """Snap an arbitrary quarter-length to the nearest supported bucket."""
    if value <= 0:
        return DURATION_BUCKETS[0]
    return min(DURATION_BUCKETS, key=lambda bucket: abs(bucket - value))


def parse_midi_file(path: str, include_rests: bool = True) -> List[str]:
    """Parse one MIDI file into a flat list of note/chord tokens.

    Raises:
        PreprocessingError: if the file is unreadable or contains no pitches.
    """
    try:
        score = converter.parse(path)
    except Exception as exc:  # music21 raises many different parse errors
        raise PreprocessingError("Could not parse " + os.path.basename(path) + ": " + str(exc))

    # Prefer a single instrument part (usually piano) when the file has several.
    try:
        parts = instrument.partitionByInstrument(score)
        elements = parts.parts[0].recurse() if parts else score.flatten().notesAndRests
    except Exception:
        elements = score.flatten().notesAndRests

    tokens: List[str] = []
    previous_offset = 0.0

    for element in elements:
        if isinstance(element, note.Note):
            symbol = str(element.pitch)
        elif isinstance(element, chord.Chord):
            # normalOrder gives a transposition-stable pitch-class signature
            symbol = ".".join(str(pitch_class) for pitch_class in element.normalOrder)
        elif isinstance(element, note.Rest) and include_rests:
            symbol = "REST"
        else:
            continue

        # Gap between events longer than a beat becomes an explicit rest token,
        # which keeps phrasing audible in the generated output.
        gap = float(element.offset) - previous_offset
        if include_rests and gap >= 1.0 and tokens:
            tokens.append("REST_" + str(quantise_duration(gap)))

        duration = quantise_duration(float(element.duration.quarterLength))
        tokens.append(symbol + "_" + str(duration))
        previous_offset = float(element.offset)

    if not tokens:
        raise PreprocessingError(os.path.basename(path) + " contains no notes or chords.")

    return tokens


def load_corpus(data_dir: str) -> Tuple[List[str], List[Dict[str, object]]]:
    """Parse every .mid/.midi file in data_dir. Bad files are skipped, not fatal."""
    patterns = ("*.mid", "*.midi", "*.MID", "*.MIDI")
    paths: List[str] = []
    for pattern in patterns:
        paths.extend(sorted(glob.glob(os.path.join(data_dir, "**", pattern), recursive=True)))
    paths = sorted(set(paths))

    if not paths:
        raise PreprocessingError(
            "No MIDI files found in " + data_dir + ". "
            "Add .mid files there (see README: Dataset setup)."
        )

    corpus: List[str] = []
    report: List[Dict[str, object]] = []

    for path in paths:
        try:
            tokens = parse_midi_file(path)
        except PreprocessingError as exc:
            print("  [skip] " + str(exc))
            report.append({"file": os.path.basename(path), "status": "skipped", "tokens": 0})
            continue
        corpus.extend(tokens)
        report.append({"file": os.path.basename(path), "status": "ok", "tokens": len(tokens)})
        print("  [ok]   " + os.path.basename(path) + " -> " + str(len(tokens)) + " events")

    if not corpus:
        raise PreprocessingError("Every MIDI file failed to parse. Check the dataset.")

    return corpus, report


def build_vocabulary(corpus: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Map each distinct token to an integer id (sorted for reproducibility)."""
    vocabulary = sorted(set(corpus))
    note_to_int = {token: index for index, token in enumerate(vocabulary)}
    int_to_note = {index: token for token, index in note_to_int.items()}
    return note_to_int, int_to_note


def create_sequences(
    corpus: List[str],
    note_to_int: Dict[str, int],
    sequence_length: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    """Slide a window over the corpus to build (X, y) for categorical training.

    X: (samples, sequence_length, 1) normalised to [0, 1]
    y: (samples, vocab_size) one-hot targets for categorical_crossentropy
    """
    vocab_size = len(note_to_int)
    if len(corpus) <= sequence_length:
        raise PreprocessingError(
            "Not enough data: " + str(len(corpus)) + " events for a sequence length of "
            + str(sequence_length) + ". Add more MIDI files or lower --sequence-length."
        )

    network_input: List[List[int]] = []
    network_output: List[int] = []

    for index in range(len(corpus) - sequence_length):
        window = corpus[index : index + sequence_length]
        target = corpus[index + sequence_length]
        network_input.append([note_to_int[token] for token in window])
        network_output.append(note_to_int[target])

    samples = len(network_input)
    X = np.reshape(np.array(network_input, dtype=np.float32), (samples, sequence_length, 1))
    X = X / float(vocab_size)  # normalise ids into [0, 1) for stable LSTM training

    y = np.zeros((samples, vocab_size), dtype=np.float32)
    y[np.arange(samples), np.array(network_output, dtype=np.int64)] = 1.0  # one-hot

    return X, y


def preprocess(data_dir: str, sequence_length: int, out_dir: str = "artifacts") -> Dict[str, object]:
    """Full pipeline: parse -> vocabulary -> sequences -> saved artifacts."""
    os.makedirs(out_dir, exist_ok=True)
    print("Scanning " + data_dir + " ...")
    corpus, report = load_corpus(data_dir)
    note_to_int, int_to_note = build_vocabulary(corpus)
    X, y = create_sequences(corpus, note_to_int, sequence_length)

    with open(os.path.join(out_dir, "corpus.pkl"), "wb") as handle:
        pickle.dump(
            {
                "corpus": corpus,
                "note_to_int": note_to_int,
                "int_to_note": int_to_note,
                "sequence_length": sequence_length,
            },
            handle,
        )

    most_common = Counter(corpus).most_common(10)
    stats = {
        "files": report,
        "total_events": len(corpus),
        "vocab_size": len(note_to_int),
        "sequence_length": sequence_length,
        "training_samples": int(X.shape[0]),
        "most_common": most_common,
    }

    print("")
    print("Events:            " + str(stats["total_events"]))
    print("Vocabulary size:   " + str(stats["vocab_size"]))
    print("Training samples:  " + str(stats["training_samples"]))
    print("Saved artifacts to " + out_dir + "/corpus.pkl")
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess MIDI files for LSTM training.")
    parser.add_argument("--data-dir", default="data/midi")
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--out-dir", default="artifacts")
    args = parser.parse_args()
    preprocess(args.data_dir, args.sequence_length, args.out_dir)
`;

const model = String.raw`"""
model.py
--------
The LSTM architecture. Deliberately small so it trains on a laptop CPU.

    Input (sequence_length, 1)
      -> LSTM(256, return_sequences=True)   captures long-range melodic context
      -> Dropout(0.3)                       regularisation against memorising
      -> LSTM(256)                          collapses the sequence to one state
      -> Dense(128, relu)
      -> Dropout(0.3)
      -> Dense(vocab_size, softmax)         probability for every possible note

Loss: categorical_crossentropy (one-hot targets)
Optimizer: Adam (RMSprop also supported, the classic choice for char/note RNNs)
"""

from __future__ import annotations

from tensorflow import keras
from tensorflow.keras import layers


def build_model(
    sequence_length: int,
    vocab_size: int,
    lstm_units: int = 256,
    dropout: float = 0.3,
    optimizer_name: str = "adam",
    learning_rate: float = 0.001,
) -> keras.Model:
    """Create and compile the note-prediction LSTM."""
    if vocab_size < 2:
        raise ValueError("Vocabulary must contain at least 2 tokens to train a classifier.")

    model = keras.Sequential(
        [
            keras.Input(shape=(sequence_length, 1), name="note_window"),
            layers.LSTM(lstm_units, return_sequences=True, name="lstm_1"),
            layers.Dropout(dropout),
            layers.LSTM(lstm_units, name="lstm_2"),
            layers.Dropout(dropout),
            layers.Dense(128, activation="relu", name="dense_hidden"),
            layers.Dropout(dropout),
            layers.Dense(vocab_size, activation="softmax", name="note_probabilities"),
        ],
        name="lstm_music_generator",
    )

    if optimizer_name.lower() == "rmsprop":
        optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model


if __name__ == "__main__":
    build_model(sequence_length=64, vocab_size=200).summary()
`;

const train = String.raw`"""
train.py
--------
Trains the LSTM on preprocessed sequences and writes artifacts/model.keras.

Run:
    python train.py --data-dir data/midi --epochs 30 --batch-size 64 --sequence-length 64
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from typing import Callable, Dict, Optional

import numpy as np
from tensorflow import keras

from model import build_model
from preprocess import PreprocessingError, build_vocabulary, create_sequences, load_corpus


class ProgressCallback(keras.callbacks.Callback):
    """Forwards per-epoch metrics to any listener (the API server uses this)."""

    def __init__(self, total_epochs: int, on_epoch: Optional[Callable[[Dict], None]] = None):
        super().__init__()
        self.total_epochs = total_epochs
        self.on_epoch = on_epoch

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        payload = {
            "epoch": epoch + 1,
            "total_epochs": self.total_epochs,
            "loss": float(logs.get("loss", 0.0)),
            "accuracy": float(logs.get("accuracy", 0.0)),
            "val_loss": float(logs.get("val_loss", 0.0)) if "val_loss" in logs else None,
            "val_accuracy": float(logs.get("val_accuracy", 0.0)) if "val_accuracy" in logs else None,
        }
        if self.on_epoch:
            self.on_epoch(payload)


def train(
    data_dir: str = "data/midi",
    epochs: int = 30,
    batch_size: int = 64,
    sequence_length: int = 64,
    lstm_units: int = 256,
    dropout: float = 0.3,
    optimizer_name: str = "adam",
    validation_split: float = 0.1,
    out_dir: str = "artifacts",
    on_epoch: Optional[Callable[[Dict], None]] = None,
) -> Dict:
    """Preprocess, fit, and persist the model plus its vocabulary."""
    os.makedirs(out_dir, exist_ok=True)

    corpus, file_report = load_corpus(data_dir)
    note_to_int, int_to_note = build_vocabulary(corpus)
    X, y = create_sequences(corpus, note_to_int, sequence_length)

    if X.shape[0] < batch_size:
        raise PreprocessingError(
            "Only " + str(X.shape[0]) + " training samples, which is fewer than the batch size ("
            + str(batch_size) + "). Add more MIDI files or reduce the batch size."
        )

    model = build_model(
        sequence_length=sequence_length,
        vocab_size=len(note_to_int),
        lstm_units=lstm_units,
        dropout=dropout,
        optimizer_name=optimizer_name,
    )

    callbacks = [
        ProgressCallback(epochs, on_epoch),
        keras.callbacks.ModelCheckpoint(
            os.path.join(out_dir, "model.keras"), monitor="loss", save_best_only=True
        ),
        keras.callbacks.EarlyStopping(monitor="loss", patience=8, restore_best_weights=True),
    ]

    history = model.fit(
        X,
        y,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split if X.shape[0] > 20 else 0.0,
        callbacks=callbacks,
        verbose=2,
    )

    model.save(os.path.join(out_dir, "model.keras"))
    with open(os.path.join(out_dir, "vocabulary.pkl"), "wb") as handle:
        pickle.dump(
            {
                "note_to_int": note_to_int,
                "int_to_note": int_to_note,
                "sequence_length": sequence_length,
                "corpus": corpus,
            },
            handle,
        )

    summary = {
        "files": file_report,
        "epochs_run": len(history.history.get("loss", [])),
        "final_loss": float(history.history["loss"][-1]),
        "final_accuracy": float(history.history["accuracy"][-1]),
        "vocab_size": len(note_to_int),
        "training_samples": int(X.shape[0]),
        "sequence_length": sequence_length,
        "history": {key: [float(v) for v in values] for key, values in history.history.items()},
    }

    with open(os.path.join(out_dir, "training_summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)

    print("")
    print("Final loss:     " + str(round(summary["final_loss"], 4)))
    print("Final accuracy: " + str(round(summary["final_accuracy"], 4)))
    print("Model saved to  " + os.path.join(out_dir, "model.keras"))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the LSTM music generator.")
    parser.add_argument("--data-dir", default="data/midi")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--lstm-units", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--optimizer", default="adam", choices=["adam", "rmsprop"])
    args = parser.parse_args()

    try:
        train(
            data_dir=args.data_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
            lstm_units=args.lstm_units,
            dropout=args.dropout,
            optimizer_name=args.optimizer,
        )
    except PreprocessingError as exc:
        raise SystemExit("Dataset error: " + str(exc))
`;

const generate = String.raw`"""
generate.py
-----------
Autoregressive sampling from the trained LSTM, written back out as a MIDI file.

Run:
    python generate.py --length 200 --temperature 1.0 --output output/generated.mid
"""

from __future__ import annotations

import argparse
import os
import pickle
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
from music21 import chord, instrument, note, stream
from tensorflow import keras


class GenerationError(RuntimeError):
    """Raised when generation cannot run (no model, no vocabulary, bad seed)."""


def load_artifacts(out_dir: str = "artifacts") -> Tuple[keras.Model, Dict]:
    model_path = os.path.join(out_dir, "model.keras")
    vocab_path = os.path.join(out_dir, "vocabulary.pkl")
    if not os.path.exists(model_path) or not os.path.exists(vocab_path):
        raise GenerationError("No trained model found. Run train.py first.")
    model = keras.models.load_model(model_path)
    with open(vocab_path, "rb") as handle:
        vocabulary = pickle.load(handle)
    return model, vocabulary


def sample_with_temperature(probabilities: np.ndarray, temperature: float) -> int:
    """Temperature sampling. temperature <= 0 falls back to argmax (most likely note)."""
    if temperature <= 0:
        return int(np.argmax(probabilities))
    probabilities = np.asarray(probabilities).astype("float64")
    probabilities = np.log(np.maximum(probabilities, 1e-10)) / temperature
    exponentiated = np.exp(probabilities)
    probabilities = exponentiated / np.sum(exponentiated)
    return int(np.random.choice(len(probabilities), p=probabilities))


def generate_tokens(
    model: keras.Model,
    vocabulary: Dict,
    length: int = 200,
    temperature: float = 1.0,
    seed_index: Optional[int] = None,
) -> Tuple[List[str], List[str]]:
    """Slide a window forward, predicting one token at a time."""
    corpus: List[str] = vocabulary["corpus"]
    note_to_int: Dict[str, int] = vocabulary["note_to_int"]
    int_to_note: Dict[int, str] = vocabulary["int_to_note"]
    sequence_length: int = vocabulary["sequence_length"]
    vocab_size = len(note_to_int)

    if len(corpus) <= sequence_length:
        raise GenerationError("Corpus is shorter than the model's sequence length.")

    if seed_index is None:
        seed_index = random.randint(0, len(corpus) - sequence_length - 1)
    seed_index = max(0, min(seed_index, len(corpus) - sequence_length - 1))

    seed_tokens = corpus[seed_index : seed_index + sequence_length]
    window = [note_to_int[token] for token in seed_tokens]
    generated: List[str] = []

    for _ in range(length):
        model_input = np.reshape(np.array(window, dtype=np.float32), (1, sequence_length, 1))
        model_input = model_input / float(vocab_size)
        probabilities = model.predict(model_input, verbose=0)[0]
        next_index = sample_with_temperature(probabilities, temperature)
        generated.append(int_to_note[next_index])
        window.append(next_index)
        window = window[1:]  # slide the window forward by one step

    return generated, seed_tokens


def tokens_to_stream(tokens: List[str]) -> stream.Stream:
    """Turn "C4_0.5" / "4.7.11_1.0" / "REST_1.0" tokens back into a music21 stream."""
    output_notes = []
    offset = 0.0

    for token in tokens:
        symbol, _, duration_text = token.partition("_")
        try:
            duration = float(duration_text) if duration_text else 0.5
        except ValueError:
            duration = 0.5

        if symbol == "REST":
            offset += duration
            continue

        if "." in symbol or symbol.isdigit():
            pitches = []
            for part in symbol.split("."):
                try:
                    new_note = note.Note(int(part))
                except ValueError:
                    continue
                new_note.storedInstrument = instrument.Piano()
                pitches.append(new_note)
            if not pitches:
                offset += duration
                continue
            new_chord = chord.Chord(pitches)
            new_chord.offset = offset
            new_chord.quarterLength = duration
            output_notes.append(new_chord)
        else:
            try:
                new_note = note.Note(symbol)
            except Exception:
                offset += duration
                continue
            new_note.offset = offset
            new_note.quarterLength = duration
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        offset += duration

    if not output_notes:
        raise GenerationError("The generated sequence contained no playable notes.")

    return stream.Stream(output_notes)


def generate(
    length: int = 200,
    temperature: float = 1.0,
    seed_index: Optional[int] = None,
    out_dir: str = "artifacts",
    output_path: str = "output/generated.mid",
) -> Dict:
    model, vocabulary = load_artifacts(out_dir)
    tokens, seed_tokens = generate_tokens(model, vocabulary, length, temperature, seed_index)
    midi_stream = tokens_to_stream(tokens)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    midi_stream.write("midi", fp=output_path)

    unique_tokens = sorted(set(tokens))
    result = {
        "output_path": output_path,
        "tokens": tokens,
        "seed_tokens": seed_tokens,
        "length": len(tokens),
        "unique_tokens": len(unique_tokens),
        "temperature": temperature,
        "chord_count": sum(1 for token in tokens if "." in token.split("_")[0]),
        "rest_count": sum(1 for token in tokens if token.startswith("REST")),
    }
    print("Wrote " + output_path + " (" + str(len(tokens)) + " events, "
          + str(len(unique_tokens)) + " unique)")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate music with the trained LSTM.")
    parser.add_argument("--length", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed-index", type=int, default=None)
    parser.add_argument("--output", default="output/generated.mid")
    args = parser.parse_args()

    try:
        generate(
            length=args.length,
            temperature=args.temperature,
            seed_index=args.seed_index,
            output_path=args.output,
        )
    except GenerationError as exc:
        raise SystemExit("Generation error: " + str(exc))
`;

const server = String.raw`"""
server.py
---------
FastAPI wrapper so the web interface can drive real training and generation
on your machine.

Run:
    uvicorn server:app --reload --port 8000

Endpoints:
    GET  /health              -> API alive + model/dataset status
    GET  /dataset             -> MIDI files found in data/midi and their event counts
    POST /train               -> start training in a background thread
    GET  /train/status        -> live epoch metrics while training runs
    POST /generate            -> generate a sequence and write a MIDI file
    GET  /download/{filename} -> download a generated MIDI file
"""

from __future__ import annotations

import os
import threading
import time
import traceback
import uuid
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from generate import GenerationError, generate
from preprocess import PreprocessingError, load_corpus
from train import train

DATA_DIR = os.environ.get("MIDI_DATA_DIR", "data/midi")
ARTIFACT_DIR = os.environ.get("ARTIFACT_DIR", "artifacts")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "output")

app = FastAPI(title="LSTM Music Generator API", version="1.0.0")

# The web UI runs on a different origin, so allow browser calls from anywhere.
# Tighten this list if you deploy the API beyond localhost.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATE: Dict[str, object] = {
    "status": "idle",  # idle | training | done | error
    "epochs": [],
    "message": "",
    "summary": None,
    "started_at": None,
}
LOCK = threading.Lock()


class TrainRequest(BaseModel):
    epochs: int = Field(30, ge=1, le=500)
    batch_size: int = Field(64, ge=1, le=512)
    sequence_length: int = Field(64, ge=8, le=256)
    lstm_units: int = Field(256, ge=32, le=512)
    dropout: float = Field(0.3, ge=0.0, le=0.8)
    optimizer: str = Field("adam")


class GenerateRequest(BaseModel):
    length: int = Field(200, ge=8, le=2000)
    temperature: float = Field(1.0, ge=0.0, le=2.0)
    seed_index: Optional[int] = None


def model_is_trained() -> bool:
    return os.path.exists(os.path.join(ARTIFACT_DIR, "model.keras")) and os.path.exists(
        os.path.join(ARTIFACT_DIR, "vocabulary.pkl")
    )


@app.get("/health")
def health() -> Dict[str, object]:
    return {
        "ok": True,
        "data_dir": DATA_DIR,
        "model_trained": model_is_trained(),
        "training_status": STATE["status"],
    }


@app.get("/dataset")
def dataset() -> Dict[str, object]:
    try:
        corpus, report = load_corpus(DATA_DIR)
    except PreprocessingError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"files": report, "total_events": len(corpus), "data_dir": DATA_DIR}


def _run_training(request: TrainRequest) -> None:
    def on_epoch(payload: Dict) -> None:
        with LOCK:
            STATE["epochs"].append(payload)

    try:
        summary = train(
            data_dir=DATA_DIR,
            epochs=request.epochs,
            batch_size=request.batch_size,
            sequence_length=request.sequence_length,
            lstm_units=request.lstm_units,
            dropout=request.dropout,
            optimizer_name=request.optimizer,
            out_dir=ARTIFACT_DIR,
            on_epoch=on_epoch,
        )
        with LOCK:
            STATE["status"] = "done"
            STATE["summary"] = summary
            STATE["message"] = "Training finished."
    except (PreprocessingError, Exception) as exc:  # surface the real reason to the UI
        traceback.print_exc()
        with LOCK:
            STATE["status"] = "error"
            STATE["message"] = str(exc)


@app.post("/train")
def start_training(request: TrainRequest) -> Dict[str, object]:
    with LOCK:
        if STATE["status"] == "training":
            raise HTTPException(status_code=409, detail="Training is already running.")
        STATE.update(
            {"status": "training", "epochs": [], "message": "", "summary": None,
             "started_at": time.time()}
        )
    threading.Thread(target=_run_training, args=(request,), daemon=True).start()
    return {"started": True, "config": request.model_dump()}


@app.get("/train/status")
def training_status() -> Dict[str, object]:
    with LOCK:
        epochs: List[Dict] = list(STATE["epochs"])
        return {
            "status": STATE["status"],
            "message": STATE["message"],
            "epochs": epochs,
            "summary": STATE["summary"],
            "elapsed": (time.time() - STATE["started_at"]) if STATE["started_at"] else 0,
            "model_trained": model_is_trained(),
        }


@app.post("/generate")
def generate_music(request: GenerateRequest) -> Dict[str, object]:
    if not model_is_trained():
        raise HTTPException(status_code=409, detail="Model is not trained yet. Train it first.")
    filename = "generated_" + uuid.uuid4().hex[:8] + ".mid"
    output_path = os.path.join(OUTPUT_DIR, filename)
    try:
        result = generate(
            length=request.length,
            temperature=request.temperature,
            seed_index=request.seed_index,
            out_dir=ARTIFACT_DIR,
            output_path=output_path,
        )
    except GenerationError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Generation failed: " + str(exc))

    result["filename"] = filename
    result["download_url"] = "/download/" + filename
    return result


@app.get("/download/{filename}")
def download(filename: str) -> FileResponse:
    safe_name = os.path.basename(filename)
    path = os.path.join(OUTPUT_DIR, safe_name)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(path, media_type="audio/midi", filename=safe_name)
`;

const requirements = String.raw`# Python 3.9 - 3.11 recommended
tensorflow==2.16.1
music21==9.1.0
numpy==1.26.4
fastapi==0.111.0
uvicorn[standard]==0.30.1
pydantic==2.7.1
`;

const readme = String.raw`# AI Music Generator using LSTM

Train an LSTM on classical piano MIDI files and generate new music.
Pure Python + TensorFlow/Keras + music21, with an optional web UI on top.

## 1. Setup

    python -m venv .venv
    source .venv/bin/activate        # Windows: .venv\Scripts\activate
    pip install -r requirements.txt

Project layout:

    .
    |- data/midi/          <- put your .mid files here
    |- artifacts/          <- model.keras, vocabulary.pkl (created by training)
    |- output/             <- generated .mid files
    |- preprocess.py
    |- model.py
    |- train.py
    |- generate.py
    |- server.py
    |- requirements.txt

## 2. Dataset

Create the folder and drop MIDI files in it:

    mkdir -p data/midi

Good public sources (all free, piano-heavy, small enough for a laptop):

* Classical Piano MIDI Page - http://www.piano-midi.de  (Chopin, Bach, Mozart, Beethoven)
* MAESTRO dataset - https://magenta.tensorflow.org/datasets/maestro  (large; take a subset)
* The Lakh MIDI Dataset - https://colinraffel.com/projects/lmd/
* Video-game piano MIDIs (the classic "Final Fantasy" set used in most LSTM music tutorials)

Start with **10-30 files by a single composer**. One composer keeps the style
consistent, and 10-30 files is enough for the model to learn phrasing without
needing a GPU.

## 3. Train

    python train.py --data-dir data/midi --epochs 30 --batch-size 64 --sequence-length 64

Useful flags: --lstm-units 256, --dropout 0.3, --optimizer adam|rmsprop

Expect roughly 1-3 minutes per epoch on a laptop CPU with ~20 MIDI files.
Checkpoints go to artifacts/model.keras; the run summary is written to
artifacts/training_summary.json.

## 4. Generate

    python generate.py --length 200 --temperature 1.0 --output output/generated.mid

* temperature 0.0  -> argmax, safest and most repetitive
* temperature 0.7  -> coherent with some variation (recommended)
* temperature 1.2+ -> adventurous, can wander out of key

Open the resulting .mid in any DAW, MuseScore, or a browser MIDI player.

## 5. Run the API for the web UI

    uvicorn server:app --reload --port 8000

Then open the web interface, set the backend URL to http://localhost:8000 and
press Connect. The UI will drive real training and generation on your machine
and let you download the generated MIDI.

## 6. Troubleshooting

| Problem | Fix |
| --- | --- |
| "No MIDI files found" | Files must be .mid/.midi inside data/midi (subfolders are scanned too). |
| "Could not parse X" | That one file is corrupt; it is skipped automatically. |
| "Not enough data" | Add more MIDI files or lower --sequence-length. |
| "No trained model found" | Run train.py before generate.py. |
| TensorFlow install fails | Use Python 3.9-3.11; 3.12+ wheels are not always available. |
| Browser cannot reach the API | Make sure uvicorn is running and the URL includes http:// and the port. |
`;

export const PYTHON_FILES: PythonFile[] = [
  {
    name: "preprocess.py",
    language: "python",
    purpose: "MIDI parsing, note/chord extraction, vocabulary, sequence windows",
    code: preprocess,
  },
  {
    name: "model.py",
    language: "python",
    purpose: "The Keras LSTM architecture and compilation",
    code: model,
  },
  {
    name: "train.py",
    language: "python",
    purpose: "Training loop, checkpoints, per-epoch metrics",
    code: train,
  },
  {
    name: "generate.py",
    language: "python",
    purpose: "Temperature sampling and MIDI export via music21",
    code: generate,
  },
  {
    name: "server.py",
    language: "python",
    purpose: "FastAPI endpoints the web interface talks to",
    code: server,
  },
  {
    name: "requirements.txt",
    language: "text",
    purpose: "Pinned Python dependencies",
    code: requirements,
  },
  {
    name: "README.md",
    language: "markdown",
    purpose: "Dataset setup and exact local run instructions",
    code: readme,
  },
];
