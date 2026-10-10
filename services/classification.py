"""Load the trained intent classifier and run inference."""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


MAX_SEQUENCE_LENGTH = 64
EMBEDDING_SIZE = 128
ARCHITECTURES = ("fasttext", "textcnn", "bilstm")
TOKEN_PATTERN = re.compile(r"[a-z]+(?:'[a-z]+)?|[0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.casefold())


def encode_text(text: str, vocabulary: dict[str, int]) -> tuple[list[int], int]:
    tokens = tokenize(text)[:MAX_SEQUENCE_LENGTH]
    ids = [vocabulary.get(token, 1) for token in tokens]
    length = max(1, len(ids))
    if not ids:
        ids = [1]
    ids.extend([0] * (MAX_SEQUENCE_LENGTH - len(ids)))
    return ids, length


class FastTextClassifier(nn.Module):
    def __init__(self, vocab_size: int, class_count: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBEDDING_SIZE, padding_idx=0)
        self.classifier = nn.Sequential(
            nn.Linear(EMBEDDING_SIZE, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, class_count),
        )

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        mask = input_ids.ne(0).unsqueeze(-1)
        pooled = (embedded * mask).sum(dim=1) / lengths.to(embedded.device).clamp_min(1).unsqueeze(1)
        return self.classifier(pooled)


class TextCNNClassifier(nn.Module):
    def __init__(self, vocab_size: int, class_count: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBEDDING_SIZE, padding_idx=0)
        self.convolutions = nn.ModuleList(
            nn.Conv1d(EMBEDDING_SIZE, 128, kernel_size=size) for size in (2, 3, 4)
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 3, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, class_count),
        )

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids).transpose(1, 2)
        pooled = [
            torch.relu(convolution(embedded)).amax(dim=2)
            for convolution in self.convolutions
        ]
        return self.classifier(torch.cat(pooled, dim=1))


class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size: int, class_count: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBEDDING_SIZE, padding_idx=0)
        self.encoder = nn.LSTM(
            EMBEDDING_SIZE,
            hidden_size=96,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(192, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, class_count),
        )

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        packed = pack_padded_sequence(
            embedded,
            lengths.detach().cpu().clamp_min(1),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (hidden, _) = self.encoder(packed)
        representation = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.classifier(representation)


MODEL_TYPES: dict[str, type[nn.Module]] = {
    "fasttext": FastTextClassifier,
    "textcnn": TextCNNClassifier,
    "bilstm": BiLSTMClassifier,
}


def create_model(name: str, vocab_size: int, class_count: int) -> nn.Module:
    try:
        model_type = MODEL_TYPES[name]
    except KeyError as error:
        raise ValueError(f"Unknown architecture {name!r}; choose from {ARCHITECTURES}") from error
    return model_type(vocab_size, class_count)


def load_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device | None = None,
) -> tuple[nn.Module, dict[str, Any], torch.device]:
    selected_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(
        checkpoint_path,
        map_location=selected_device,
        weights_only=True,
    )
    model = create_model(
        checkpoint["architecture"],
        checkpoint["model_config"]["vocab_size"],
        checkpoint["model_config"]["class_count"],
    ).to(selected_device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint, selected_device


def predict(
    model: nn.Module,
    checkpoint: dict[str, Any],
    text: str,
    device: torch.device,
) -> dict[str, Any]:
    ids, length = encode_text(text, checkpoint["vocabulary"])
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    lengths = torch.tensor([length], dtype=torch.long)
    with torch.inference_mode():
        probabilities = torch.softmax(model(input_ids, lengths), dim=1)[0]
    class_id = int(probabilities.argmax().cpu())
    intent = checkpoint["intent_labels"][class_id]
    return {
        "category": checkpoint["intent_to_category"][intent],
        "intent": intent,
        "confidence": float(probabilities[class_id].cpu()),
    }


class ClassificationService:
    def __init__(self) -> None:
        checkpoint_path = Path(
            os.environ.get(
                "CLASSIFIER_CHECKPOINT",
                str(
                    Path(__file__).resolve().parents[1]
                    / "artifacts"
                    / "classifier"
                    / "models"
                    / "classifier_v1.pt"
                ),
            )
        ).resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Classifier checkpoint not found at {checkpoint_path}")

        self._model, self._checkpoint, self._device = load_checkpoint(
            checkpoint_path,
            torch.device(os.environ.get("CLASSIFIER_DEVICE", "cpu")),
        )
        self._model_name = (
            f"{self._checkpoint.get('model_version', checkpoint_path.stem)} "
            f"({self._checkpoint['architecture'].upper()})"
        )

    def close(self) -> None:
        pass

    def classify(self, text: str) -> dict[str, str | float]:
        started = time.perf_counter()
        prediction = predict(self._model, self._checkpoint, text, self._device)
        duration_ms = (time.perf_counter() - started) * 1000
        return {
            "category": prediction["category"],
            "intent": prediction["intent"],
            "confidence": prediction["confidence"],
            "model": self._model_name,
            "duration_ms": round(duration_ms, 3),
        }
