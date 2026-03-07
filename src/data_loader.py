"""PyTorch dataset utilities for emotion recognition pickles."""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

TensorTransform = Callable[[torch.Tensor], torch.Tensor]


def _ensure_numpy(array_like: np.ndarray | Sequence[float]) -> np.ndarray:
    """Return a contiguous numpy array regardless of source type."""
    if isinstance(array_like, np.ndarray):
        return array_like
    return np.asarray(array_like)


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    label_tensor: torch.Tensor
    vision_path: Path
    audio_path: Path


class EmotionDataset(Dataset):
    """Loads paired ViT + wav2vec2 activations with emotion targets."""

    def __init__(
        self,
        csv_path: str | Path,
        vision_dir: str | Path,
        audio_dir: str | Path,
        label_columns: Optional[Sequence[str]] = None,
        vision_transform: Optional[TensorTransform] = None,
        audio_transform: Optional[TensorTransform] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.vision_dir = Path(vision_dir)
        self.audio_dir = Path(audio_dir)
        self.vision_transform = vision_transform
        self.audio_transform = audio_transform
        self.dtype = dtype

        df = pd.read_csv(self.csv_path, dtype={"Filename": str})
        if label_columns is None:
            label_columns = [c for c in df.columns if c != "Filename"]
        self.label_columns = list(label_columns)

        filenames = df["Filename"].astype(str).tolist()
        label_matrix = torch.tensor(
            df[self.label_columns].to_numpy(),
            dtype=self.dtype,
        )

        records: List[SampleRecord] = []
        for sample_id, labels in zip(filenames, label_matrix):
            vision_path = self.vision_dir / f"{sample_id}.pkl"
            audio_path = self.audio_dir / f"{sample_id}.pkl"
            if not vision_path.exists():
                raise FileNotFoundError(f"Missing ViT pickle for {sample_id}: {vision_path}")
            if not audio_path.exists():
                raise FileNotFoundError(f"Missing wav2vec2 pickle for {sample_id}: {audio_path}")
            records.append(SampleRecord(sample_id, labels, vision_path, audio_path))

        self.records = records

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.records)

    def _load_pickle_tensor(self, path: Path) -> torch.Tensor:
        with open(path, "rb") as f:
            arr = pickle.load(f)
        np_arr = _ensure_numpy(arr)
        tensor = torch.from_numpy(np_arr).to(self.dtype)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return tensor

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        record = self.records[index]
        vision = self._load_pickle_tensor(record.vision_path)
        audio = self._load_pickle_tensor(record.audio_path)

        if self.vision_transform:
            vision = self.vision_transform(vision)
        if self.audio_transform:
            audio = self.audio_transform(audio)

        vision_mask = torch.ones(vision.size(0), dtype=torch.bool)
        audio_mask = torch.ones(audio.size(0), dtype=torch.bool)

        return {
            "id": record.sample_id,
            "vision": vision,
            "vision_mask": vision_mask,
            "audio": audio,
            "audio_mask": audio_mask,
            "labels": record.label_tensor,
        }


def multimodal_collate(batch: Sequence[Dict[str, torch.Tensor | str]]) -> Dict[str, torch.Tensor | List[str]]:
    """Pads variable-length sequences and stacks tensors."""

    def _pad(key: str) -> Tuple[torch.Tensor, torch.Tensor]:
        seqs = [item[key] for item in batch]
        if not all(isinstance(seq, torch.Tensor) for seq in seqs):
            raise TypeError(f"Expected tensors for key '{key}'")
        lengths = torch.tensor([seq.size(0) for seq in seqs], dtype=torch.long)
        padded = pad_sequence(seqs, batch_first=True)
        mask = torch.arange(padded.size(1))[None, :] < lengths[:, None]
        return padded, mask

    vision, vision_mask = _pad("vision")
    audio, audio_mask = _pad("audio")
    labels = torch.stack([item["labels"] for item in batch])
    ids = [item["id"] for item in batch]

    return {
        "ids": ids,
        "vision": vision,
        "vision_mask": vision_mask,
        "audio": audio,
        "audio_mask": audio_mask,
        "labels": labels,
    }


def create_dataloader(
    csv_path: str | Path,
    vision_dir: str | Path,
    audio_dir: str | Path,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    **dataset_kwargs,
) -> DataLoader:
    dataset = EmotionDataset(csv_path, vision_dir, audio_dir, **dataset_kwargs)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=multimodal_collate,
    )


__all__ = [
    "EmotionDataset",
    "multimodal_collate",
    "create_dataloader",
]