"""
Staged Multimodal Fusion Training Framework

This module implements a four-stage training pipeline for multimodal emotion recognition:
- Stage 1: TextMLPEncoder  (trained on text embeddings)
- Stage 2: AudioAttentionEncoder (trained on audio embeddings)
- Stage 3: VisionAttentionEncoder (trained on visual embeddings)
- Stage 4: FusionRegressor (trained on frozen encoder outputs + optional motion)

Each stage is independently trained with early stopping and learning rate scheduling.
Motion feature support is optional and can be added as a 5th encoder stage.

Author: [Your Name]
Date: 2025
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


# ============================================================================
# Logging Setup
# ============================================================================

def setup_logging(log_dir: str) -> str:
    """Initialize logging to file and stdout."""
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"staged_fusion_{ts}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, mode="a"),
        ],
    )
    return log_path


def log(msg: str) -> None:
    """Log a message to configured loggers."""
    logging.info(msg)


# ============================================================================
# Constants
# ============================================================================

LABEL_COLUMNS = ["Admiration", "Amusement", "Determination",
                 "Empathic Pain", "Excitement", "Joy"]

MOTION_FEATURE_COLUMNS = [
    "motion_mag_mean",
    "motion_mag_std",
    "motion_mag_max",
    "motion_mag_temporal_std",
    "motion_energy",
    "motion_angle_mean",
    "motion_angle_std",
    "motion_burst_ratio",
]


# ============================================================================
# Dataset
# ============================================================================

class StagedDataset(Dataset):
    """
    Loads multimodal embeddings (face/audio/text) + optional motion vectors.
    
    Expected structure:
    - face_dir/: {video_id}.pkl files containing visual embeddings (T, D_v)
    - audio_dir/: {video_id}.pkl files containing audio embeddings (T, D_a)
    - text_dir/: {video_id}.pkl files containing text embeddings (D_t,)
    - motion_seq_dir/: {video_id}.pkl files containing motion sequences (T-1, D_m)
    """

    @staticmethod
    def _norm_id(v) -> str:
        """Normalize video ID to consistent format."""
        v = str(v).strip()
        if v.endswith(".0"):
            v = v[:-2]
        if v.isdigit():
            v = v.zfill(5)
        return v

    def __init__(
        self,
        split_csv: str,
        face_dir: str,
        audio_dir: str,
        text_dir: str,
        label_columns: List[str] = LABEL_COLUMNS,
        load_face: bool = True,
        load_audio: bool = True,
        load_text: bool = True,
        text_input_dim: int = 768,
        use_motion_seq: bool = False,
        motion_seq_dir: Optional[str] = None,
    ):
        self.face_dir = Path(face_dir)
        self.audio_dir = Path(audio_dir)
        self.text_dir = Path(text_dir)
        self.load_face = load_face
        self.load_audio = load_audio
        self.load_text = load_text
        self.text_input_dim = text_input_dim
        self.label_columns = label_columns

        df = pd.read_csv(split_csv, dtype={"Filename": str})

        # Motion sequence features
        self.use_motion_seq = use_motion_seq
        self.motion_seq_dir = Path(motion_seq_dir) if motion_seq_dir else None
        self.motion_feat_dim = 23
 
        if use_motion_seq and motion_seq_dir is None:
            raise ValueError("use_motion_seq=True but motion_seq_dir not provided")

        # Validate samples and track missing data
        valid, skipped, missing_text = [], [], []
        for _, row in df.iterrows():
            fid = self._norm_id(row["Filename"])
            has_face  = (self.face_dir  / f"{fid}.pkl").exists()
            has_audio = (self.audio_dir / f"{fid}.pkl").exists()
            has_text  = (self.text_dir  / f"{fid}.pkl").exists()

            need_face  = load_face
            need_audio = load_audio

            if (not need_face or has_face) and (not need_audio or has_audio):
                valid.append(row)
                if load_text and not has_text:
                    missing_text.append(fid)
            else:
                skipped.append(fid)

        self.samples = pd.DataFrame(valid).reset_index(drop=True)
        self.missing_text_ids = set(missing_text)

        if skipped:
            log(f"⚠ Skipped {len(skipped)} samples missing required pkl. "
                f"Example: {skipped[:3]}")
        if missing_text:
            log(f"⚠ Zero text embeddings for {len(missing_text)} samples. "
                f"Using zero vectors. Example: {missing_text[:3]}")
        if use_motion_seq and self.motion_seq_dir is not None:
            missing_motion_seq = [
                self._norm_id(r["Filename"])
                for _, r in self.samples.iterrows()
                if not (self.motion_seq_dir / f"{self._norm_id(r['Filename'])}.pkl").exists()
            ]
            if missing_motion_seq:
                log(f"⚠ Missing motion seq pkls for {len(missing_motion_seq)} samples. "
                    f"Example: {missing_motion_seq[:3]}")

    @staticmethod
    def _load_motion(motion_csv: str) -> Tuple[Dict[str, np.ndarray], int]:
        """Load motion features from CSV."""
        mdf = pd.read_csv(motion_csv)
        id_col = next(
            (c for c in ["Filename", "filename", "video_id", "id"]
             if c in mdf.columns), None
        )
        missing_cols = [c for c in MOTION_FEATURE_COLUMNS if c not in mdf.columns]
        if missing_cols:
            raise ValueError(
                f"Motion CSV is missing required columns: {missing_cols}. "
                f"Available columns: {list(mdf.columns)}"
            )
        feat_cols = MOTION_FEATURE_COLUMNS
        motion_map: Dict[str, np.ndarray] = {}
        for idx, row in mdf.iterrows():
            key = StagedDataset._norm_id(row[id_col] if id_col else idx)
            vals = pd.to_numeric(row[feat_cols], errors="coerce").fillna(0.0).to_numpy(np.float32)
            motion_map[key] = vals
        return motion_map, len(feat_cols)

    def __len__(self) -> int:
        return len(self.samples)

    def _load_pkl(self, path: Path) -> torch.Tensor:
        """Load pickle file and return as float32 tensor."""
        with open(path, "rb") as f:
            arr = pickle.load(f)
        if isinstance(arr, np.ndarray):
            t = torch.from_numpy(arr).float()
        else:
            t = torch.tensor(arr, dtype=torch.float32)
        if t.dim() == 1:
            t = t.unsqueeze(0)
        return t

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.samples.iloc[idx]
        fid = self._norm_id(row["Filename"])
        out: Dict[str, torch.Tensor] = {}

        out["labels"] = torch.tensor(
            [row[c] for c in self.label_columns], dtype=torch.float32
        )

        if self.load_face:
            face = self._load_pkl(self.face_dir / f"{fid}.pkl")
            out["vision"] = face
            out["vision_mask"] = torch.ones(face.size(0), dtype=torch.bool)

        if self.load_audio:
            audio = self._load_pkl(self.audio_dir / f"{fid}.pkl")
            out["audio"] = audio
            out["audio_mask"] = torch.ones(audio.size(0), dtype=torch.bool)

        if self.load_text:
            if fid in self.missing_text_ids:
                text = torch.zeros(self.text_input_dim, dtype=torch.float32)
            else:
                raw = self._load_pkl(self.text_dir / f"{fid}.pkl")
                text = raw.reshape(-1)
            # L2 normalization (matches standard practice)
            norm = text.norm().clamp(min=1e-12)
            out["text"] = text / norm

        if self.use_motion_seq:
            seq_path = self.motion_seq_dir / f"{fid}.pkl"
            if seq_path.exists():
                motion_seq = self._load_pkl(seq_path)  # (T-1, D_m)
            else:
                # fallback: single zero frame
                motion_seq = torch.zeros((1, self.motion_feat_dim),
                                         dtype=torch.float32)
            out["motion_seq"] = motion_seq
            out["motion_seq_mask"] = torch.ones(
                motion_seq.size(0), dtype=torch.bool
            )

        return out


def staged_collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate function: pad sequences, stack labels, create masks."""
    labels = torch.stack([b["labels"] for b in batch])
    out: Dict[str, torch.Tensor] = {"labels": labels}

    for seq_key, mask_key in [("vision", "vision_mask"), ("audio", "audio_mask")]:
        if seq_key in batch[0]:
            seqs = [b[seq_key] for b in batch]
            lengths = torch.tensor([s.size(0) for s in seqs])
            padded = pad_sequence(seqs, batch_first=True)
            mask = torch.arange(padded.size(1))[None, :] < lengths[:, None]
            out[seq_key] = padded
            out[mask_key] = mask

    if "text" in batch[0]:
        out["text"] = torch.stack([b["text"] for b in batch])

    if "motion_seq" in batch[0]:
        seqs = [b["motion_seq"] for b in batch]
        lengths = torch.tensor([s.size(0) for s in seqs])
        padded = pad_sequence(seqs, batch_first=True)
        mask = torch.arange(padded.size(1))[None, :] < lengths[:, None]
        out["motion_seq"] = padded
        out["motion_seq_mask"] = mask

    return out


# ============================================================================
# Model Components
# ============================================================================

def _attention_pool(sequence: torch.Tensor,
                    mask: torch.Tensor,
                    scorer: nn.Linear) -> torch.Tensor:
    """Masked attention pooling over time dimension."""
    mask = mask.bool()
    scores = scorer(sequence).squeeze(-1)
    scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    weights = torch.softmax(scores, dim=1).masked_fill(~mask, 0.0)
    return (sequence * weights.unsqueeze(-1)).sum(dim=1)


class TextMLPEncoder(nn.Module):
    """Text encoder: 2-layer MLP with LayerNorm and GELU."""
    
    def __init__(self, input_dim=768, hidden_dim=384,
                 num_labels=6, dropout=0.45):
        super().__init__()
        self.feature = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )
        self.head: nn.Module | None = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
        )

    def encode(self, text: torch.Tensor) -> torch.Tensor:
        return self.feature(text)

    def forward(self, text: torch.Tensor) -> torch.Tensor:
        if self.head is None:
            raise RuntimeError("Head removed")
        return self.head(self.encode(text))

    def remove_head(self) -> None:
        self.head = None


class AudioAttentionEncoder(nn.Module):
    """Audio encoder: attention pooling + MLP."""
    
    def __init__(self, input_dim=768, hidden_dim=384,
                 num_labels=6, dropout=0.45):
        super().__init__()
        self.attn = nn.Linear(input_dim, 1)
        self.proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )
        self.head: nn.Module | None = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
        )

    def encode(self, audio: torch.Tensor,
               mask: torch.Tensor) -> torch.Tensor:
        return self.proj(_attention_pool(audio, mask, self.attn))

    def forward(self, audio: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        if self.head is None:
            raise RuntimeError("Head removed")
        return self.head(self.encode(audio, mask))

    def remove_head(self) -> None:
        self.head = None


class VisionAttentionEncoder(nn.Module):
    """Vision encoder: attention pooling + MLP."""
    
    def __init__(self, input_dim=384, hidden_dim=384,
                 num_labels=6, dropout=0.45):
        super().__init__()
        self.attn = nn.Linear(input_dim, 1)
        self.proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )
        self.head: nn.Module | None = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
        )

    def encode(self, vision: torch.Tensor,
               mask: torch.Tensor) -> torch.Tensor:
        return self.proj(_attention_pool(vision, mask, self.attn))

    def forward(self, vision: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        if self.head is None:
            raise RuntimeError("Head removed")
        return self.head(self.encode(vision, mask))

    def remove_head(self) -> None:
        self.head = None


class MotionAttentionEncoder(nn.Module):
    """Motion encoder: attention pooling over per-frame motion sequences."""
    
    def __init__(self, input_dim=23, hidden_dim=128,
                 num_labels=6, dropout=0.45):
        super().__init__()
        self.input_dim = input_dim
        self.attn = nn.Linear(input_dim, 1)
        self.proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )
        self.head: nn.Module | None = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
        )
 
    def encode(self, motion: torch.Tensor,
               mask: torch.Tensor) -> torch.Tensor:
        if motion.size(-1) != self.input_dim:
            raise ValueError(
                f"Motion feature dim mismatch: got {motion.size(-1)}, "
                f"expected {self.input_dim}"
            )
        return self.proj(_attention_pool(motion, mask, self.attn))
 
    def forward(self, motion: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        if self.head is None:
            raise RuntimeError("Head removed")
        return self.head(self.encode(motion, mask))
 
    def remove_head(self) -> None:
        self.head = None


class FusionRegressor(nn.Module):
    """Fusion head: concatenates frozen encoder outputs and regresses emotions."""
    
    def __init__(self, text_dim, audio_dim, vision_dim=0, 
                 motion_dim=0, hidden_dim=128, num_labels=6, dropout=0.45):
        super().__init__()
        input_dim = text_dim + audio_dim + vision_dim + motion_dim
        self.mlp = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Dropout(dropout),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
        )
    
    def forward(self, *features):
        return self.mlp(torch.cat(list(features), dim=1))


class StackedFusionModel(nn.Module):
    """Container holding all four encoding stages and fusion head."""
    
    def __init__(
        self,
        text_encoder: TextMLPEncoder,
        audio_encoder: AudioAttentionEncoder,
        vision_encoder: VisionAttentionEncoder,
        fusion_head: FusionRegressor,
        motion_encoder: MotionAttentionEncoder | None = None,
        modality_drop_prob: float = 0.0,
    ):
        super().__init__()
        self.text_encoder   = text_encoder
        self.audio_encoder  = audio_encoder
        self.vision_encoder = vision_encoder
        self.motion_encoder = motion_encoder
        self.fusion_head    = fusion_head
        self.modality_drop_prob = modality_drop_prob
 
    def train(self, mode: bool = True):
        """Override to keep frozen encoders in eval mode."""
        super().train(mode)
        return self
 
    def _modality_dropout(self, feats):
        """Randomly drop modalities during training (for robustness)."""
        if self.modality_drop_prob <= 0.0 or not self.training:
            return feats
        B, N = feats[0].size(0), len(feats)
        device = feats[0].device
        keep = (torch.rand(B, N, device=device) >= self.modality_drop_prob).float()
        all_dropped = keep.sum(dim=1) == 0
        if all_dropped.any():
            choice = torch.randint(0, N, (all_dropped.sum(),), device=device)
            keep[all_dropped, choice] = 1.0
        scale = 1.0 / max(1e-3, 1.0 - self.modality_drop_prob)
        return [f * keep[:, i].unsqueeze(1) * scale for i, f in enumerate(feats)]
 
    def forward(
        self,
        text, audio, audio_mask, vision, vision_mask,
        motion_seq=None, motion_seq_mask=None,
    ):
        """Forward pass through all encoders + fusion head."""
        feats = [
            self.text_encoder.encode(text),
            self.audio_encoder.encode(audio, audio_mask),
            self.vision_encoder.encode(vision, vision_mask),
        ]
        if (self.motion_encoder is not None
                and motion_seq is not None
                and motion_seq_mask is not None):
            feats.append(self.motion_encoder.encode(motion_seq, motion_seq_mask))
 
        feats = self._modality_dropout(feats)
        return self.fusion_head(*feats)
 

# ============================================================================
# Loss Functions
# ============================================================================

def ccc_loss(preds: torch.Tensor, targets: torch.Tensor,
             eps: float = 1e-8) -> torch.Tensor:
    """Concordance Correlation Coefficient loss (1 - CCC, averaged over dimensions)."""
    loss = torch.tensor(0.0, device=preds.device)
    for d in range(preds.size(1)):
        x, y = preds[:, d], targets[:, d]
        mx, my = x.mean(), y.mean()
        vx = x.var(unbiased=False)
        vy = y.var(unbiased=False)
        cov = ((x - mx) * (y - my)).mean()
        ccc = (2.0 * cov) / (vx + vy + (mx - my).pow(2) + eps)
        loss = loss + (1.0 - ccc)
    return loss / preds.size(1)


def combined_loss(preds: torch.Tensor, targets: torch.Tensor,
                  alpha: float = 0.7) -> torch.Tensor:
    """Combined loss: alpha * CCC + (1-alpha) * MSE."""
    return alpha * ccc_loss(preds, targets) + (1.0 - alpha) * F.mse_loss(preds, targets)


# ============================================================================
# Metrics
# ============================================================================

def pearson_per_dim_fixed(
    preds: torch.Tensor, targets: torch.Tensor, eps: float = 1e-8
) -> Tuple[List[float], float]:
    """
    Compute Pearson correlation per dimension (clamped to [-1, 1]).
    Returns: (per_dim_list, average_value)
    """
    pc = preds   - preds.mean(dim=0, keepdim=True)
    tc = targets - targets.mean(dim=0, keepdim=True)
    num  = (pc * tc).sum(dim=0)
    denom = torch.sqrt(pc.pow(2).sum(dim=0) * tc.pow(2).sum(dim=0)).clamp(min=eps)
    per_dim = (num / denom).clamp(-1.0, 1.0)
    return per_dim.cpu().tolist(), per_dim.mean().item()


# ============================================================================
# Training Utilities
# ============================================================================

def move_batch(batch: Dict, device: torch.device) -> Dict:
    """Move batch tensors to device."""
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()}


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    device: torch.device,
    forward_fn: Callable,
    optimizer: Optional[torch.optim.Optimizer],
    grad_clip: float,
) -> Dict[str, float]:
    """Run one epoch of training or evaluation."""
    training = optimizer is not None

    if training:
        model.train()
    else:
        model.eval()

    total_loss = total_mse = 0.0
    n_batches = 0
    preds_all: List[torch.Tensor] = []
    targets_all: List[torch.Tensor] = []

    for batch in loader:
        batch = move_batch(batch, device)
        if training:
            optimizer.zero_grad()
            preds = forward_fn(model, batch)
            targets = batch["labels"]
            loss = criterion(preds, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        else:
            with torch.no_grad():
                preds = forward_fn(model, batch)
                targets = batch["labels"]
                loss = criterion(preds, targets)

        total_loss += loss.item()
        total_mse  += F.mse_loss(preds, targets).item()
        preds_all.append(preds.detach().cpu())
        targets_all.append(targets.detach().cpu())
        n_batches += 1

    per_dim: List[float] = []
    pearson_avg = float("nan")
    if preds_all:
        pf = torch.cat(preds_all,   dim=0)
        tf = torch.cat(targets_all, dim=0)
        per_dim, pearson_avg = pearson_per_dim_fixed(pf, tf)

    return {
        "loss":             total_loss / max(n_batches, 1),
        "mse":              total_mse  / max(n_batches, 1),
        "pearson":          pearson_avg,
        "pearson_per_dim":  per_dim,
    }


def train_stage(
    name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    criterion,
    device: torch.device,
    forward_fn: Callable,
    epochs: int,
    lr: float,
    weight_decay: float,
    grad_clip: float,
    patience: Optional[int],
    save_path: Optional[str] = None,
) -> None:
    """Train a single stage with early stopping."""
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=max(3, (patience or 10) // 3)
    )

    best_state = None
    best_val   = float("-inf")
    best_epoch = 0
    best_per_dim: List[float] = []
    no_improve  = 0

    log(f"--- Stage: {name} | epochs={epochs} lr={lr} wd={weight_decay} ---")

    for epoch in range(1, epochs + 1):
        tr = run_epoch(model, train_loader, criterion, device,
                       forward_fn, optimizer, grad_clip)

        msg = (f"[{name}] Epoch {epoch}/{epochs} | "
               f"Train Loss: {tr['loss']:.6f} | "
               f"Train Pearson: {tr['pearson']:.6f}")

        if val_loader is not None:
            vl = run_epoch(model, val_loader, criterion, device,
                           forward_fn, None, grad_clip)
            scheduler.step(vl["pearson"])
            msg += (f" | Val Loss: {vl['loss']:.6f} | "
                    f"Val Pearson: {vl['pearson']:.6f} | "
                    f"Per-dim: {[round(v, 6) for v in vl['pearson_per_dim']]}")

            if not torch.isnan(torch.tensor(vl["pearson"])):
                if vl["pearson"] > best_val + 1e-4:
                    best_val      = vl["pearson"]
                    best_state    = deepcopy(model.state_dict())
                    best_epoch    = epoch
                    best_per_dim  = vl["pearson_per_dim"]
                    no_improve    = 0
                    msg += " *** best"
                    if save_path:
                        torch.save({"epoch": epoch,
                                    "model_state_dict": best_state,
                                    "best_pearson": best_val}, save_path)
                else:
                    no_improve += 1
                    if patience and no_improve >= patience:
                        log(msg)
                        log(f"[{name}] Early stop at epoch {epoch}")
                        break

        log(msg)

    if best_state is not None:
        model.load_state_dict(best_state)
        log(f"[{name}] Best | epoch={best_epoch} | "
            f"val_pearson={best_val:.6f} | "
            f"per_dim={[round(v, 6) for v in best_per_dim]}")


def make_loader(csv, args, load_face, load_audio, load_text,
                batch_size, shuffle, load_motion_seq=False):
    """Create a DataLoader for specified modalities."""
    ds = StagedDataset(
        split_csv=csv,
        face_dir=args.face_dir,
        audio_dir=args.audio_dir,
        text_dir=args.text_dir,
        load_face=load_face,
        load_audio=load_audio,
        load_text=load_text,
        text_input_dim=args.text_input_dim,
        use_motion_seq=load_motion_seq,
        motion_seq_dir=args.motion_seq_dir if load_motion_seq else None,
    )
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=args.num_workers, pin_memory=False,
        collate_fn=staged_collate,
    )
    return loader, len(ds)


# ============================================================================
# Main Training Script
# ============================================================================

def main() -> None:
    """Main training pipeline."""
    parser = argparse.ArgumentParser(
        description="Staged Multimodal Fusion Training"
    )

    # Data paths
    parser.add_argument("--face_dir",  type=str, required=True,
        help="Path to directory containing face embedding pkl files")
    parser.add_argument("--audio_dir", type=str, required=True,
        help="Path to directory containing audio embedding pkl files")
    parser.add_argument("--text_dir",  type=str, required=True,
        help="Path to directory containing text embedding pkl files")
    parser.add_argument("--train_csv", type=str, required=True,
        help="Path to training split CSV (with Filename and label columns)")
    parser.add_argument("--valid_csv", type=str, required=True,
        help="Path to validation split CSV")

    # Modality dimensions
    parser.add_argument("--face_input_dim",  type=int, default=768,
        help="Face embedding dimension (e.g., DINOv2-base=768)")
    parser.add_argument("--text_input_dim",  type=int, default=768,
        help="Text embedding dimension")
    parser.add_argument("--audio_input_dim", type=int, default=1024,
        help="Audio embedding dimension")

    # Architecture
    parser.add_argument("--text_hidden",   type=int,   default=384)
    parser.add_argument("--audio_hidden",  type=int,   default=384)
    parser.add_argument("--vision_hidden", type=int,   default=384)
    parser.add_argument("--fusion_hidden", type=int,   default=384)
    parser.add_argument("--dropout",       type=float, default=0.45)
    parser.add_argument("--modality_drop_prob", type=float, default=0.3,
        help="Probability of dropping modalities during training (0-1)")

    # Training hyperparams
    parser.add_argument("--text_epochs",   type=int,   default=50)
    parser.add_argument("--audio_epochs",  type=int,   default=50)
    parser.add_argument("--vision_epochs", type=int,   default=50)
    parser.add_argument("--fusion_epochs", type=int,   default=50)
    parser.add_argument("--lr",            type=float, default=2e-4,
        help="Learning rate")
    parser.add_argument("--weight_decay",  type=float, default=1e-2)
    parser.add_argument("--grad_clip",     type=float, default=1.0)
    parser.add_argument("--batch_size",    type=int,   default=16)
    parser.add_argument("--eval_batch_size", type=int, default=None,
        help="Evaluation batch size (default: same as batch_size)")
    parser.add_argument("--num_workers",   type=int,   default=0)
    parser.add_argument("--patience",      type=int,   default=10,
        help="Early stopping patience")
    parser.add_argument("--loss",          type=str,   default="combined",
        choices=["ccc", "combined"],
        help="Loss function to use")
    parser.add_argument("--loss_alpha",    type=float, default=0.7,
        help="Weight on CCC in combined loss (0-1)")

    # Motion (optional)
    parser.add_argument("--use_motion_seq", action="store_true",
        help="Use per-frame motion sequence pkl files as additional encoder")
    parser.add_argument("--motion_seq_dir", type=str, default=None,
        help="Path to directory containing motion sequence pkl files")
    parser.add_argument("--motion_hidden_dim", type=int, default=128,
        help="Hidden dimension for motion attention encoder")
    parser.add_argument("--motion_epochs", type=int, default=100,
        help="Epochs for motion encoder stage")

    # Output
    parser.add_argument("--save_dir", type=str, required=True,
        help="Path to save best checkpoints and metadata")
    parser.add_argument("--log_dir",  type=str, default="./logs",
        help="Path to save training logs")
    parser.add_argument("--device",   type=str, default="cuda:0",
        help="Device to train on (e.g., cuda:0, cpu)")
    parser.add_argument("--seed",     type=int, default=42)

    args = parser.parse_args()

    # Setup
    log_path = setup_logging(args.log_dir)
    log(f"Logging to: {log_path}")
    log(f"Arguments:\n{json.dumps(vars(args), indent=2)}")

    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")

    os.makedirs(args.save_dir, exist_ok=True)
    eval_bs = args.eval_batch_size or args.batch_size

    # Loss function
    criterion = (
        combined_loss if args.loss == "combined"
        else ccc_loss
    )
    if args.loss == "combined":
        alpha = args.loss_alpha
        criterion = lambda p, t: combined_loss(p, t, alpha=alpha)

    # =======================================================================
    # Stage 1: Text MLP Encoder
    # =======================================================================
    log("=" * 70)
    log("STAGE 1 — Text MLP Encoder")
    tr_text, tr_text_n = make_loader(args.train_csv, args,
        load_face=False, load_audio=False, load_text=True,
        batch_size=args.batch_size, shuffle=True)
    va_text, _ = make_loader(args.valid_csv, args,
        load_face=False, load_audio=False, load_text=True,
        batch_size=eval_bs, shuffle=False)
    log(f"Text training set size: {tr_text_n}")

    text_model = TextMLPEncoder(
        input_dim=args.text_input_dim,
        hidden_dim=args.text_hidden,
        dropout=args.dropout,
    ).to(device)

    text_ckpt_path = os.path.join(args.save_dir, "text_encoder_best.pt")
    if os.path.exists(text_ckpt_path):
        log(f"✓ Loading existing checkpoint: {text_ckpt_path}")
        text_model.load_state_dict(
            torch.load(text_ckpt_path, map_location=device)["model_state_dict"]
        )
    else:
        train_stage(
            "text", text_model, tr_text, va_text, criterion, device,
            forward_fn=lambda m, b: m(b["text"]),
            epochs=args.text_epochs, lr=args.lr,
            weight_decay=args.weight_decay, grad_clip=args.grad_clip,
            patience=args.patience,
            save_path=text_ckpt_path,
        )
    text_model.remove_head()

    # =======================================================================
    # Stage 2: Audio Attention Encoder
    # =======================================================================
    log("=" * 70)
    log("STAGE 2 — Audio Attention Encoder")
    tr_audio, tr_audio_n = make_loader(args.train_csv, args,
        load_face=False, load_audio=True, load_text=False,
        batch_size=args.batch_size, shuffle=True)
    va_audio, _ = make_loader(args.valid_csv, args,
        load_face=False, load_audio=True, load_text=False,
        batch_size=eval_bs, shuffle=False)
    log(f"Audio training set size: {tr_audio_n}")

    audio_model = AudioAttentionEncoder(
        input_dim=args.audio_input_dim,
        hidden_dim=args.audio_hidden,
        dropout=args.dropout,
    ).to(device)

    audio_ckpt_path = os.path.join(args.save_dir, "audio_encoder_best.pt")
    if os.path.exists(audio_ckpt_path):
        log(f"✓ Loading existing checkpoint: {audio_ckpt_path}")
        audio_model.load_state_dict(
            torch.load(audio_ckpt_path, map_location=device)["model_state_dict"]
        )
    else:
        train_stage(
            "audio", audio_model, tr_audio, va_audio, criterion, device,
            forward_fn=lambda m, b: m(b["audio"], b["audio_mask"]),
            epochs=args.audio_epochs, lr=args.lr,
            weight_decay=args.weight_decay, grad_clip=args.grad_clip,
            patience=args.patience,
            save_path=audio_ckpt_path,
        )
    audio_model.remove_head()

    # =======================================================================
    # Stage 3: Vision Attention Encoder
    # =======================================================================
    log("=" * 70)
    log("STAGE 3 — Vision Attention Encoder")
    tr_vis, tr_vis_n = make_loader(args.train_csv, args,
        load_face=True, load_audio=False, load_text=False,
        batch_size=args.batch_size, shuffle=True)
    va_vis, _ = make_loader(args.valid_csv, args,
        load_face=True, load_audio=False, load_text=False,
        batch_size=eval_bs, shuffle=False)
    log(f"Vision training set size: {tr_vis_n}")

    vision_model = VisionAttentionEncoder(
        input_dim=args.face_input_dim,
        hidden_dim=args.vision_hidden,
        dropout=args.dropout,
    ).to(device)

    vision_ckpt_path = os.path.join(args.save_dir, "vision_encoder_best.pt")
    if os.path.exists(vision_ckpt_path):
        log(f"✓ Loading existing checkpoint: {vision_ckpt_path}")
        vision_model.load_state_dict(
            torch.load(vision_ckpt_path, map_location=device)["model_state_dict"]
        )
    else:
        train_stage(
            "vision", vision_model, tr_vis, va_vis, criterion, device,
            forward_fn=lambda m, b: m(b["vision"], b["vision_mask"]),
            epochs=args.vision_epochs, lr=args.lr,
            weight_decay=args.weight_decay, grad_clip=args.grad_clip,
            patience=args.patience,
            save_path=vision_ckpt_path,
        )
    vision_model.remove_head()

    # =======================================================================
    # Stage 4 (Optional): Motion Attention Encoder
    # =======================================================================
    motion_model = None
    motion_enc_dim = 0
 
    if args.use_motion_seq:
        log("=" * 70)
        log("STAGE 4 — Motion Attention Encoder")
        tr_mot, tr_mot_n = make_loader(args.train_csv, args,
            load_face=False, load_audio=False, load_text=False,
            batch_size=args.batch_size, shuffle=True,
            load_motion_seq=True)
        va_mot, _ = make_loader(args.valid_csv, args,
            load_face=False, load_audio=False, load_text=False,
            batch_size=eval_bs, shuffle=False,
            load_motion_seq=True)
        log(f"Motion training set size: {tr_mot_n}")

        motion_feat_dim = tr_mot.dataset.motion_feat_dim
        log(f"Motion feature dimension: {motion_feat_dim}")
 
        motion_model = MotionAttentionEncoder(
            input_dim=motion_feat_dim,
            hidden_dim=args.motion_hidden_dim,
            dropout=args.dropout,
        ).to(device)
 
        motion_ckpt_path = os.path.join(args.save_dir, "motion_encoder_best.pt")
        if os.path.exists(motion_ckpt_path):
            log(f"✓ Loading existing checkpoint: {motion_ckpt_path}")
            motion_model.load_state_dict(
                torch.load(motion_ckpt_path, map_location=device)["model_state_dict"]
            )
        else:
            train_stage(
                "motion", motion_model, tr_mot, va_mot, criterion, device,
                forward_fn=lambda m, b: m(b["motion_seq"], b["motion_seq_mask"]),
                epochs=args.motion_epochs, lr=args.lr,
                weight_decay=args.weight_decay, grad_clip=args.grad_clip,
                patience=args.patience,
                save_path=motion_ckpt_path,
            )
        motion_model.remove_head()
        motion_enc_dim = args.motion_hidden_dim

    # =======================================================================
    # Stage 5: Fusion Regressor
    # =======================================================================
    log("=" * 70)
    motion_tag = "WITH motion" if args.use_motion_seq else "WITHOUT motion"
    log(f"STAGE 5 — Fusion Regressor ({motion_tag})")

    tr_fus, tr_fus_n = make_loader(args.train_csv, args,
        load_face=True, load_audio=True, load_text=True,
        batch_size=args.batch_size, shuffle=True,
        load_motion_seq=args.use_motion_seq)
    va_fus, _ = make_loader(args.valid_csv, args,
        load_face=True, load_audio=True, load_text=True,
        batch_size=eval_bs, shuffle=False,
        load_motion_seq=args.use_motion_seq)
    log(f"Fusion training set size: {tr_fus_n}")

    fusion_head = FusionRegressor(
        text_dim=args.text_hidden,
        audio_dim=args.audio_hidden,
        vision_dim=args.vision_hidden,
        motion_dim=motion_enc_dim,
        hidden_dim=args.fusion_hidden,
        dropout=args.dropout,
    ).to(device)

    stacked = StackedFusionModel(
        text_encoder=text_model.to(device),
        audio_encoder=audio_model.to(device),
        vision_encoder=vision_model.to(device),
        fusion_head=fusion_head,
        motion_encoder=motion_model,
        modality_drop_prob=args.modality_drop_prob,
    )

    # Setup optimizer
    fusion_optimizer = torch.optim.AdamW([
        {"params": fusion_head.parameters(),        "lr": args.lr},
        {"params": text_model.parameters(),         "lr": args.lr * 0.05},
        {"params": audio_model.parameters(),        "lr": args.lr * 0.05},
        {"params": vision_model.parameters(),       "lr": args.lr * 0.05},
    ], weight_decay=args.weight_decay)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        fusion_optimizer, mode="max", factor=0.5,
        patience=max(3, args.patience // 3)
    )

    # Forward function handling optional motion
    def fusion_forward(m, b):
        return m(
            b["text"], b["audio"], b["audio_mask"],
            b["vision"], b["vision_mask"],
            motion_seq=b.get("motion_seq"),
            motion_seq_mask=b.get("motion_seq_mask"),
        )
    
    # Training loop
    best_fusion_state = None
    best_fusion_val   = float("-inf")
    best_fusion_epoch = 0
    best_fusion_per_dim: List[float] = []
    no_improve = 0

    best_path = os.path.join(
        args.save_dir,
        f"fusion_{'motion' if args.use_motion_seq else 'no_motion'}_best.pt"
    )

    trainable_n = sum(p.numel() for p in stacked.parameters())
    log(f"Total trainable parameters: {trainable_n:,}")

    for epoch in range(1, args.fusion_epochs + 1):
        tr = run_epoch(stacked, tr_fus, criterion, device,
                       fusion_forward, fusion_optimizer, args.grad_clip)

        vl = run_epoch(stacked, va_fus, criterion, device,
                       fusion_forward, None, args.grad_clip)

        scheduler.step(vl["pearson"])

        is_best = (
            not torch.isnan(torch.tensor(vl["pearson"]))
            and vl["pearson"] > best_fusion_val + 1e-4
        )

        if is_best:
            best_fusion_val   = vl["pearson"]
            best_fusion_state = deepcopy(fusion_head.state_dict())
            best_fusion_epoch = epoch
            best_fusion_per_dim = vl["pearson_per_dim"]
            no_improve = 0
            torch.save({
                "epoch":            epoch,
                "model_state_dict": stacked.state_dict(),
                "fusion_state_dict": fusion_head.state_dict(),
                "best_pearson":     best_fusion_val,
                "pearson_per_dim":  best_fusion_per_dim,
                "use_motion":       args.use_motion_seq,
            }, best_path)
        else:
            no_improve += 1

        log(
            f"[fusion] Epoch {epoch}/{args.fusion_epochs} | "
            f"Train Loss: {tr['loss']:.6f} | "
            f"Val Loss: {vl['loss']:.6f} | "
            f"Val Pearson(avg): {vl['pearson']:.6f} | "
            f"Per-dim: {[round(v, 6) for v in vl['pearson_per_dim']]}"
            + (" ⭐ best" if is_best else "")
        )

        if no_improve >= args.patience:
            log(f"[fusion] Early stopping at epoch {epoch}")
            break

    if best_fusion_state is not None:
        fusion_head.load_state_dict(best_fusion_state)

    # Summary and save metadata
    log("=" * 70)
    log(
        f"✓ TRAINING COMPLETE | "
        f"Best Validation Pearson: {best_fusion_val:.6f} | "
        f"Epoch: {best_fusion_epoch}"
    )
    log(f"Per-dimension Pearson: {[round(v, 6) for v in best_fusion_per_dim]}")

    meta = {
        "best_epoch":           best_fusion_epoch,
        "best_val_pearson_avg": best_fusion_val,
        "best_val_pearson_per_dim": best_fusion_per_dim,
        "use_motion":           args.use_motion_seq,
        "label_columns":        LABEL_COLUMNS,
        "args":                 vars(args),
        "timestamp":            datetime.now().isoformat(),
    }
    meta_path = best_path.replace(".pt", "_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    log(f"✓ Metadata saved: {meta_path}")
    log(f"✓ Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
