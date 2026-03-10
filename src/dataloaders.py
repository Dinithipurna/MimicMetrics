"""
Data loaders for baseline training.
Loads pre-computed ViT and Wav2Vec2 embeddings.
"""

import os
import pickle
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch


class EmbeddingDataset(Dataset):
    """Load pre-computed embeddings and corresponding emotion labels."""

    @staticmethod
    def _normalize_filename(filename_value):
        """Normalize CSV filename values to match embedding files like 00000.pkl."""
        value = str(filename_value).strip()

        if value.endswith('.0'):
            value = value[:-2]

        if value.isdigit():
            value = value.zfill(5)

        return value
    
    def __init__(self, embedding_dir, split_csv, label_columns=None):
        """
        Args:
            embedding_dir: Directory with .pkl embedding files (one per sample)
            split_csv: CSV file with filename and emotion labels
            label_columns: List of emotion label column names. If None, uses all except 'Filename'
        """
        self.embedding_dir = embedding_dir
        self.samples = pd.read_csv(split_csv, dtype={'Filename': str})
        
        if label_columns is None:
            # Default: all columns except 'Filename'
            label_columns = [col for col in self.samples.columns if col != 'Filename']
        
        self.label_columns = label_columns
        
        # Verify all embedding files exist
        missing = []
        for filename in self.samples['Filename']:
            normalized_filename = self._normalize_filename(filename)
            pkl_path = os.path.join(embedding_dir, f"{normalized_filename}.pkl")
            if not os.path.exists(pkl_path):
                missing.append(normalized_filename)
        
        if missing:
            print(f"Warning: {len(missing)} embedding files not found")
            print(f"  Example missing: {missing[:3]}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Returns:
            embedding: (seq_len, emb_dim) as torch.Tensor
            labels: (num_emotions,) as torch.Tensor
        """
        row = self.samples.iloc[idx]
        filename = self._normalize_filename(row['Filename'])
        
        # Load embedding
        pkl_path = os.path.join(self.embedding_dir, f"{filename}.pkl")
        with open(pkl_path, 'rb') as f:
            embedding = pickle.load(f)  # (seq_len, emb_dim)
        
        # Convert to tensor
        embedding = torch.from_numpy(embedding).float()
        
        # Load labels
        labels = torch.tensor([row[col] for col in self.label_columns], dtype=torch.float32)
        
        return embedding, labels


def create_dataloaders(vit_dir, wav2vec2_dir, train_csv, valid_csv, batch_size=32, num_workers=4):
    """
    Create dataloaders for both modalities.
    
    Args:
        vit_dir: Path to ViT embeddings directory
        wav2vec2_dir: Path to Wav2Vec2 embeddings directory
        train_csv: Path to training split CSV
        valid_csv: Path to validation split CSV
        batch_size: Batch size
        num_workers: Number of data loader workers
    
    Returns:
        dict with keys:
            'vit_train', 'vit_valid', 'wav2vec2_train', 'wav2vec2_valid'
    """
    
    # Emotion labels to use
    label_columns = ['Admiration', 'Amusement', 'Determination', 'Empathic Pain', 'Excitement', 'Joy']
    
    # Create datasets
    vit_train_dataset = EmbeddingDataset(vit_dir, train_csv, label_columns)
    vit_valid_dataset = EmbeddingDataset(vit_dir, valid_csv, label_columns)
    
    wav2vec2_train_dataset = EmbeddingDataset(wav2vec2_dir, train_csv, label_columns)
    wav2vec2_valid_dataset = EmbeddingDataset(wav2vec2_dir, valid_csv, label_columns)
    
    concat_train_dataset = EmbeddingDataset(vit_dir, train_csv, label_columns)
    concat_valid_dataset = EmbeddingDataset(vit_dir, valid_csv, label_columns)

    cross_attn_train_dataset = EmbeddingDataset(vit_dir, train_csv, label_columns)
    cross_attn_valid_dataset = EmbeddingDataset(vit_dir, valid_csv, label_columns)

    # Create dataloaders
    def collate_fn(batch):
        """Custom collate to handle variable sequence lengths."""
        embeddings, labels = zip(*batch)
        
        # Pad embeddings to max length in batch
        max_len = max(e.shape[0] for e in embeddings)
        padded_embeddings = []
        for e in embeddings:
            if e.shape[0] < max_len:
                padding = torch.zeros(max_len - e.shape[0], e.shape[1])
                e = torch.cat([e, padding], dim=0)
            padded_embeddings.append(e)
        
        embeddings = torch.stack(padded_embeddings)
        labels = torch.stack(labels)
        
        return embeddings, labels
    
    loaders = {
        'vit_train': DataLoader(
            vit_train_dataset, batch_size=batch_size, shuffle=True, 
            num_workers=num_workers, collate_fn=collate_fn
        ),
        'vit_valid': DataLoader(
            vit_valid_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, collate_fn=collate_fn
        ),
        'wav2vec2_train': DataLoader(
            wav2vec2_train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, collate_fn=collate_fn
        ),
        'wav2vec2_valid': DataLoader(
            wav2vec2_valid_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, collate_fn=collate_fn
        ),
        'concat_train': DataLoader(
            concat_train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, collate_fn=collate_fn
        ),
        'concat_valid': DataLoader(
            concat_valid_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, collate_fn=collate_fn
        ),
        'cross_attn_train': DataLoader(
            cross_attn_train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, collate_fn=collate_fn
        ),
        'cross_attn_valid': DataLoader(
            cross_attn_valid_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, collate_fn=collate_fn
        )
    }
    
    return loaders
