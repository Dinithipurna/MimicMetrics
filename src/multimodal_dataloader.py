"""
Multimodal dataloader for ViT + Wav2Vec2 fusion training.
"""

import os
import pickle
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class MultimodalDataset(Dataset):
    """Load both ViT and Wav2Vec2 embeddings for the same sample."""
    
    @staticmethod
    def _normalize_filename(filename_value):
        """Normalize CSV filename values to match embedding files."""
        value = str(filename_value).strip()
        if value.endswith('.0'):
            value = value[:-2]
        if value.isdigit():
            value = value.zfill(5)
        return value
    
    def __init__(self, vit_dir, wav2vec2_dir, split_csv, label_columns=None):
        """
        Args:
            vit_dir: Directory with ViT .pkl embedding files
            wav2vec2_dir: Directory with Wav2Vec2 .pkl embedding files
            split_csv: CSV file with filename and emotion labels
            label_columns: List of emotion label column names
        """
        self.vit_dir = vit_dir
        self.wav2vec2_dir = wav2vec2_dir
        self.samples = pd.read_csv(split_csv, dtype={'Filename': str})
        
        if label_columns is None:
            label_columns = [col for col in self.samples.columns if col != 'Filename']
        
        self.label_columns = label_columns
        
        # Verify both modalities exist for each sample
        missing_vit = []
        missing_wav = []
        for filename in self.samples['Filename']:
            normalized = self._normalize_filename(filename)
            vit_path = os.path.join(vit_dir, f"{normalized}.pkl")
            wav_path = os.path.join(wav2vec2_dir, f"{normalized}.pkl")
            
            if not os.path.exists(vit_path):
                missing_vit.append(normalized)
            if not os.path.exists(wav_path):
                missing_wav.append(normalized)
        
        if missing_vit or missing_wav:
            print(f"Warning: Missing embeddings")
            if missing_vit:
                print(f"  ViT: {len(missing_vit)} files (e.g., {missing_vit[:3]})")
            if missing_wav:
                print(f"  Wav2Vec2: {len(missing_wav)} files (e.g., {missing_wav[:3]})")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Returns:
            vit_emb: (seq_len_v, 384) tensor
            wav2vec2_emb: (seq_len_a, 768) tensor
            labels: (num_emotions,) tensor
        """
        row = self.samples.iloc[idx]
        filename = self._normalize_filename(row['Filename'])
        
        # Load ViT embedding
        vit_path = os.path.join(self.vit_dir, f"{filename}.pkl")
        with open(vit_path, 'rb') as f:
            vit_emb = pickle.load(f)
        vit_emb = torch.from_numpy(vit_emb).float()
        
        # Load Wav2Vec2 embedding
        wav_path = os.path.join(self.wav2vec2_dir, f"{filename}.pkl")
        with open(wav_path, 'rb') as f:
            wav2vec2_emb = pickle.load(f)
        wav2vec2_emb = torch.from_numpy(wav2vec2_emb).float()
        
        # Load labels
        labels = torch.tensor([row[col] for col in self.label_columns], dtype=torch.float32)
        
        return vit_emb, wav2vec2_emb, labels


def collate_multimodal(batch):
    """Custom collate to handle variable sequence lengths for both modalities."""
    vit_embs, wav2vec2_embs, labels = zip(*batch)
    
    # Pad ViT embeddings
    max_len_v = max(e.shape[0] for e in vit_embs)
    padded_vit = []
    for e in vit_embs:
        if e.shape[0] < max_len_v:
            padding = torch.zeros(max_len_v - e.shape[0], e.shape[1])
            e = torch.cat([e, padding], dim=0)
        padded_vit.append(e)
    
    # Pad Wav2Vec2 embeddings
    max_len_a = max(e.shape[0] for e in wav2vec2_embs)
    padded_wav = []
    for e in wav2vec2_embs:
        if e.shape[0] < max_len_a:
            padding = torch.zeros(max_len_a - e.shape[0], e.shape[1])
            e = torch.cat([e, padding], dim=0)
        padded_wav.append(e)
    
    vit_batch = torch.stack(padded_vit)
    wav_batch = torch.stack(padded_wav)
    labels_batch = torch.stack(labels)
    
    return vit_batch, wav_batch, labels_batch


def create_multimodal_dataloaders(vit_dir, wav2vec2_dir, train_csv, valid_csv,
                                   batch_size=32, valid_batch_size=None, num_workers=4):
    """
    Create multimodal dataloaders.
    
    Returns:
        dict with keys: 'train', 'valid'
    """
    label_columns = ['Admiration', 'Amusement', 'Determination', 'Empathic Pain', 'Excitement', 'Joy']

    if valid_batch_size is None:
        valid_batch_size = batch_size
    
    train_dataset = MultimodalDataset(vit_dir, wav2vec2_dir, train_csv, label_columns)
    valid_dataset = MultimodalDataset(vit_dir, wav2vec2_dir, valid_csv, label_columns)
    
    loaders = {
        'train': DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, collate_fn=collate_multimodal
        ),
        'valid': DataLoader(
            valid_dataset, batch_size=valid_batch_size, shuffle=False,
            num_workers=num_workers, collate_fn=collate_multimodal
        ),
    }
    
    return loaders
