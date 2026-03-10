#!/usr/bin/env python3
"""
Validation script for multimodal baseline models.
Loads best checkpoints and evaluates on validation set.
"""

import json
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
import torch.nn as nn
import tqdm  # Ensure tqdm is imported at the top of the file
from src.baselines import MultimodalConcatBaseline, MultimodalCrossAttentionBaseline
from src.multimodal_dataloader import create_multimodal_dataloaders
from src.train_baselines import BaselineTrainer
from sklearn.metrics import mean_squared_error, mean_absolute_error  # Add this import at the top of the file
from scipy.stats import pearsonr  # Add this import at the top of the file

def validate_model(model, checkpoint_path, dataloader, device, model_name):
    """Load checkpoint and evaluate."""
    print(f"\n{'='*60}")
    print(f"Validating {model_name.upper()} Baseline")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"{'='*60}")

    # Load checkpoint
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)

    # Evaluate
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0

    criterion = nn.MSELoss()

    with torch.no_grad():
        for vit_emb, wav2vec2_emb, labels in tqdm.tqdm(dataloader, desc="Evaluating"):
            vit_emb = vit_emb.to(device)
            wav2vec2_emb = wav2vec2_emb.to(device)
            labels = labels.to(device)

            logits = model(vit_emb, wav2vec2_emb)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            all_preds.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    mse = mean_squared_error(all_labels, all_preds)
    mae = mean_absolute_error(all_labels, all_preds)

    flat_preds = all_preds.flatten()
    flat_labels = all_labels.flatten()
    pearson_corr, _ = pearsonr(flat_labels, flat_preds)

    metrics = {
        'loss': total_loss / len(dataloader),
        'mse': mse,
        'mae': mae,
        'pearson': pearson_corr
    }

    print(f"\nResults:")
    print(f"  Loss:    {metrics['loss']:.6f}")
    print(f"  MSE:     {metrics['mse']:.6f}")
    print(f"  MAE:     {metrics['mae']:.6f}")
    print(f"  Pearson: {metrics['pearson']:.6f}")

    return metrics

def main():
    base_dir = Path('/home/dinithi/Documents/EMI_Challenge')
    checkpoint_dir = base_dir / 'results/checkpoints/trained'
    results_dir = base_dir / 'results/logs'
    results_dir.mkdir(parents=True, exist_ok=True)

    # Paths
    vit_dir = base_dir / 'results/checkpoints/baseline/vit'
    wav2vec2_dir = base_dir / 'results/checkpoints/baseline/wav2vec2'
    train_csv = base_dir / 'data/splits/train_split.csv'
    valid_csv = base_dir / 'data/splits/valid_split.csv'

    concat_checkpoint = checkpoint_dir / 'best_multimodal_concat.pt'
    cross_attn_checkpoint = checkpoint_dir / 'best_multimodal_attention.pt'

    # Device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load dataloaders
    print("\nLoading multimodal validation data...")
    dataloaders = create_multimodal_dataloaders(
        str(vit_dir), str(wav2vec2_dir),
        str(train_csv), str(valid_csv),
        batch_size=32, num_workers=4
    )

    results = {
        'timestamp': datetime.now().isoformat(),
        'device': str(device),
        'models': {}
    }

    # Validate Multimodal Concat Baseline
    if concat_checkpoint.exists():
        model_concat = MultimodalConcatBaseline(vit_dim=384, wav2vec2_dim=768, hidden_dim=512, output_dim=6)
        metrics_concat = validate_model(
            model_concat, str(concat_checkpoint),
            dataloaders['valid'], device, 'multimodal_concat'
        )
        results['models']['multimodal_concat'] = metrics_concat
    else:
        print(f"\n⚠ Multimodal Concat checkpoint not found: {concat_checkpoint}")

    # Validate Multimodal Cross-Attention Baseline
    if cross_attn_checkpoint.exists():
        model_cross_attn = MultimodalCrossAttentionBaseline(vit_dim=384, wav2vec2_dim=768, hidden_dim=256, num_heads=4, output_dim=6)
        dataloader = create_multimodal_dataloaders(
            vit_dir, wav2vec2_dir,
            train_csv, valid_csv,
            batch_size=1,  # Reduce batch size to 1 for cross-attention
            num_workers=4
        )['valid']
        metrics_cross_attn = validate_model(
            model_cross_attn, str(cross_attn_checkpoint),
            dataloader, device, 'multimodal_cross_attention'
        )
        results['models']['multimodal_cross_attention'] = metrics_cross_attn
    else:
        print(f"\n⚠ Multimodal Cross-Attention checkpoint not found: {cross_attn_checkpoint}")

    # Save results
    log_path = results_dir / f'validation_multimodal_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(log_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"✓ Multimodal validation complete!")
    print(f"Results saved to: {log_path}")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    main()