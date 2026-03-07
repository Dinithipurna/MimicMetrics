#!/usr/bin/env python3
"""
Quick start: Train both baseline models with default settings.
Run directly: python run_baselines.py
"""

import warnings
warnings.filterwarnings('ignore')

import os
import sys
from pathlib import Path

# Add project to path
project_dir = Path('/home/dinithi/Documents/EMI_Challenge')
sys.path.insert(0, str(project_dir))

import torch
from src.train_baselines import train_baseline, create_dataloaders

print("""
╔════════════════════════════════════════════════════════════════════╗
║                   EMI Challenge - Baseline Training                ║
║                                                                    ║
║  Models:                                                           ║
║    • ViT + 3-layer GRU (Target: MSE 0.09)                         ║
║    • Wav2Vec2 + Linear  (Target: MSE 0.24)                        ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝
""")

# Configuration
CONFIG = {
    'num_epochs': 50,
    'batch_size': 32,
    'learning_rate': 1e-3,
    'device': 'cuda:0' if torch.cuda.is_available() else 'cpu'
}

print(f"Configuration:")
print(f"  Epochs:        {CONFIG['num_epochs']}")
print(f"  Batch Size:    {CONFIG['batch_size']}")
print(f"  Learning Rate: {CONFIG['learning_rate']}")
print(f"  Device:        {CONFIG['device']}\n")

# Paths
vit_dir = project_dir / 'results/checkpoints/baseline/vit'
wav2vec2_dir = project_dir / 'results/checkpoints/baseline/wav2vec2'
train_csv = project_dir / 'data/splits/train_split.csv'
valid_csv = project_dir / 'data/splits/valid_split.csv'
checkpoint_dir = project_dir / 'results/checkpoints/trained'

# Verify paths
print("Checking data...")
missing = []
if not vit_dir.exists(): missing.append(f"ViT embeddings ({vit_dir})")
if not wav2vec2_dir.exists(): missing.append(f"Wav2Vec2 embeddings ({wav2vec2_dir})")
if not train_csv.exists(): missing.append(f"Train split ({train_csv})")
if not valid_csv.exists(): missing.append(f"Valid split ({valid_csv})")

if missing:
    print("❌ Missing files:")
    for item in missing:
        print(f"   - {item}")
    sys.exit(1)

print("✓ All data files found\n")

# Load dataloaders
print("Loading dataloaders...")
try:
    dataloaders = create_dataloaders(
        str(vit_dir), str(wav2vec2_dir),
        str(train_csv), str(valid_csv),
        batch_size=CONFIG['batch_size'],
        num_workers=4
    )
    print("✓ Dataloaders ready\n")
except Exception as e:
    print(f"❌ Error loading dataloaders: {e}")
    sys.exit(1)

# Select device
device = torch.device(CONFIG['device'] if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")

# Train both models
try:
    print("="*70)
    trainer_vit, history_vit = train_baseline(
        'vit', dataloaders, device,
        num_epochs=CONFIG['num_epochs'],
        learning_rate=CONFIG['learning_rate'],
        save_dir=str(checkpoint_dir)
    )
except KeyboardInterrupt:
    print("\n⚠ ViT training interrupted by user")
except Exception as e:
    print(f"\n❌ ViT training error: {e}")
    import traceback
    traceback.print_exc()

try:
    print("\n" + "="*70)
    trainer_wav2vec2, history_wav2vec2 = train_baseline(
        'wav2vec2', dataloaders, device,
        num_epochs=CONFIG['num_epochs'],
        learning_rate=CONFIG['learning_rate'],
        save_dir=str(checkpoint_dir)
    )
except KeyboardInterrupt:
    print("\n⚠ Wav2Vec2 training interrupted by user")
except Exception as e:
    print(f"\n❌ Wav2Vec2 training error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("✓ Training complete!")
print(f"Checkpoints saved to: {checkpoint_dir}")
print("="*70)
