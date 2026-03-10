#!/usr/bin/env python3
"""
Quick start: Train multimodal fusion baselines.
Run directly: python run_multimodal.py
"""

import warnings
warnings.filterwarnings('ignore')

import sys
import argparse
import gc
from pathlib import Path

# Add project to path
project_dir = Path('/home/dinithi/Documents/EMI_Challenge')
sys.path.insert(0, str(project_dir))

import torch
from src.train_multimodal import train_multimodal, create_multimodal_dataloaders

print("""
╔════════════════════════════════════════════════════════════════════╗
║            EMI Challenge - Multimodal Fusion Training              ║
║                                                                    ║
║  Models:                                                           ║
║    • Concat Fusion:  ViT+Wav2Vec2 → MLP                           ║
║    • Cross-Attention: ViT ⟷ Wav2Vec2 bidirectional attention     ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝
""")
# Parse arguments
parser = argparse.ArgumentParser(description='Quick multimodal training')
parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--attention_batch_size', type=int, default=8, help='Batch size for cross-attention')
parser.add_argument('--attention_eval_batch_size', type=int, default=1, help='Validation batch size for cross-attention')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--device', choices=['cpu', 'cuda:0', 'cuda:1'], default='cuda:1', help='Device')
args = parser.parse_args()


# Configuration
CONFIG = {
    'num_epochs': args.epochs,
    'batch_size': args.batch_size,
    'attention_batch_size': args.attention_batch_size,
    'attention_eval_batch_size': args.attention_eval_batch_size,
    'learning_rate': args.lr,
    'device': args.device if args.device == 'cpu' or torch.cuda.is_available() else 'cpu'
}

print(f"Configuration:")
print(f"  Epochs:        {CONFIG['num_epochs']}")
print(f"  Batch Size:    {CONFIG['batch_size']}")
print(f"  Attn Batch:    {CONFIG['attention_batch_size']}")
print(f"  Attn Eval:     {CONFIG['attention_eval_batch_size']}")
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
print("Loading multimodal dataloaders...")
try:
    dataloaders = create_multimodal_dataloaders(
        str(vit_dir), str(wav2vec2_dir),
        str(train_csv), str(valid_csv),
        batch_size=CONFIG['batch_size'],
        num_workers=4
    )

    attention_dataloaders = create_multimodal_dataloaders(
        str(vit_dir), str(wav2vec2_dir),
        str(train_csv), str(valid_csv),
        batch_size=CONFIG['attention_batch_size'],
        valid_batch_size=CONFIG['attention_eval_batch_size'],
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
    trainer_concat, history_concat = train_multimodal(
        'concat', dataloaders, device,
        num_epochs=CONFIG['num_epochs'],
        learning_rate=CONFIG['learning_rate'],
        save_dir=str(checkpoint_dir)
    )

    if str(device).startswith('cuda'):
        del trainer_concat
        gc.collect()
        torch.cuda.empty_cache()
except KeyboardInterrupt:
    print("\n⚠ Concat training interrupted by user")
except Exception as e:
    print(f"\n❌ Concat training error: {e}")
    import traceback
    traceback.print_exc()

try:
    print("\n" + "="*70)
    trainer_attention, history_attention = train_multimodal(
        'attention', attention_dataloaders, device,
        num_epochs=CONFIG['num_epochs'],
        learning_rate=CONFIG['learning_rate'],
        save_dir=str(checkpoint_dir)
    )
except KeyboardInterrupt:
    print("\n⚠ Cross-attention training interrupted by user")
except Exception as e:
    print(f"\n❌ Cross-attention training error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("✓ Training complete!")
print(f"Checkpoints saved to: {checkpoint_dir}")
print("="*70)
