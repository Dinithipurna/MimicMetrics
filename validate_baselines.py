#!/usr/bin/env python3
"""
Quick validation script for baseline models.
Loads best checkpoints and evaluates on validation set.
"""

import json
import sys
from pathlib import Path
from datetime import datetime

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

import torch
from src.baselines import VitGRUBaseline, Wav2Vec2LinearBaseline
from src.dataloaders import create_dataloaders
from src.train_baselines import BaselineTrainer

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
    trainer = BaselineTrainer(model, device)
    metrics = trainer.evaluate(dataloader)

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

    vit_checkpoint = checkpoint_dir / 'best_vit.pt'
    wav2vec2_checkpoint = checkpoint_dir / 'best_wav2vec2.pt'

    # Device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load dataloaders
    print("\nLoading validation data...")
    dataloaders = create_dataloaders(
        str(vit_dir), str(wav2vec2_dir),
        str(train_csv), str(valid_csv),
        batch_size=32, num_workers=4
    )

    results = {
        'timestamp': datetime.now().isoformat(),
        'device': str(device),
        'models': {}
    }

    # Validate ViT
    if vit_checkpoint.exists():
        model_vit = VitGRUBaseline(input_dim=384, hidden_dim=256, num_layers=3, output_dim=6)
        metrics_vit = validate_model(
            model_vit, str(vit_checkpoint), 
            dataloaders['vit_valid'], device, 'vit'
        )
        results['models']['vit'] = metrics_vit
    else:
        print(f"\n⚠ ViT checkpoint not found: {vit_checkpoint}")

    # Validate Wav2Vec2
    if wav2vec2_checkpoint.exists():
        model_wav2vec2 = Wav2Vec2LinearBaseline(input_dim=768, output_dim=6)
        metrics_wav2vec2 = validate_model(
            model_wav2vec2, str(wav2vec2_checkpoint),
            dataloaders['wav2vec2_valid'], device, 'wav2vec2'
        )
        results['models']['wav2vec2'] = metrics_wav2vec2
    else:
        print(f"\n⚠ Wav2Vec2 checkpoint not found: {wav2vec2_checkpoint}")

    # Print summary table
    print("Summary:")
    print("-" * 60)
    print(f"{'Model':<15} {'MSE':<12} {'MAE':<12} {'Pearson':<12}")
    print("-" * 60)
    for model_name, metrics in results['models'].items():
        print(f"{model_name:<15} {metrics['mse']:<12.6f} {metrics['mae']:<12.6f} {metrics['pearson']:<12.6f}")
    print("-" * 60)

    # Save results
    log_path = results_dir / f'validation_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(log_path, "w", encoding="utf-8") as f:
        for k, v in results.items():
            f.write(f"{k}: {v}\n")

    print(f"\n{'='*60}")
    print(f"✓ Validation complete!")
    print(f"Results saved to: {log_path}")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    main()
