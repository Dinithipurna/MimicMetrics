"""
Training script for EMI baselines.
Usage:
    python train_baselines.py --model vit     # Train ViT+GRU baseline
    python train_baselines.py --model wav2vec2 # Train Wav2Vec2+Linear baseline
    python train_baselines.py --model both    # Train both
"""

import argparse
import os
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tqdm

from src.baselines import VitGRUBaseline, Wav2Vec2LinearBaseline
from src.dataloaders import create_dataloaders


class BaselineTrainer:
    """Trainer for baseline models."""
    
    def __init__(self, model, device, learning_rate=1e-3):
        self.model = model.to(device)
        self.device = device
        self.optimizer = Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
    
    def train_epoch(self, dataloader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        for embeddings, labels in tqdm.tqdm(dataloader, desc="Training"):
            embeddings = embeddings.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            logits = self.model(embeddings)
            loss = self.criterion(logits, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    @torch.no_grad()
    def evaluate(self, dataloader):
        """Evaluate on validation set. Returns MSE, MAE, Pearson correlation."""
        self.model.eval()
        
        all_preds = []
        all_labels = []
        total_loss = 0
        
        for embeddings, labels in tqdm.tqdm(dataloader, desc="Evaluating"):
            embeddings = embeddings.to(self.device)
            labels = labels.to(self.device)
            
            logits = self.model(embeddings)
            loss = self.criterion(logits, labels)
            
            total_loss += loss.item()
            all_preds.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
        
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        mse = mean_squared_error(all_labels, all_preds)
        mae = mean_absolute_error(all_labels, all_preds)
        
        # Compute Pearson correlation across all samples/emotions
        flat_preds = all_preds.flatten()
        flat_labels = all_labels.flatten()
        pearson_corr, _ = pearsonr(flat_labels, flat_preds)
        
        return {
            'loss': total_loss / len(dataloader),
            'mse': mse,
            'mae': mae,
            'pearson': pearson_corr
        }
    
    def save_checkpoint(self, path):
        """Save model checkpoint."""
        torch.save(self.model.state_dict(), path)
        print(f"Checkpoint saved to {path}")


def train_baseline(model_name, dataloaders, device, num_epochs=50, learning_rate=1e-3, save_dir=None):
    """
    Train a baseline model.
    
    Args:
        model_name: 'vit' or 'wav2vec2'
        dataloaders: Dict of dataloaders from create_dataloaders()
        device: torch device
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        save_dir: Directory to save checkpoints (optional)
    """
    
    # Create model
    if model_name == 'vit':
        model = VitGRUBaseline(input_dim=384, hidden_dim=256, num_layers=3, output_dim=6)
        train_loader = dataloaders['vit_train']
        valid_loader = dataloaders['vit_valid']
    elif model_name == 'wav2vec2':
        model = Wav2Vec2LinearBaseline(input_dim=768, output_dim=6)
        train_loader = dataloaders['wav2vec2_train']
        valid_loader = dataloaders['wav2vec2_valid']
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    trainer = BaselineTrainer(model, device, learning_rate=learning_rate)
    
    best_mse = float('inf')
    best_epoch = 0
    history = {'train_loss': [], 'val_metrics': []}
    
    print(f"\n{'='*60}")
    print(f"Training {model_name.upper()} Baseline")
    print(f"{'='*60}\n")
    
    for epoch in range(num_epochs):
        train_loss = trainer.train_epoch(train_loader)
        val_metrics = trainer.evaluate(valid_loader)
        
        history['train_loss'].append(train_loss)
        history['val_metrics'].append(val_metrics)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val MSE:    {val_metrics['mse']:.4f}")
        print(f"  Val MAE:    {val_metrics['mae']:.4f}")
        print(f"  Val Pearson: {val_metrics['pearson']:.4f}")
        
        # Save best checkpoint
        if val_metrics['mse'] < best_mse:
            best_mse = val_metrics['mse']
            best_epoch = epoch
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                checkpoint_path = os.path.join(save_dir, f"best_{model_name}.pt")
                trainer.save_checkpoint(checkpoint_path)
    
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"Best Epoch: {best_epoch+1}, Best Val MSE: {best_mse:.4f}")
    print(f"{'='*60}\n")
    
    return trainer, history


def main():
    parser = argparse.ArgumentParser(description='Train EMI baselines')
    parser.add_argument('--model', choices=['vit', 'wav2vec2', 'both'], default='both',
                        help='Which model to train')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda', help='Device')
    args = parser.parse_args()
    
    # Paths
    base_dir = Path('/home/dinithi/Documents/EMI_Challenge')
    vit_dir = base_dir / 'results/checkpoints/baseline/vit'
    wav2vec2_dir = base_dir / 'results/checkpoints/baseline/wav2vec2'
    train_csv = base_dir / 'data/splits/train_split.csv'
    valid_csv = base_dir / 'data/splits/valid_split.csv'
    checkpoint_dir = base_dir / 'results/checkpoints/trained'
    
    # Verify paths exist
    assert vit_dir.exists(), f"ViT embeddings not found at {vit_dir}"
    assert wav2vec2_dir.exists(), f"Wav2Vec2 embeddings not found at {wav2vec2_dir}"
    assert train_csv.exists(), f"Train split CSV not found at {train_csv}"
    assert valid_csv.exists(), f"Valid split CSV not found at {valid_csv}"
    
    # Create dataloaders
    print("Loading data...")
    dataloaders = create_dataloaders(
        str(vit_dir), str(wav2vec2_dir),
        str(train_csv), str(valid_csv),
        batch_size=args.batch_size, num_workers=4
    )
    
    # Device
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}\n")
    
    # Train models
    results = {}
    
    if args.model in ['vit', 'both']:
        trainer_vit, history_vit = train_baseline(
            'vit', dataloaders, device,
            num_epochs=args.epochs, learning_rate=args.lr,
            save_dir=str(checkpoint_dir)
        )
        results['vit'] = history_vit
    
    if args.model in ['wav2vec2', 'both']:
        trainer_wav2vec2, history_wav2vec2 = train_baseline(
            'wav2vec2', dataloaders, device,
            num_epochs=args.epochs, learning_rate=args.lr,
            save_dir=str(checkpoint_dir)
        )
        results['wav2vec2'] = history_wav2vec2
    
    # Save results
    if not results:
        print("No models trained!")
        return
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(item) for item in obj]
        return obj
    
    results_json = convert_for_json(results)
    
    results_path = checkpoint_dir / 'training_results.json'
    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"Results saved to {results_path}")


if __name__ == '__main__':
    main()
