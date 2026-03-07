"""
Inference script for baseline models.
Generates predictions on test set or any split.
"""

import argparse
import os
import pickle
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error

from src.baselines import VitGRUBaseline, Wav2Vec2LinearBaseline
from src.dataloaders import EmbeddingDataset


def evaluate_baseline(model_path, model_name, embedding_dir, csv_path, device='cuda'):
    """
    Evaluate a trained baseline model.
    
    Args:
        model_path: Path to trained model checkpoint
        model_name: 'vit' or 'wav2vec2'
        embedding_dir: Directory with embedding files
        csv_path: CSV with filenames and labels
        device: torch device
    
    Returns:
        dict with metrics and predictions
    """
    
    # Load model
    if model_name == 'vit':
        model = VitGRUBaseline(input_dim=384, hidden_dim=256, num_layers=3, output_dim=6)
    elif model_name == 'wav2vec2':
        model = Wav2Vec2LinearBaseline(input_dim=768, output_dim=6)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Load data
    dataset = EmbeddingDataset(embedding_dir, csv_path)
    
    all_preds = []
    all_labels = []
    filenames = []
    
    print(f"Running inference ({model_name})...")
    
    with torch.no_grad():
        for idx in range(len(dataset)):
            embedding, labels = dataset[idx]
            embedding = embedding.unsqueeze(0).to(device)  # Add batch dim
            
            logits = model(embedding)
            all_preds.append(logits.cpu().numpy()[0])
            all_labels.append(labels.numpy())
            filenames.append(dataset.samples.iloc[idx]['Filename'])
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Compute metrics
    mse = mean_squared_error(all_labels, all_preds)
    mae = mean_absolute_error(all_labels, all_preds)
    flat_preds = all_preds.flatten()
    flat_labels = all_labels.flatten()
    pearson, _ = pearsonr(flat_labels, flat_preds)
    
    results = {
        'mse': mse,
        'mae': mae,
        'pearson': pearson,
        'predictions': all_preds,
        'labels': all_labels,
        'filenames': filenames
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate baseline models')
    parser.add_argument('--model', choices=['vit', 'wav2vec2', 'both'], default='both',
                        help='Which model to evaluate')
    parser.add_argument('--split', choices=['train', 'valid', 'test'], default='valid',
                        help='Which split to evaluate on')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda', help='Device')
    args = parser.parse_args()
    
    # Paths
    base_dir = Path('/home/dinithi/Documents/EMI_Challenge')
    vit_dir = base_dir / 'results/checkpoints/baseline/vit'
    wav2vec2_dir = base_dir / 'results/checkpoints/baseline/wav2vec2'
    checkpoint_dir = base_dir / 'results/checkpoints/trained'
    
    split_csv = base_dir / f'data/splits/{args.split}_split.csv'
    if not split_csv.exists():
        print(f"Split CSV not found: {split_csv}")
        return
    
    device = torch.device(args.device if args.device == 'cpu' or torch.cuda.is_available() else 'cpu')
    
    results = {}
    
    if args.model in ['vit', 'both']:
        vit_checkpoint = checkpoint_dir / 'best_vit.pt'
        if vit_checkpoint.exists():
            print(f"\nEvaluating ViT model on {args.split} split...")
            vit_results = evaluate_baseline(
                str(vit_checkpoint), 'vit', str(vit_dir), str(split_csv), str(device)
            )
            results['vit'] = vit_results
            print(f"  MSE: {vit_results['mse']:.4f}")
            print(f"  MAE: {vit_results['mae']:.4f}")
            print(f"  Pearson: {vit_results['pearson']:.4f}")
        else:
            print(f"ViT checkpoint not found: {vit_checkpoint}")
    
    if args.model in ['wav2vec2', 'both']:
        wav2vec2_checkpoint = checkpoint_dir / 'best_wav2vec2.pt'
        if wav2vec2_checkpoint.exists():
            print(f"\nEvaluating Wav2Vec2 model on {args.split} split...")
            wav2vec2_results = evaluate_baseline(
                str(wav2vec2_checkpoint), 'wav2vec2', str(wav2vec2_dir), str(split_csv), str(device)
            )
            results['wav2vec2'] = wav2vec2_results
            print(f"  MSE: {wav2vec2_results['mse']:.4f}")
            print(f"  MAE: {wav2vec2_results['mae']:.4f}")
            print(f"  Pearson: {wav2vec2_results['pearson']:.4f}")
        else:
            print(f"Wav2Vec2 checkpoint not found: {wav2vec2_checkpoint}")
    
    # Save predictions
    if results:
        output_dir = base_dir / f'results/predictions/{args.split}'
        os.makedirs(output_dir, exist_ok=True)
        
        for model_name, model_results in results.items():
            # Save predictions as CSV
            pred_df = pd.DataFrame({
                'Filename': model_results['filenames'],
                'Pred_Admiration': model_results['predictions'][:, 0],
                'Pred_Amusement': model_results['predictions'][:, 1],
                'Pred_Determination': model_results['predictions'][:, 2],
                'Pred_Empathic_Pain': model_results['predictions'][:, 3],
                'Pred_Excitement': model_results['predictions'][:, 4],
                'Pred_Joy': model_results['predictions'][:, 5],
            })
            pred_df.to_csv(output_dir / f'{model_name}_predictions.csv', index=False)
            print(f"\nPredictions saved: {output_dir / f'{model_name}_predictions.csv'}")


if __name__ == '__main__':
    main()
