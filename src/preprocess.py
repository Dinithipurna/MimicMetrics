"""
preprocess.py — Data Preprocessing for EMI Challenge
=====================================================
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Preprocess EMI data")
    parser.add_argument("--raw_dir", type=str, default="data/raw", help="Raw data directory")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="Output directory")
    args = parser.parse_args()
    
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("  EMI Challenge — Data Preprocessing")
    print("=" * 60)
    print(f"  Input:  {raw_dir}")
    print(f"  Output: {output_dir}")
    print("=" * 60)
    
    # TODO: Implement preprocessing
    pass


if __name__ == "__main__":
    main()
