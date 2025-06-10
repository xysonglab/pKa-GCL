import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import argparse


def split_and_save_datasets(input_file, output_dir, ratios):
    """
    Randomly split the input dataset by different ratios and save them

    Parameters:
        input_file: Path to input CSV file
        output_dir: Output directory
        ratios: List of split ratios, e.g. [0.1, 0.2, ..., 1.0]
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Read original data
    df = pd.read_csv(input_file)

    # Check required columns exist
    if 'smiles' not in df.columns or 'pka' not in df.columns:
        raise ValueError("Input file must contain both 'smiles' and 'pka' columns")

    # Shuffle data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save complete dataset
    full_path = os.path.join(output_dir, "full_dataset.csv")
    df.to_csv(full_path, index=False)
    print(f"Complete dataset saved to: {full_path} (Total {len(df)} samples)")

    # Split dataset by different ratios
    for ratio in ratios:
        # Calculate sample size for current ratio
        n_samples = int(len(df) * ratio)

        # Randomly select samples
        subset = df.iloc[:n_samples].copy()

        # Save subset
        subset_path = os.path.join(output_dir, f"train_{int(ratio * 100)}%.csv")
        subset.to_csv(subset_path, index=False)

        print(f"{int(ratio * 100)}% dataset saved to: {subset_path} (Total {len(subset)} samples)")


def main():
    # Set command line arguments
    parser = argparse.ArgumentParser(description='Randomly split dataset by ratios')
    parser.add_argument('--input', type=str, required=True, help='Path to input CSV file')
    parser.add_argument('--output_dir', type=str, default='split_datasets', help='Output directory')
    parser.add_argument('--start_ratio', type=float, default=0.1, help='Starting ratio')
    parser.add_argument('--end_ratio', type=float, default=1.0, help='Ending ratio')
    parser.add_argument('--step', type=float, default=0.1, help='Step size')

    args = parser.parse_args()

    # Generate ratio list
    ratios = np.arange(args.start_ratio, args.end_ratio + args.step, args.step)
    ratios = [round(r, 2) for r in ratios]  # Avoid floating point precision issues

    print(f"Will split dataset by following ratios: {ratios}")

    # Execute splitting and saving
    split_and_save_datasets(args.input, args.output_dir, ratios)


if __name__ == "__main__":
    main()