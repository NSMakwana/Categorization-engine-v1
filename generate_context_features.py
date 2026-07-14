#!/usr/bin/env python3
"""
Generate context features (is_recurring, salary_probability) for new data.

This script enriches a dataset with contextual features needed for the enhanced ML model.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import pickle

# Configuration
RECURRING_KEYWORDS = {
    'SALARY', 'EMI', 'SUBSCRIPTION', 'AIRTEL', 'FLIPKART', 'AMAZON', 
    'NETFLIX', 'SPOTIFY', 'INSURANCE', 'ELECTRICITY', 'WATER', 'GAS',
    'RECHARGE', 'MOBILE', 'JIOPREPAID', 'PREPAID', 'BUPA', 'HEALTH',
    'UTILITY', 'BILL', 'TRANSFER', 'DEPOSIT', 'WITHDRAWAL', 'PETROL', 'FUEL'
}

SALARY_KEYWORDS = {
    'SALARY', 'PAYMENT', 'BRN/', 'RLACAD'
}

SALARY_AMOUNT_THRESHOLD = 5000  # Minimum amount for salary heuristic
RECURRING_SALARY_SCORE = 0.6    # Score for large recurring credit


def compute_is_recurring(narration):
    """Check if transaction is recurring."""
    if pd.isna(narration):
        return 0
    
    narration_upper = str(narration).upper()
    
    # Check for exact keyword matches (token-safe)
    for keyword in RECURRING_KEYWORDS:
        if keyword in narration_upper:
            return 1
    
    return 0


def compute_salary_probability(narration, amount=0, is_recurring=0):
    """
    Compute heuristic salary probability.
    
    Returns:
        1.0 if direct salary match
        0.6 if large credit + recurring
        0.0 otherwise
    """
    if pd.isna(narration):
        return 0.0
    
    narration_upper = str(narration).upper()
    
    # Signal 1: Direct salary keyword match
    if any(keyword in narration_upper for keyword in SALARY_KEYWORDS):
        return 1.0
    
    # Signal 2: Large credit + recurring (heuristic for salary)
    if amount >= SALARY_AMOUNT_THRESHOLD and is_recurring:
        return 0.6
    
    # Signal 3: No salary signal
    return 0.0


def generate_features(df):
    """Generate context features for the dataset."""
    print("Generating context features...")
    
    # Ensure required columns exist
    if 'Normalized Narration' not in df.columns:
        print("ERROR: 'Normalized Narration' column required")
        return None
    
    # Get amount column (try multiple possible names)
    amount_col = None
    for possible_col in ['Amount', 'amount', 'Debits', 'Credits', 'Value']:
        if possible_col in df.columns:
            amount_col = possible_col
            break
    
    if amount_col:
        amounts = pd.to_numeric(df[amount_col], errors='coerce').fillna(0).values
    else:
        print("WARNING: Amount column not found; using 0 for salary probability heuristic")
        amounts = np.zeros(len(df))
    
    # Generate is_recurring
    print("  Computing is_recurring...")
    df['is_recurring'] = df['Normalized Narration'].apply(compute_is_recurring)
    
    # Generate salary_probability
    print("  Computing salary_probability...")
    df['salary_probability'] = df.apply(
        lambda row: compute_salary_probability(
            row['Normalized Narration'],
            amount=amounts[row.name],
            is_recurring=row['is_recurring']
        ),
        axis=1
    )
    
    print(f"✓ Features generated")
    print(f"  - is_recurring: {df['is_recurring'].sum()} recurring transactions")
    print(f"  - salary_probability: mean={df['salary_probability'].mean():.3f}")
    
    return df


def save_features(df, output_path):
    """Save dataset with features."""
    df.to_csv(output_path, index=False)
    print(f"✓ Saved to {output_path}")


def main():
    """Main workflow."""
    print("=" * 80)
    print("CONTEXT FEATURE GENERATION")
    print("=" * 80)
    
    # Ask for input file
    input_file = input("\nEnter path to your CSV file: ").strip()
    
    if not Path(input_file).exists():
        print(f"ERROR: File not found: {input_file}")
        return
    
    # Load data
    print(f"\nLoading {input_file}...")
    try:
        df = pd.read_csv(input_file)
        print(f"✓ Loaded {len(df)} rows, {len(df.columns)} columns")
    except Exception as e:
        print(f"ERROR: Failed to load file: {e}")
        return
    
    # Generate features
    df = generate_features(df)
    if df is None:
        return
    
    # Ask for output file
    default_output = Path(input_file).stem + "_with_features.csv"
    output_file = input(f"\nOutput file [{default_output}]: ").strip() or default_output
    
    # Save
    print(f"\nSaving to {output_file}...")
    save_features(df, output_file)
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Input:  {input_file}")
    print(f"Output: {output_file}")
    print(f"Rows:   {len(df)}")
    print(f"New columns: is_recurring, salary_probability")
    print("\nYou can now use this file with the enhanced ML model!")
    print("=" * 80)


if __name__ == "__main__":
    main()
