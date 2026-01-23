#!/usr/bin/env python3
"""
Script to combine dataset_01.csv and demo_dataset_1200.csv into dataset_final.csv
"""
import pandas as pd
import os

def combine_datasets():
    # Set the data directory path
    data_dir = os.path.join('data', 'datasets')
    
    # File paths
    dataset1_path = os.path.join(data_dir, 'dataset_01.csv')
    dataset2_path = os.path.join(data_dir, 'demo_dataset_1200.csv')
    output_path = os.path.join(data_dir, 'dataset_final.csv')
    
    # Check if input files exist
    if not os.path.exists(dataset1_path):
        raise FileNotFoundError(f"File not found: {dataset1_path}")
    if not os.path.exists(dataset2_path):
        raise FileNotFoundError(f"File not found: {dataset2_path}")
    
    print(f"Reading {dataset1_path}...")
    # Read the first dataset
    df1 = pd.read_csv(dataset1_path)
    print(f"Dataset 1 shape: {df1.shape}")
    print(f"Dataset 1 columns: {list(df1.columns)}")
    
    print(f"\nReading {dataset2_path}...")
    # Read the second dataset
    df2 = pd.read_csv(dataset2_path)
    print(f"Dataset 2 shape: {df2.shape}")
    print(f"Dataset 2 columns: {list(df2.columns)}")
    
    # Check if columns match
    if list(df1.columns) == list(df2.columns):
        print("\n✓ Columns match between datasets")
    else:
        print("\n⚠ Warning: Columns don't match exactly")
        print(f"Dataset 1 columns: {list(df1.columns)}")
        print(f"Dataset 2 columns: {list(df2.columns)}")
    
    # Combine the datasets
    print("\nCombining datasets...")
    combined_df = pd.concat([df1, df2], ignore_index=True)
    
    print(f"Combined dataset shape: {combined_df.shape}")
    print(f"Total rows: {len(combined_df)}")
    
    # Check for duplicates
    duplicates = combined_df.duplicated().sum()
    print(f"Number of duplicate rows: {duplicates}")
    
    # Remove duplicates if any
    if duplicates > 0:
        print("Removing duplicate rows...")
        combined_df = combined_df.drop_duplicates()
        print(f"Dataset shape after removing duplicates: {combined_df.shape}")
    
    # Save the combined dataset
    print(f"\nSaving combined dataset to {output_path}...")
    combined_df.to_csv(output_path, index=False)
    
    print(f"✓ Successfully created {output_path}")
    print(f"Final dataset contains {len(combined_df)} rows and {len(combined_df.columns)} columns")
    
    # Display some basic statistics
    print("\n=== Dataset Summary ===")
    print(f"Columns: {list(combined_df.columns)}")
    print(f"Data types:\n{combined_df.dtypes}")
    
    if 'job_category' in combined_df.columns:
        print(f"\nJob categories distribution:")
        print(combined_df['job_category'].value_counts().head(10))

if __name__ == "__main__":
    combine_datasets()