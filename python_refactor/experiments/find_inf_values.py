#!/usr/bin/env python3
"""
Find and fix inf values in the FTSE data
"""

import numpy as np
import pandas as pd
import sys
import os
import glob

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def find_inf_values():
    """Find where the inf values are coming from"""
    
    ftse_data_path = "../../ASMOO/executable/data/ftse-original"
    csv_files = glob.glob(os.path.join(ftse_data_path, "table (*).csv"))
    csv_files.sort()
    
    print("=== Finding Inf Values ===")
    
    for i, file_path in enumerate(csv_files[:20]):
        try:
            df = pd.read_csv(file_path)
            df['Date'] = pd.to_datetime(df['Date'])
            
            # Check for inf values in Adj Close
            inf_mask = np.isinf(df['Adj Close'])
            if inf_mask.any():
                print(f"FTSE_ASSET_{i+1:02d}: Found {inf_mask.sum()} inf values")
                inf_rows = df[inf_mask]
                print(f"  Inf rows: {inf_rows[['Date', 'Adj Close']].to_string()}")
            
            # Check for zero values that might cause inf in pct_change
            zero_mask = (df['Adj Close'] == 0)
            if zero_mask.any():
                print(f"FTSE_ASSET_{i+1:02d}: Found {zero_mask.sum()} zero values")
                zero_rows = df[zero_mask]
                print(f"  Zero rows: {zero_rows[['Date', 'Adj Close']].to_string()}")
            
            # Check for negative values
            neg_mask = (df['Adj Close'] < 0)
            if neg_mask.any():
                print(f"FTSE_ASSET_{i+1:02d}: Found {neg_mask.sum()} negative values")
                neg_rows = df[neg_mask]
                print(f"  Negative rows: {neg_rows[['Date', 'Adj Close']].to_string()}")
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

def fix_data_issues():
    """Create a fixed version of the data loading function"""
    
    ftse_data_path = "../../ASMOO/executable/data/ftse-original"
    csv_files = glob.glob(os.path.join(ftse_data_path, "table (*).csv"))
    csv_files.sort()
    
    print("\n=== Loading Data with Fixes ===")
    
    all_data = []
    for i, file_path in enumerate(csv_files[:20]):
        try:
            df = pd.read_csv(file_path)
            df['Date'] = pd.to_datetime(df['Date'])
            
            # Fix inf values
            df['Adj Close'] = df['Adj Close'].replace([np.inf, -np.inf], np.nan)
            
            # Fix zero values (replace with previous value or 1.0)
            zero_mask = (df['Adj Close'] == 0) | (df['Adj Close'] < 0)
            if zero_mask.any():
                print(f"FTSE_ASSET_{i+1:02d}: Fixing {zero_mask.sum()} problematic values")
                # Forward fill, then backward fill, then fill remaining with 1.0
                df['Adj Close'] = df['Adj Close'].fillna(method='ffill').fillna(method='bfill').fillna(1.0)
            
            # Drop any remaining NaN values
            df = df.dropna(subset=['Adj Close'])
            
            asset_name = f'FTSE_ASSET_{i+1:02d}'
            asset_data = df[['Date', 'Adj Close']].copy()
            asset_data.columns = ['Date', asset_name]
            all_data.append(asset_data)
            print(f"Loaded {asset_name}: {len(df)} rows")
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue
    
    if not all_data:
        raise ValueError("No valid FTSE data files found")
    
    # Merge all assets
    merged_data = all_data[0]
    for asset_data in all_data[1:]:
        merged_data = merged_data.merge(asset_data, on='Date', how='inner')
    
    merged_data.set_index('Date', inplace=True)
    
    # Calculate returns with additional safety checks
    returns = merged_data.pct_change()
    
    # Replace inf values in returns with NaN
    returns = returns.replace([np.inf, -np.inf], np.nan)
    
    # Drop rows with any NaN values
    returns = returns.dropna()
    
    print(f"\nFixed data: {returns.shape[0]} days, {returns.shape[1]} assets")
    print(f"Date range: {returns.index.min()} to {returns.index.max()}")
    print(f"NaN values in returns: {returns.isna().sum().sum()}")
    print(f"Inf values in returns: {np.isinf(returns.values).sum()}")
    print(f"Returns range: {returns.min().min():.6f} to {returns.max().max():.6f}")
    
    return returns

def main():
    """Main function"""
    # Find inf values
    find_inf_values()
    
    # Fix data issues
    returns_data = fix_data_issues()
    
    # Test the fixed data
    print("\n=== Testing Fixed Data ===")
    
    historical_days = 120
    stride_days = 30
    total_days = len(returns_data)
    n_periods = max(1, (total_days - historical_days) // stride_days)
    
    print(f"Total days: {total_days}")
    print(f"Number of periods: {n_periods}")
    
    # Test first period
    start_idx = 0
    end_idx = start_idx + historical_days
    future_start = end_idx
    future_end = min(end_idx + 60, len(returns_data))
    
    historical_data = returns_data.iloc[start_idx:end_idx]
    future_data = returns_data.iloc[future_start:future_end]
    
    print(f"\nPeriod 1 with fixed data:")
    print(f"  Historical data shape: {historical_data.shape}")
    print(f"  Future data shape: {future_data.shape}")
    print(f"  Historical data range: {historical_data.min().min():.6f} to {historical_data.max().max():.6f}")
    print(f"  Future data range: {future_data.min().min():.6f} to {future_data.max().max():.6f}")
    
    # Test equal weighted portfolio performance
    weights = np.ones(20) / 20
    if len(future_data) > 0:
        period_returns = future_data.values @ weights
        period_roi = np.mean(period_returns)
        print(f"  Equal weighted period ROI: {period_roi:.6f}")
        print(f"  Period returns range: {period_returns.min():.6f} to {period_returns.max():.6f}")
        print(f"  Any NaN in period returns: {np.isnan(period_returns).any()}")
        print(f"  Any Inf in period returns: {np.isinf(period_returns).any()}")

if __name__ == "__main__":
    main() 