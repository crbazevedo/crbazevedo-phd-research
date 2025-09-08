#!/usr/bin/env python3
"""
Debug script to check for NaN and inf values in the experiment results
"""

import numpy as np
import pandas as pd
import sys
import os
import glob

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def load_existing_ftse_data():
    """Load existing FTSE data from the repository"""
    
    ftse_data_path = "../../ASMOO/executable/data/ftse-original"
    csv_files = glob.glob(os.path.join(ftse_data_path, "table (*).csv"))
    csv_files.sort()
    
    print(f"Found {len(csv_files)} FTSE data files")
    
    all_data = []
    for i, file_path in enumerate(csv_files[:20]):  # Use first 20 assets
        try:
            df = pd.read_csv(file_path)
            df['Date'] = pd.to_datetime(df['Date'])
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
    returns = merged_data.pct_change().dropna()
    
    print(f"Combined data: {returns.shape[0]} days, {returns.shape[1]} assets")
    print(f"Date range: {returns.index.min()} to {returns.index.max()}")
    
    # Check for NaN and inf values in returns
    print(f"\nNaN values in returns: {returns.isna().sum().sum()}")
    print(f"Inf values in returns: {np.isinf(returns.values).sum()}")
    print(f"Returns range: {returns.min().min():.6f} to {returns.max().max():.6f}")
    
    return returns

def test_wealth_calculation():
    """Test the wealth calculation logic"""
    
    print("\n=== Testing Wealth Calculation ===")
    
    # Test case 1: Normal returns
    current_wealth = 100000.0
    period_roi = 0.05  # 5% return
    wealth_change = current_wealth * period_roi
    new_wealth = current_wealth + wealth_change
    total_roi = (new_wealth - 100000.0) / 100000.0
    
    print(f"Test 1 - Normal returns:")
    print(f"  Current wealth: {current_wealth}")
    print(f"  Period ROI: {period_roi}")
    print(f"  Wealth change: {wealth_change}")
    print(f"  New wealth: {new_wealth}")
    print(f"  Total ROI: {total_roi}")
    
    # Test case 2: Zero returns
    period_roi = 0.0
    wealth_change = current_wealth * period_roi
    new_wealth = current_wealth + wealth_change
    total_roi = (new_wealth - 100000.0) / 100000.0
    
    print(f"\nTest 2 - Zero returns:")
    print(f"  Period ROI: {period_roi}")
    print(f"  Wealth change: {wealth_change}")
    print(f"  New wealth: {new_wealth}")
    print(f"  Total ROI: {total_roi}")
    
    # Test case 3: Negative returns
    period_roi = -0.02  # -2% return
    wealth_change = current_wealth * period_roi
    new_wealth = current_wealth + wealth_change
    total_roi = (new_wealth - 100000.0) / 100000.0
    
    print(f"\nTest 3 - Negative returns:")
    print(f"  Period ROI: {period_roi}")
    print(f"  Wealth change: {wealth_change}")
    print(f"  New wealth: {new_wealth}")
    print(f"  Total ROI: {total_roi}")
    
    # Test case 4: Very small returns
    period_roi = 1e-10
    wealth_change = current_wealth * period_roi
    new_wealth = current_wealth + wealth_change
    total_roi = (new_wealth - 100000.0) / 100000.0
    
    print(f"\nTest 4 - Very small returns:")
    print(f"  Period ROI: {period_roi}")
    print(f"  Wealth change: {wealth_change}")
    print(f"  New wealth: {new_wealth}")
    print(f"  Total ROI: {total_roi}")

def test_portfolio_weights():
    """Test portfolio weight calculations"""
    
    print("\n=== Testing Portfolio Weights ===")
    
    # Test equal weighted portfolio
    n_assets = 20
    weights = np.ones(n_assets) / n_assets
    print(f"Equal weighted portfolio:")
    print(f"  Weights sum: {np.sum(weights):.6f}")
    print(f"  Weights range: {weights.min():.6f} to {weights.max():.6f}")
    print(f"  Any NaN: {np.isnan(weights).any()}")
    print(f"  Any Inf: {np.isinf(weights).any()}")

def main():
    """Main debug function"""
    print("=== Debugging Real Data Experiment ===")
    
    # Load data
    returns_data = load_existing_ftse_data()
    
    # Test wealth calculation
    test_wealth_calculation()
    
    # Test portfolio weights
    test_portfolio_weights()
    
    # Test a few periods manually
    print("\n=== Testing Manual Period Calculation ===")
    
    historical_days = 120
    stride_days = 30
    total_days = len(returns_data)
    n_periods = max(1, (total_days - historical_days) // stride_days)
    
    print(f"Total days: {total_days}")
    print(f"Historical days: {historical_days}")
    print(f"Stride days: {stride_days}")
    print(f"Number of periods: {n_periods}")
    
    # Test first period
    start_idx = 0
    end_idx = start_idx + historical_days
    future_start = end_idx
    future_end = min(end_idx + 60, len(returns_data))
    
    historical_data = returns_data.iloc[start_idx:end_idx]
    future_data = returns_data.iloc[future_start:future_end]
    
    print(f"\nPeriod 1:")
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