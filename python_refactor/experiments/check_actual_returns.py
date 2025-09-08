#!/usr/bin/env python3
"""
Check the actual returns being used in the experiment
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
    for i, file_path in enumerate(csv_files[:20]):
        try:
            df = pd.read_csv(file_path)
            df['Date'] = pd.to_datetime(df['Date'])
            
            # Fix inf values
            df['Adj Close'] = df['Adj Close'].replace([np.inf, -np.inf], np.nan)
            
            # Fix zero and negative values
            zero_mask = (df['Adj Close'] == 0) | (df['Adj Close'] < 0)
            if zero_mask.any():
                print(f"FTSE_ASSET_{i+1:02d}: Fixing {zero_mask.sum()} problematic values")
                df['Adj Close'] = df['Adj Close'].ffill().bfill().fillna(1.0)
            
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
    
    print(f"\nCombined data: {returns.shape[0]} days, {returns.shape[1]} assets")
    print(f"Date range: {returns.index.min()} to {returns.index.max()}")
    
    return returns

def check_period_returns():
    """Check what the actual period returns look like"""
    
    returns_data = load_existing_ftse_data()
    
    print("\n=== Checking Actual Period Returns ===")
    
    historical_days = 120
    stride_days = 30
    total_days = len(returns_data)
    n_periods = max(1, (total_days - historical_days) // stride_days)
    
    print(f"Total days: {total_days}")
    print(f"Number of periods: {n_periods}")
    
    # Test first few periods
    for period in range(min(5, n_periods)):
        start_idx = period * stride_days
        end_idx = start_idx + historical_days
        future_start = end_idx
        future_end = min(end_idx + 60, len(returns_data))
        
        if end_idx >= len(returns_data):
            break
        
        # Get data
        historical_data = returns_data.iloc[start_idx:end_idx]
        future_data = returns_data.iloc[future_start:future_end]
        
        print(f"\nPeriod {period + 1}:")
        print(f"  Historical data shape: {historical_data.shape}")
        print(f"  Future data shape: {future_data.shape}")
        print(f"  Historical data range: {historical_data.min().min():.6f} to {historical_data.max().max():.6f}")
        print(f"  Future data range: {future_data.min().min():.6f} to {future_data.max().max():.6f}")
        
        # Test equal weighted portfolio
        weights = np.ones(20) / 20
        
        if len(future_data) > 0:
            period_returns = future_data.values @ weights
            period_roi = np.mean(period_returns)
            
            print(f"  Equal weighted period ROI: {period_roi:.6f}")
            print(f"  Period returns range: {period_returns.min():.6f} to {period_returns.max():.6f}")
            print(f"  Period returns std: {period_returns.std():.6f}")
            
            # Check for extreme values
            extreme_positive = (period_returns > 0.1).sum()
            extreme_negative = (period_returns < -0.1).sum()
            print(f"  Returns > 10%: {extreme_positive}")
            print(f"  Returns < -10%: {extreme_negative}")
            
            # Check for very extreme values
            very_extreme_positive = (period_returns > 0.5).sum()
            very_extreme_negative = (period_returns < -0.5).sum()
            print(f"  Returns > 50%: {very_extreme_positive}")
            print(f"  Returns < -50%: {very_extreme_negative}")
            
            if very_extreme_positive > 0 or very_extreme_negative > 0:
                print(f"  WARNING: Found extreme returns!")
                extreme_indices = np.where((period_returns > 0.5) | (period_returns < -0.5))[0]
                for idx in extreme_indices:
                    print(f"    Day {idx}: {period_returns[idx]:.4f}")

def test_wealth_calculation_with_real_data():
    """Test wealth calculation with real data"""
    
    print("\n=== Testing Wealth Calculation with Real Data ===")
    
    returns_data = load_existing_ftse_data()
    
    historical_days = 120
    stride_days = 30
    total_days = len(returns_data)
    n_periods = max(1, (total_days - historical_days) // stride_days)
    
    # Test first 5 periods
    current_wealth = 100000.0
    wealth_history = [current_wealth]
    
    for period in range(min(5, n_periods)):
        start_idx = period * stride_days
        end_idx = start_idx + historical_days
        future_start = end_idx
        future_end = min(end_idx + 60, len(returns_data))
        
        if end_idx >= len(returns_data):
            break
        
        future_data = returns_data.iloc[future_start:future_end]
        
        if len(future_data) > 0:
            weights = np.ones(20) / 20
            period_returns = future_data.values @ weights
            period_roi = np.mean(period_returns)
            
            wealth_change = current_wealth * period_roi
            new_wealth = current_wealth + wealth_change
            
            print(f"Period {period + 1}: ROI = {period_roi:.6f}, Change = {wealth_change:.2f}, New Wealth = {new_wealth:,.2f}")
            
            wealth_history.append(new_wealth)
            current_wealth = new_wealth
    
    print(f"\nFinal wealth after 5 periods: {current_wealth:,.2f}")
    print(f"Total return: {(current_wealth - 100000.0) / 100000.0:.4f}")

if __name__ == "__main__":
    check_period_returns()
    test_wealth_calculation_with_real_data() 