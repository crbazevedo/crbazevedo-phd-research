#!/usr/bin/env python3
"""
Comprehensive debug script for ASMS-EMOA algorithm
"""

import numpy as np
import pandas as pd
import sys
import os
import glob
import logging

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_existing_ftse_data():
    """Load existing FTSE data from the repository"""
    
    ftse_data_path = "../../ASMOO/executable/data/ftse-original"
    csv_files = glob.glob(os.path.join(ftse_data_path, "table (*).csv"))
    csv_files.sort()
    
    logger.info(f"Found {len(csv_files)} FTSE data files")
    
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
                logger.info(f"FTSE_ASSET_{i+1:02d}: Fixing {zero_mask.sum()} problematic values")
                df['Adj Close'] = df['Adj Close'].ffill().bfill().fillna(1.0)
            
            # Drop any remaining NaN values
            df = df.dropna(subset=['Adj Close'])
            
            asset_name = f'FTSE_ASSET_{i+1:02d}'
            asset_data = df[['Date', 'Adj Close']].copy()
            asset_data.columns = ['Date', asset_name]
            all_data.append(asset_data)
            logger.info(f"Loaded {asset_name}: {len(df)} rows")
        except Exception as e:
            logger.warning(f"Error loading {file_path}: {e}")
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
    
    logger.info(f"Combined data: {returns.shape[0]} days, {returns.shape[1]} assets")
    logger.info(f"Date range: {returns.index.min()} to {returns.index.max()}")
    
    return returns

def debug_portfolio_evaluation():
    """Debug portfolio evaluation process"""
    
    print("\n=== Debugging Portfolio Evaluation ===")
    
    from src.portfolio.portfolio import Portfolio
    from src.algorithms.solution import Solution
    
    # Load data
    returns_data = load_existing_ftse_data()
    
    # Get first period data
    historical_days = 120
    start_idx = 0
    end_idx = start_idx + historical_days
    historical_data = returns_data.iloc[start_idx:end_idx]
    
    print(f"Historical data shape: {historical_data.shape}")
    print(f"Historical data range: {historical_data.min().min():.6f} to {historical_data.max().max():.6f}")
    
    # Set up portfolio static variables (both robust and non-robust)
    Portfolio.mean_ROI = historical_data.mean().values
    Portfolio.covariance = historical_data.cov().values
    Portfolio.median_ROI = historical_data.median().values  # Fix: use median per asset, not scalar
    Portfolio.robust_covariance = historical_data.cov().values
    
    print(f"Portfolio static variables:")
    print(f"  Mean ROI shape: {Portfolio.mean_ROI.shape}")
    print(f"  Mean ROI range: {Portfolio.mean_ROI.min():.6f} to {Portfolio.mean_ROI.max():.6f}")
    print(f"  Median ROI: {Portfolio.median_ROI.min():.6f} to {Portfolio.median_ROI.max():.6f}")
    print(f"  Covariance shape: {Portfolio.covariance.shape}")
    print(f"  Robust covariance shape: {Portfolio.robust_covariance.shape}")
    
    # Create test solutions
    num_assets = 20
    test_solutions = []
    
    for i in range(3):
        # Create solution
        solution = Solution(num_assets)
        
        # Set different weights for testing
        if i == 0:
            # Equal weighted
            solution.P.investment = np.ones(num_assets) / num_assets
        elif i == 1:
            # Random weights
            solution.P.investment = np.random.dirichlet(np.ones(num_assets))
        else:
            # Concentrated weights
            weights = np.zeros(num_assets)
            weights[0] = 0.8
            weights[1:5] = 0.05
            solution.P.investment = weights
        
        # Evaluate the solution
        try:
            # This should compute ROI and risk
            Portfolio.compute_efficiency(solution.P)
            
            print(f"Solution {i+1}:")
            print(f"  Weights sum: {np.sum(solution.P.investment):.6f}")
            print(f"  Weights range: {np.min(solution.P.investment):.6f} to {np.max(solution.P.investment):.6f}")
            print(f"  ROI: {solution.P.ROI:.6f}")
            print(f"  Risk: {solution.P.risk:.6f}")
            
            test_solutions.append(solution)
            
        except Exception as e:
            print(f"❌ Solution {i+1} evaluation failed: {e}")
            import traceback
            traceback.print_exc()
    
    return test_solutions

def debug_sms_emoa_optimization():
    """Debug SMS-EMOA optimization process"""
    
    print("\n=== Debugging SMS-EMOA Optimization ===")
    
    from src.algorithms.sms_emoa import SMSEMOA
    from src.portfolio.portfolio import Portfolio
    
    # Load data
    returns_data = load_existing_ftse_data()
    
    # Get first period data
    historical_days = 120
    start_idx = 0
    end_idx = start_idx + historical_days
    historical_data = returns_data.iloc[start_idx:end_idx]
    
    print(f"Historical data shape: {historical_data.shape}")
    print(f"Historical data range: {historical_data.min().min():.6f} to {historical_data.max().max():.6f}")
    
    # Set up portfolio static variables (both robust and non-robust)
    Portfolio.mean_ROI = historical_data.mean().values
    Portfolio.covariance = historical_data.cov().values
    Portfolio.median_ROI = historical_data.median().values  # Fix: use median per asset, not scalar
    Portfolio.robust_covariance = historical_data.cov().values
    
    print(f"Portfolio static variables set:")
    print(f"  Mean ROI shape: {Portfolio.mean_ROI.shape}")
    print(f"  Mean ROI range: {Portfolio.mean_ROI.min():.6f} to {Portfolio.mean_ROI.max():.6f}")
    print(f"  Median ROI range: {Portfolio.median_ROI.min():.6f} to {Portfolio.median_ROI.max():.6f}")
    print(f"  Covariance shape: {Portfolio.covariance.shape}")
    print(f"  Robust covariance shape: {Portfolio.robust_covariance.shape}")
    
    # Initialize SMS-EMOA
    sms_emoa = SMSEMOA(
        population_size=10,  # Small population for debugging
        generations=5,       # Few generations for debugging
        reference_point_1=-0.2,
        reference_point_2=0.3
    )
    
    # Create data dictionary in the correct format
    data_dict = {
        'returns': historical_data.values,
        'num_assets': len(historical_data.columns),
        'anticipation_horizon': 1
    }
    
    print(f"Data dictionary created:")
    print(f"  Returns shape: {data_dict['returns'].shape}")
    print(f"  Num assets: {data_dict['num_assets']}")
    print(f"  Anticipation horizon: {data_dict['anticipation_horizon']}")
    
    # Run optimization
    try:
        pareto_frontier = sms_emoa.run(data_dict)
        print(f"✅ Optimization completed successfully")
        print(f"  Pareto frontier size: {len(pareto_frontier)}")
        
        # Analyze solutions
        for i, solution in enumerate(pareto_frontier[:3]):  # First 3 solutions
            print(f"  Solution {i+1}:")
            print(f"    ROI: {solution.P.ROI:.6f}")
            print(f"    Risk: {solution.P.risk:.6f}")
            print(f"    Weights sum: {np.sum(solution.P.investment):.6f}")
            print(f"    Weights range: {np.min(solution.P.investment):.6f} to {np.max(solution.P.investment):.6f}")
            print(f"    Hypervolume contribution: {getattr(solution, 'hypervolume_contribution', 'N/A')}")
        
        return pareto_frontier
        
    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def debug_portfolio_performance():
    """Debug portfolio performance calculation"""
    
    print("\n=== Debugging Portfolio Performance ===")
    
    from src.portfolio.portfolio import Portfolio
    from src.algorithms.solution import Solution
    
    # Load data
    returns_data = load_existing_ftse_data()
    
    # Get test data
    historical_days = 120
    start_idx = 0
    end_idx = start_idx + historical_days
    future_start = end_idx
    future_end = min(end_idx + 60, len(returns_data))
    
    historical_data = returns_data.iloc[start_idx:end_idx]
    future_data = returns_data.iloc[future_start:future_end]
    
    print(f"Test data:")
    print(f"  Historical shape: {historical_data.shape}")
    print(f"  Future shape: {future_data.shape}")
    
    # Create test portfolios
    test_portfolios = []
    
    # 1. Equal weighted
    portfolio_equal = Portfolio(20)
    portfolio_equal.investment = np.ones(20) / 20
    test_portfolios.append(("Equal Weighted", portfolio_equal))
    
    # 2. Random weights
    portfolio_random = Portfolio(20)
    portfolio_random.investment = np.random.dirichlet(np.ones(20))
    test_portfolios.append(("Random Weights", portfolio_random))
    
    # 3. Concentrated weights (potential issue)
    portfolio_concentrated = Portfolio(20)
    weights_concentrated = np.zeros(20)
    weights_concentrated[0] = 0.8  # 80% in first asset
    weights_concentrated[1:5] = 0.05  # 5% each in next 4 assets
    portfolio_concentrated.investment = weights_concentrated
    test_portfolios.append(("Concentrated Weights", portfolio_concentrated))
    
    # Test each portfolio
    for name, portfolio in test_portfolios:
        print(f"\n{name}:")
        print(f"  Weights sum: {np.sum(portfolio.investment):.6f}")
        print(f"  Weights range: {np.min(portfolio.investment):.6f} to {np.max(portfolio.investment):.6f}")
        
        # Calculate performance
        if len(future_data) > 0:
            period_returns = future_data.values @ portfolio.investment
            
            # Cap extreme returns
            period_returns = np.clip(period_returns, -0.20, 0.20)
            
            period_roi = np.mean(period_returns)
            
            print(f"  Period ROI: {period_roi:.6f}")
            print(f"  Returns range: {period_returns.min():.6f} to {period_returns.max():.6f}")
            print(f"  Returns std: {period_returns.std():.6f}")
            
            # Test wealth calculation
            initial_wealth = 100000.0
            wealth_change = initial_wealth * period_roi
            new_wealth = initial_wealth + wealth_change
            
            print(f"  Wealth change: {wealth_change:.2f}")
            print(f"  New wealth: {new_wealth:,.2f}")

def debug_real_experiment_step():
    """Debug a single step of the real experiment"""
    
    print("\n=== Debugging Real Experiment Step ===")
    
    from src.algorithms.sms_emoa import SMSEMOA
    from src.portfolio.portfolio import Portfolio
    from src.algorithms.solution import Solution
    
    # Load data
    returns_data = load_existing_ftse_data()
    
    # Get first period data
    historical_days = 120
    start_idx = 0
    end_idx = start_idx + historical_days
    future_start = end_idx
    future_end = min(end_idx + 60, len(returns_data))
    
    historical_data = returns_data.iloc[start_idx:end_idx]
    future_data = returns_data.iloc[future_start:future_end]
    
    print(f"Data setup:")
    print(f"  Historical data: {historical_data.shape}")
    print(f"  Future data: {future_data.shape}")
    
    # Set up portfolio static variables (both robust and non-robust)
    Portfolio.mean_ROI = historical_data.mean().values
    Portfolio.covariance = historical_data.cov().values
    Portfolio.median_ROI = historical_data.median().values  # Fix: use median per asset, not scalar
    Portfolio.robust_covariance = historical_data.cov().values
    
    print(f"Portfolio static variables:")
    print(f"  Mean ROI shape: {Portfolio.mean_ROI.shape}")
    print(f"  Mean ROI range: {Portfolio.mean_ROI.min():.6f} to {Portfolio.mean_ROI.max():.6f}")
    print(f"  Median ROI range: {Portfolio.median_ROI.min():.6f} to {Portfolio.median_ROI.max():.6f}")
    print(f"  Covariance shape: {Portfolio.covariance.shape}")
    print(f"  Robust covariance shape: {Portfolio.robust_covariance.shape}")
    
    # Initialize SMS-EMOA
    sms_emoa = SMSEMOA(
        population_size=10,  # Small for debugging
        generations=5,       # Small for debugging
        reference_point_1=-0.2,
        reference_point_2=0.3
    )
    
    # Create data dictionary
    data_dict = {
        'returns': historical_data.values,
        'num_assets': len(historical_data.columns),
        'anticipation_horizon': 1
    }
    
    # Run optimization
    try:
        pareto_frontier = sms_emoa.run(data_dict)
        print(f"✅ Optimization completed")
        print(f"  Pareto frontier size: {len(pareto_frontier)}")
        
        # Test portfolio selection
        if pareto_frontier:
            # Select first solution for testing
            selected_solution = pareto_frontier[0]
            
            print(f"Selected solution:")
            print(f"  ROI: {selected_solution.P.ROI:.6f}")
            print(f"  Risk: {selected_solution.P.risk:.6f}")
            print(f"  Weights sum: {np.sum(selected_solution.P.investment):.6f}")
            print(f"  Weights range: {np.min(selected_solution.P.investment):.6f} to {np.max(selected_solution.P.investment):.6f}")
            
            # Calculate performance
            if len(future_data) > 0:
                period_returns = future_data.values @ selected_solution.P.investment
                
                # Cap extreme returns
                period_returns = np.clip(period_returns, -0.20, 0.20)
                
                period_roi = np.mean(period_returns)
                
                print(f"Performance calculation:")
                print(f"  Period ROI: {period_roi:.6f}")
                print(f"  Returns range: {period_returns.min():.6f} to {period_returns.max():.6f}")
                print(f"  Returns std: {period_returns.std():.6f}")
                
                # Test wealth calculation
                initial_wealth = 100000.0
                wealth_change = initial_wealth * period_roi
                new_wealth = initial_wealth + wealth_change
                
                print(f"  Wealth change: {wealth_change:.2f}")
                print(f"  New wealth: {new_wealth:,.2f}")
        
    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main debug function"""
    
    print("=== Comprehensive ASMS-EMOA Debug ===")
    
    # 1. Debug portfolio evaluation
    test_solutions = debug_portfolio_evaluation()
    
    # 2. Debug portfolio performance
    debug_portfolio_performance()
    
    # 3. Debug SMS-EMOA optimization
    pareto_frontier = debug_sms_emoa_optimization()
    
    # 4. Debug real experiment step
    debug_real_experiment_step()
    
    print("\n=== Debug Complete ===")

if __name__ == "__main__":
    main() 