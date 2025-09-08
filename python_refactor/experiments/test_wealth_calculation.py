#!/usr/bin/env python3
"""
Test wealth calculation logic
"""

import numpy as np

def test_wealth_calculation():
    """Test the wealth calculation logic step by step"""
    
    print("=== Testing Wealth Calculation Logic ===")
    
    # Initial wealth
    initial_wealth = 100000.0
    current_wealth = initial_wealth
    
    print(f"Initial wealth: {initial_wealth:,.2f}")
    
    # Test with realistic returns (daily returns around 0.001 to 0.002)
    test_returns = [0.001, -0.0005, 0.002, -0.001, 0.0015]
    
    print("\nTesting with realistic daily returns:")
    for i, daily_return in enumerate(test_returns):
        # Calculate wealth change
        wealth_change = current_wealth * daily_return
        new_wealth = current_wealth + wealth_change
        
        print(f"Day {i+1}: Return = {daily_return:.4f}, Change = {wealth_change:.2f}, New Wealth = {new_wealth:,.2f}")
        
        current_wealth = new_wealth
    
    print(f"\nFinal wealth after 5 days: {current_wealth:,.2f}")
    print(f"Total return: {(current_wealth - initial_wealth) / initial_wealth:.4f}")
    
    # Test with the problematic returns from our data
    print("\n=== Testing with problematic returns ===")
    
    # Let's simulate what might be happening with our data
    current_wealth = initial_wealth
    
    # Simulate 19 periods with 30-day rebalancing
    for period in range(19):
        # Simulate a period return (average of 30 daily returns)
        period_return = np.random.normal(0.001, 0.02)  # 0.1% mean, 2% std
        
        # Calculate wealth change
        wealth_change = current_wealth * period_return
        new_wealth = current_wealth + wealth_change
        
        if period < 5:  # Show first 5 periods
            print(f"Period {period+1}: Return = {period_return:.4f}, Change = {wealth_change:.2f}, New Wealth = {new_wealth:,.2f}")
        
        current_wealth = new_wealth
    
    print(f"\nFinal wealth after 19 periods: {current_wealth:,.2f}")
    print(f"Total return: {(current_wealth - initial_wealth) / initial_wealth:.4f}")
    
    # Test with very large returns (this might be our issue)
    print("\n=== Testing with very large returns (potential issue) ===")
    
    current_wealth = initial_wealth
    
    # Simulate with very large returns
    large_returns = [0.1, 0.2, -0.15, 0.3, 0.25]  # 10%, 20%, -15%, 30%, 25%
    
    for i, large_return in enumerate(large_returns):
        wealth_change = current_wealth * large_return
        new_wealth = current_wealth + wealth_change
        
        print(f"Period {i+1}: Return = {large_return:.2f}, Change = {wealth_change:,.2f}, New Wealth = {new_wealth:,.2f}")
        
        current_wealth = new_wealth
    
    print(f"\nFinal wealth with large returns: {current_wealth:,.2f}")
    print(f"Total return: {(current_wealth - initial_wealth) / initial_wealth:.4f}")

def test_period_roi_calculation():
    """Test how period ROI is calculated"""
    
    print("\n=== Testing Period ROI Calculation ===")
    
    # Simulate 30 days of daily returns
    daily_returns = np.random.normal(0.001, 0.02, 30)  # 30 days
    
    print(f"Daily returns range: {daily_returns.min():.4f} to {daily_returns.max():.4f}")
    print(f"Daily returns mean: {daily_returns.mean():.4f}")
    
    # Method 1: Simple average (what we're doing)
    period_roi_simple = np.mean(daily_returns)
    print(f"Period ROI (simple average): {period_roi_simple:.4f}")
    
    # Method 2: Compound return
    period_roi_compound = (1 + daily_returns).prod() - 1
    print(f"Period ROI (compound): {period_roi_compound:.4f}")
    
    # Test wealth calculation with both methods
    initial_wealth = 100000.0
    
    # Method 1: Simple average
    wealth_change_1 = initial_wealth * period_roi_simple
    final_wealth_1 = initial_wealth + wealth_change_1
    
    # Method 2: Compound
    final_wealth_2 = initial_wealth * (1 + period_roi_compound)
    
    print(f"\nWealth calculation comparison:")
    print(f"Method 1 (simple): {final_wealth_1:,.2f}")
    print(f"Method 2 (compound): {final_wealth_2:,.2f}")
    print(f"Difference: {abs(final_wealth_1 - final_wealth_2):,.2f}")

if __name__ == "__main__":
    test_wealth_calculation()
    test_period_roi_calculation() 