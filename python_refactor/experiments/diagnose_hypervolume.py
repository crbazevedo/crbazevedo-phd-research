#!/usr/bin/env python3
"""
Diagnostic script to check reference points and ROI/risk values
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.algorithms.sms_emoa import SMSEMOA
from src.portfolio.portfolio import Portfolio
from src.algorithms.solution import Solution

def diagnose_hypervolume():
    """Diagnose hypervolume calculation issues"""
    
    print("=== Hypervolume Diagnosis ===")
    
    # Test with different reference points
    reference_points = [
        (0.0, 1.0),    # Current default
        (-0.1, 0.1),   # More realistic for returns
        (-0.2, 0.2),   # Even more realistic
        (-0.5, 0.5),   # Very wide range
    ]
    
    # Create some test solutions with realistic ROI/risk values
    test_solutions = []
    for i in range(10):
        solution = Solution(num_assets=5)
        
        # Set realistic ROI and risk values (typical for financial data)
        solution.P.ROI = np.random.uniform(-0.15, 0.15)  # -15% to +15% ROI
        solution.P.risk = np.random.uniform(0.01, 0.25)  # 1% to 25% risk
        solution.stability = np.random.uniform(0.5, 1.0)  # Stability factor
        
        test_solutions.append(solution)
    
    print(f"\nTest solutions created:")
    for i, sol in enumerate(test_solutions):
        print(f"Solution {i}: ROI={sol.P.ROI:.4f}, Risk={sol.P.risk:.4f}, Stability={sol.stability:.4f}")
    
    # Test each reference point configuration
    for r1, r2 in reference_points:
        print(f"\n--- Testing Reference Points: R1={r1}, R2={r2} ---")
        
        # Create SMS-EMOA with these reference points
        sms_emoa = SMSEMOA(
            population_size=10,
            generations=1,
            reference_point_1=r1,
            reference_point_2=r2
        )
        
        # Set population
        sms_emoa.population = test_solutions.copy()
        
        # Compute hypervolume contributions
        sms_emoa._compute_hypervolume_contributions()
        
        # Check results
        total_hv = 0
        for i, sol in enumerate(sms_emoa.population):
            hv = getattr(sol, 'hypervolume_contribution', 0.0)
            total_hv += hv
            print(f"Solution {i}: HV = {hv:.6f}")
        
        print(f"Total Hypervolume: {total_hv:.6f}")
        
        # Check if any solutions have positive hypervolume
        positive_hv_count = sum(1 for sol in sms_emoa.population 
                              if getattr(sol, 'hypervolume_contribution', 0.0) > 0)
        print(f"Solutions with positive HV: {positive_hv_count}/10")
    
    # Analyze the issue
    print(f"\n=== Analysis ===")
    print("The issue is likely that:")
    print("1. ROI values are typically small (e.g., -0.15 to 0.15)")
    print("2. Risk values are typically small (e.g., 0.01 to 0.25)")
    print("3. With R1=0.0, R2=1.0:")
    print("   - ROI - R1 = ROI - 0.0 = ROI (small values)")
    print("   - R2 - Risk = 1.0 - Risk (large values)")
    print("   - But if ROI is negative, the product becomes negative")
    print("   - And if ROI is very small positive, the product is very small")
    
    # Suggest better reference points
    print(f"\n=== Suggested Reference Points ===")
    print("For financial data, better reference points might be:")
    print("R1 = -0.2 (minimum expected ROI)")
    print("R2 = 0.3 (maximum acceptable risk)")
    print("This ensures ROI - R1 is always positive and R2 - Risk is reasonable")

if __name__ == "__main__":
    diagnose_hypervolume() 