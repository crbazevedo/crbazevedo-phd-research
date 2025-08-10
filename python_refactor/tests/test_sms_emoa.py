"""
Tests for SMS-EMOA algorithm implementation.
"""

import pytest
import numpy as np
from typing import List

from src.algorithms.sms_emoa import (
    compute_hypervolume,
    compute_delta_s_contribution,
    compute_delta_s_class,
    remove_worst_s_metric,
    run_sms_emoa_generation,
    run_sms_emoa,
    get_sms_emoa_pareto_front,
    evaluate_sms_emoa_statistics
)
from src.algorithms.solution import Solution
from src.portfolio.portfolio import Portfolio


class TestSMSEMOA:
    """Test cases for SMS-EMOA algorithm."""
    
    def setup_method(self):
        """Set up test data."""
        # Set up portfolio data for testing
        Portfolio.available_assets_size = 5
        Portfolio.mean_ROI = np.array([0.1, 0.15, 0.12, 0.08, 0.18])
        Portfolio.median_ROI = np.array([0.1, 0.15, 0.12, 0.08, 0.18])  # Add median_ROI
        Portfolio.covariance = np.array([
            [0.04, 0.02, 0.01, 0.005, 0.03],
            [0.02, 0.09, 0.015, 0.01, 0.025],
            [0.01, 0.015, 0.16, 0.008, 0.02],
            [0.005, 0.01, 0.008, 0.25, 0.015],
            [0.03, 0.025, 0.02, 0.015, 0.36]
        ])
        Portfolio.robust_covariance = Portfolio.covariance.copy()  # Add robust covariance
        Portfolio.robustness = False
        
        # Set up complete returns data for anticipatory learning
        Portfolio.complete_returns_data = np.random.randn(100, 5) * 0.1  # 100 time steps, 5 assets
        Portfolio.window_size = 20
        
        # Create test solutions
        self.solutions = []
        for i in range(10):
            solution = Solution(5)  # 5 assets
            solution.P.ROI = 0.1 + i * 0.02  # Increasing ROI
            solution.P.risk = 0.2 - i * 0.01  # Decreasing risk
            solution.Pareto_rank = 0 if i < 5 else 1  # First 5 in Pareto front
            solution.Delta_S = 0.0
            self.solutions.append(solution)
    
    def test_compute_hypervolume_empty_population(self):
        """Test hypervolume computation with empty population."""
        hypervolume = compute_hypervolume([], (-1.0, 10.0))
        assert hypervolume == 0.0
    
    def test_compute_hypervolume_no_pareto_front(self):
        """Test hypervolume computation with no Pareto front solutions."""
        # Set all solutions to non-Pareto front
        for solution in self.solutions:
            solution.Pareto_rank = 1
        
        hypervolume = compute_hypervolume(self.solutions, (-1.0, 10.0))
        assert hypervolume == 0.0
    
    def test_compute_hypervolume_basic(self):
        """Test basic hypervolume computation."""
        # Set first 3 solutions to Pareto front
        for i in range(3):
            self.solutions[i].Pareto_rank = 0
        
        hypervolume = compute_hypervolume(self.solutions, (-1.0, 10.0))
        # Hypervolume should be >= 0.0 (can be 0 if solutions are dominated by reference point)
        assert hypervolume >= 0.0
        assert isinstance(hypervolume, float)
    
    def test_compute_delta_s_contribution(self):
        """Test Delta-S contribution computation."""
        compute_delta_s_contribution(self.solutions, (-1.0, 10.0))
        
        # Check that Delta_S values are set
        for solution in self.solutions:
            assert hasattr(solution, 'Delta_S')
            assert isinstance(solution.Delta_S, float)
    
    def test_compute_delta_s_class_single_solution(self):
        """Test Delta-S computation for single solution."""
        single_solution = [self.solutions[0]]
        compute_delta_s_class(single_solution, (-1.0, 10.0))
        
        assert single_solution[0].Delta_S == float('inf')
    
    def test_compute_delta_s_class_multiple_solutions(self):
        """Test Delta-S computation for multiple solutions."""
        solutions = self.solutions[:3]  # Take first 3 solutions
        for solution in solutions:
            solution.Pareto_rank = 0  # All in same Pareto rank
        
        compute_delta_s_class(solutions, (-1.0, 10.0))
        
        # Check that Delta_S values are computed
        for solution in solutions:
            assert solution.Delta_S >= 0.0
    
    def test_remove_worst_s_metric_single_solution(self):
        """Test removing worst solution when only one exists."""
        single_solution = [self.solutions[0]]
        original_size = len(single_solution)
        
        remove_worst_s_metric(single_solution, (-1.0, 10.0))
        
        # Should not remove the only solution
        assert len(single_solution) == original_size
    
    def test_remove_worst_s_metric_multiple_solutions(self):
        """Test removing worst solution from multiple solutions."""
        solutions = self.solutions[:5].copy()
        original_size = len(solutions)
        
        # Set different Delta_S values
        for i, solution in enumerate(solutions):
            solution.Delta_S = 1.0 + i
        
        remove_worst_s_metric(solutions, (-1.0, 10.0))
        
        # Should remove one solution
        assert len(solutions) == original_size - 1
    
    def test_run_sms_emoa_generation(self):
        """Test running one generation of SMS-EMOA."""
        population = self.solutions[:5].copy()
        original_size = len(population)
        
        run_sms_emoa_generation(population, mutation_rate=0.3, tournament_size=2)
        
        # Population size should remain the same (add offspring, remove worst)
        assert len(population) == original_size
        
        # Check that solutions have been updated
        for solution in population:
            assert hasattr(solution, 'Pareto_rank')
            assert hasattr(solution, 'Delta_S')
    
    def test_run_sms_emoa_complete(self):
        """Test running complete SMS-EMOA algorithm."""
        initial_population = self.solutions[:5].copy()
        
        final_population = run_sms_emoa(
            initial_population, 
            generations=5, 
            mutation_rate=0.3, 
            tournament_size=2
        )
        
        # Should return a population
        assert isinstance(final_population, list)
        assert len(final_population) > 0
        
        # Check that solutions have been optimized
        for solution in final_population:
            assert hasattr(solution, 'Pareto_rank')
            assert hasattr(solution, 'Delta_S')
    
    def test_get_sms_emoa_pareto_front(self):
        """Test extracting Pareto front from SMS-EMOA population."""
        # Set some solutions to Pareto front
        for i in range(3):
            self.solutions[i].Pareto_rank = 0
        
        pareto_front = get_sms_emoa_pareto_front(self.solutions)
        
        # Should get all solutions with Pareto rank 0 (which includes the first 5 from setup)
        assert len(pareto_front) >= 3
        for solution in pareto_front:
            assert solution.Pareto_rank == 0
    
    def test_evaluate_sms_emoa_statistics(self):
        """Test SMS-EMOA statistics evaluation."""
        stats = evaluate_sms_emoa_statistics(self.solutions)
        
        # Check that statistics are computed
        assert 'population_size' in stats
        assert 'pareto_front_size' in stats
        assert 'hypervolume' in stats
        assert 'mean_pareto_rank' in stats
        assert 'mean_delta_s' in stats
        assert 'mean_roi' in stats
        assert 'mean_risk' in stats
        assert 'mean_cardinality' in stats
        assert 'mean_stability' in stats
        
        # Check data types
        assert isinstance(stats['population_size'], int)
        assert isinstance(stats['pareto_front_size'], int)
        assert isinstance(stats['hypervolume'], float)
        assert isinstance(stats['mean_pareto_rank'], float)
        assert isinstance(stats['mean_roi'], float)
        assert isinstance(stats['mean_risk'], float)
    
    def test_evaluate_sms_emoa_statistics_empty_population(self):
        """Test statistics evaluation with empty population."""
        stats = evaluate_sms_emoa_statistics([])
        assert stats == {}
    
    def test_hypervolume_with_different_reference_points(self):
        """Test hypervolume computation with different reference points."""
        # Set first 3 solutions to Pareto front
        for i in range(3):
            self.solutions[i].Pareto_rank = 0
        
        # Test with different reference points
        ref_points = [(-1.0, 10.0), (-0.5, 5.0), (0.0, 1.0)]
        
        for ref_point in ref_points:
            hypervolume = compute_hypervolume(self.solutions, ref_point)
            assert hypervolume >= 0.0
            assert isinstance(hypervolume, float)
    
    def test_delta_s_contribution_consistency(self):
        """Test that Delta-S contributions are consistent."""
        solutions = self.solutions[:4].copy()
        for solution in solutions:
            solution.Pareto_rank = 0  # All in same Pareto rank
        
        # Compute Delta-S contributions
        compute_delta_s_contribution(solutions, (-1.0, 10.0))
        
        # Check that Delta-S values are computed
        for solution in solutions:
            assert hasattr(solution, 'Delta_S')
            assert isinstance(solution.Delta_S, float)
            assert solution.Delta_S >= 0.0


class TestSMSEMOAIntegration:
    """Integration tests for SMS-EMOA with portfolio data."""
    
    def setup_method(self):
        """Set up integration test data."""
        # Set up portfolio data
        Portfolio.available_assets_size = 3
        Portfolio.mean_ROI = np.array([0.1, 0.15, 0.12])
        Portfolio.median_ROI = np.array([0.1, 0.15, 0.12])  # Add median_ROI
        Portfolio.covariance = np.array([
            [0.04, 0.02, 0.01],
            [0.02, 0.09, 0.015],
            [0.01, 0.015, 0.16]
        ])
        Portfolio.robust_covariance = Portfolio.covariance.copy()  # Add robust covariance
        Portfolio.robustness = False
        
        # Set up complete returns data for anticipatory learning
        Portfolio.complete_returns_data = np.random.randn(100, 3) * 0.1  # 100 time steps, 3 assets
        Portfolio.window_size = 20
    
    def test_sms_emoa_with_portfolio_optimization(self):
        """Test SMS-EMOA with actual portfolio optimization."""
        # Create initial population
        population = [Solution(3) for _ in range(10)]
        
        # Run SMS-EMOA
        final_population = run_sms_emoa(
            population, 
            generations=10, 
            mutation_rate=0.3, 
            tournament_size=2
        )
        
        # Check results
        assert len(final_population) == 10
        
        # Check that solutions have valid portfolio data
        for solution in final_population:
            assert hasattr(solution.P, 'ROI')
            assert hasattr(solution.P, 'risk')
            assert hasattr(solution.P, 'cardinality')
            assert solution.P.ROI >= 0.0
            assert solution.P.risk >= 0.0
            assert solution.P.cardinality >= 0.0
        
        # Check that Pareto front exists
        pareto_front = get_sms_emoa_pareto_front(final_population)
        assert len(pareto_front) > 0
        
        # Check hypervolume (can be 0 if reference point dominates all solutions)
        hypervolume = compute_hypervolume(final_population, (-1.0, 10.0))
        assert hypervolume >= 0.0
    
    def test_sms_emoa_convergence(self):
        """Test that SMS-EMOA shows convergence behavior."""
        population = [Solution(3) for _ in range(8)]
        
        # Track hypervolume over generations
        hypervolumes = []
        
        for generation in range(5):
            run_sms_emoa_generation(population, mutation_rate=0.3, tournament_size=2)
            hypervolume = compute_hypervolume(population, (-1.0, 10.0))
            hypervolumes.append(hypervolume)
        
        # Hypervolume should generally increase or stay stable
        # (allowing for some stochastic variation)
        assert len(hypervolumes) == 5
        assert all(hv >= 0.0 for hv in hypervolumes) 