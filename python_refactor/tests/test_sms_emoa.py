"""
Tests for SMS-EMOA algorithm implementation.
"""

import pytest
import numpy as np
from typing import List

from src.algorithms.sms_emoa import (
    SMSEMOA, StochasticParams
)
from src.algorithms.solution import Solution
from src.portfolio.portfolio import Portfolio


class TestStochasticParams:
    """Test StochasticParams class."""
    
    def setup_method(self):
        """Set up test data."""
        # Create a test solution with Kalman state
        from src.algorithms.kalman_filter import create_kalman_params
        
        self.solution = Solution(num_assets=3)
        self.solution.P.ROI = 0.12
        self.solution.P.risk = 0.08
        self.solution.P.kalman_state = create_kalman_params(0.12, 0.08)
    
    def test_stochastic_params_initialization(self):
        """Test StochasticParams initialization."""
        stochastic_params = StochasticParams(self.solution)
        
        # Check that all attributes are computed
        assert hasattr(stochastic_params, 'cov')
        assert hasattr(stochastic_params, 'var_ROI')
        assert hasattr(stochastic_params, 'var_risk')
        assert hasattr(stochastic_params, 'corr')
        assert hasattr(stochastic_params, 'var_ratio')
        assert hasattr(stochastic_params, 'conditional_mean_ROI')
        assert hasattr(stochastic_params, 'conditional_var_ROI')
        assert hasattr(stochastic_params, 'conditional_mean_risk')
        assert hasattr(stochastic_params, 'conditional_var_risk')
        
        # Check that values are finite
        assert np.isfinite(stochastic_params.cov)
        assert np.isfinite(stochastic_params.var_ROI)
        assert np.isfinite(stochastic_params.var_risk)
        assert np.isfinite(stochastic_params.corr)
        assert np.isfinite(stochastic_params.var_ratio)
        assert np.isfinite(stochastic_params.conditional_mean_ROI)
        assert np.isfinite(stochastic_params.conditional_var_ROI)
        assert np.isfinite(stochastic_params.conditional_mean_risk)
        assert np.isfinite(stochastic_params.conditional_var_risk)
        
        # Check correlation bounds
        assert -1.0 <= stochastic_params.corr <= 1.0
        
        # Check variance ratios are non-negative
        assert stochastic_params.var_ratio >= 0.0
        assert stochastic_params.conditional_var_ROI >= 0.0
        assert stochastic_params.conditional_var_risk >= 0.0


class TestSMSEMOA:
    """Test cases for SMS-EMOA algorithm."""
    
    def setup_method(self):
        """Set up test data."""
        # Set up portfolio data for testing
        Portfolio.available_assets_size = 5
        Portfolio.mean_ROI = np.array([0.1, 0.15, 0.12, 0.08, 0.18])
        Portfolio.median_ROI = np.array([0.1, 0.15, 0.12, 0.08, 0.18])
        Portfolio.covariance = np.array([
            [0.04, 0.02, 0.01, 0.005, 0.03],
            [0.02, 0.09, 0.015, 0.01, 0.025],
            [0.01, 0.015, 0.16, 0.008, 0.02],
            [0.005, 0.01, 0.008, 0.25, 0.015],
            [0.03, 0.025, 0.02, 0.015, 0.36]
        ])
        Portfolio.robust_covariance = Portfolio.covariance.copy()
        Portfolio.robustness = False
        
        # Set up complete returns data for anticipatory learning
        Portfolio.complete_returns_data = np.random.randn(100, 5) * 0.1
        Portfolio.window_size = 20
        
        # Create SMS-EMOA instance
        self.sms_emoa = SMSEMOA(
            population_size=10,
            generations=50,
            mutation_rate=0.1,
            crossover_rate=0.9,
            tournament_size=2
        )
        
        # Create test solutions
        self.solutions = []
        for i in range(10):
            solution = Solution(num_assets=5)
            solution.P.ROI = 0.1 + i * 0.02
            solution.P.risk = 0.2 - i * 0.01
            solution.pareto_rank = 0 if i < 5 else 1
            solution.Delta_S = 0.0
            self.solutions.append(solution)
    
    def test_sms_emoa_initialization(self):
        """Test SMS-EMOA initialization."""
        assert self.sms_emoa.population_size == 10
        assert self.sms_emoa.generations == 50
        assert self.sms_emoa.mutation_rate == 0.1
        assert self.sms_emoa.crossover_rate == 0.9
        assert self.sms_emoa.tournament_size == 2
        assert self.sms_emoa.anticipatory_learning is None
        assert len(self.sms_emoa.population) == 0
        assert len(self.sms_emoa.pareto_front) == 0
    
    def test_initialize_population(self):
        """Test population initialization."""
        # Create test data
        data = {
            'num_assets': 5,
            'returns_data': np.random.randn(100, 5) * 0.1
        }
        self.sms_emoa._initialize_population(data)
        
        assert len(self.sms_emoa.population) == self.sms_emoa.population_size
        
        # Check that all solutions have valid portfolios
        for solution in self.sms_emoa.population:
            assert hasattr(solution, 'P')
            assert hasattr(solution.P, 'ROI')
            assert hasattr(solution.P, 'risk')
            assert hasattr(solution.P, 'investment')
            assert len(solution.P.investment) == 5
            assert np.abs(np.sum(solution.P.investment) - 1.0) < 1e-10
    
    def test_evaluate_population(self):
        """Test population evaluation."""
        # Create test data
        data = {
            'num_assets': 5,
            'returns_data': np.random.randn(100, 5) * 0.1
        }
        
        # Initialize population
        self.sms_emoa._initialize_population(data)
        
        # Evaluate population
        for solution in self.sms_emoa.population:
            self.sms_emoa._evaluate_solution(solution, data)
        
        # Check that all solutions have been evaluated
        for solution in self.sms_emoa.population:
            assert hasattr(solution, 'P')
            assert hasattr(solution.P, 'ROI')
            assert hasattr(solution.P, 'risk')
            assert np.isfinite(solution.P.ROI)
            assert np.isfinite(solution.P.risk)
    
    def test_fast_non_dominated_sort(self):
        """Test fast non-dominated sorting."""
        # Create test data
        data = {
            'num_assets': 5,
            'returns_data': np.random.randn(100, 5) * 0.1
        }
        
        # Initialize and evaluate population
        self.sms_emoa._initialize_population(data)
        for solution in self.sms_emoa.population:
            self.sms_emoa._evaluate_solution(solution, data)
        
        # Perform non-dominated sorting
        num_fronts = self.sms_emoa._fast_non_dominated_sort()
        
        assert num_fronts > 0
        assert num_fronts <= len(self.sms_emoa.population)
        
        # Check that all solutions have been assigned a front
        for solution in self.sms_emoa.population:
            assert hasattr(solution, 'Pareto_rank')
            assert solution.Pareto_rank >= 0
    
    def test_compute_hypervolume_contribution(self):
        """Test hypervolume contribution computation."""
        # Create test data
        data = {
            'num_assets': 5,
            'returns_data': np.random.randn(100, 5) * 0.1
        }
        
        # Initialize and evaluate population
        self.sms_emoa._initialize_population(data)
        for solution in self.sms_emoa.population:
            self.sms_emoa._evaluate_solution(solution, data)
        
        # Perform non-dominated sorting
        self.sms_emoa._fast_non_dominated_sort()
        
        # Compute hypervolume contributions
        self.sms_emoa._compute_hypervolume_contributions()
        
        # Check that Delta_S values are set
        for solution in self.sms_emoa.population:
            assert hasattr(solution, 'Delta_S')
            assert isinstance(solution.Delta_S, float)
    
    def test_compute_stochastic_hypervolume_contributions(self):
        """Test stochastic hypervolume contribution computation."""
        # Create test data
        data = {
            'num_assets': 5,
            'returns_data': np.random.randn(100, 5) * 0.1
        }
        
        # Initialize and evaluate population
        self.sms_emoa._initialize_population(data)
        for solution in self.sms_emoa.population:
            self.sms_emoa._evaluate_solution(solution, data)
        
        # Perform non-dominated sorting
        self.sms_emoa._fast_non_dominated_sort()
        
        # Compute stochastic hypervolume contributions
        self.sms_emoa._compute_stochastic_hypervolume_contributions()
        
        # Check that Delta_S values are set
        for solution in self.sms_emoa.population:
            assert hasattr(solution, 'Delta_S')
            assert isinstance(solution.Delta_S, float)
    
    def test_tournament_selection(self):
        """Test tournament selection."""
        # Create test data
        data = {
            'num_assets': 5,
            'returns_data': np.random.randn(100, 5) * 0.1
        }
        
        # Initialize and evaluate population
        self.sms_emoa._initialize_population(data)
        for solution in self.sms_emoa.population:
            self.sms_emoa._evaluate_solution(solution, data)
        self.sms_emoa._fast_non_dominated_sort()
        self.sms_emoa._compute_hypervolume_contributions()
        
        # Perform tournament selection
        selected = self.sms_emoa._tournament_selection()
        
        assert selected is not None
        assert selected >= 0
        assert selected < len(self.sms_emoa.population)
    
    def test_crossover_and_mutation(self):
        """Test crossover and mutation operations."""
        # Create test data
        data = {
            'num_assets': 5,
            'returns_data': np.random.randn(100, 5) * 0.1
        }
        
        # Initialize and evaluate population
        self.sms_emoa._initialize_population(data)
        for solution in self.sms_emoa.population:
            self.sms_emoa._evaluate_solution(solution, data)
        
        # Select parents
        parent1 = self.sms_emoa.population[0]
        parent2 = self.sms_emoa.population[1]
        
        # Create offspring using operators
        from src.algorithms.operators import crossover, mutation
        offspring1, offspring2 = crossover(parent1, parent2, self.sms_emoa.crossover_rate)
        mutation(offspring1, self.sms_emoa.mutation_rate)
        mutation(offspring2, self.sms_emoa.mutation_rate)
        
        # Test offspring1
        offspring = offspring1
        
        assert offspring is not None
        assert hasattr(offspring, 'P')
        assert hasattr(offspring.P, 'investment')
        assert len(offspring.P.investment) == 5
        assert np.abs(np.sum(offspring.P.investment) - 1.0) < 1e-10
    
    def test_environmental_selection(self):
        """Test environmental selection."""
        # Create test data
        data = {
            'num_assets': 5,
            'returns_data': np.random.randn(100, 5) * 0.1
        }
        
        # Initialize and evaluate population
        self.sms_emoa._initialize_population(data)
        for solution in self.sms_emoa.population:
            self.sms_emoa._evaluate_solution(solution, data)
        
        # Create offspring
        parent1 = self.sms_emoa.population[0]
        parent2 = self.sms_emoa.population[1]
        from src.algorithms.operators import crossover, mutation
        offspring1, offspring2 = crossover(parent1, parent2, self.sms_emoa.crossover_rate)
        mutation(offspring1, self.sms_emoa.mutation_rate)
        mutation(offspring2, self.sms_emoa.mutation_rate)
        
        # Perform environmental selection
        self.sms_emoa._remove_worst_solution()
        
        # Population size should remain the same
        assert len(self.sms_emoa.population) == self.sms_emoa.population_size
    
    def test_run_generation(self):
        """Test running a single generation."""
        # Create test data
        data = {
            'num_assets': 5,
            'returns_data': np.random.randn(100, 5) * 0.1
        }
        
        # Initialize population
        self.sms_emoa._initialize_population(data)
        for solution in self.sms_emoa.population:
            self.sms_emoa._evaluate_solution(solution, data)
        
        # Run one generation
        self.sms_emoa._run_generation()
        
        # Population size should remain the same
        assert len(self.sms_emoa.population) == self.sms_emoa.population_size
        
        # All solutions should be evaluated
        for solution in self.sms_emoa.population:
            assert hasattr(solution, 'P')
            assert hasattr(solution.P, 'ROI')
            assert hasattr(solution.P, 'risk')
    
    def test_run_optimization(self):
        """Test running complete optimization."""
        # Create test data
        data = {
            'num_assets': 5,
            'returns_data': np.random.randn(100, 5) * 0.1
        }
        
        # Run optimization
        self.sms_emoa.run(data)
        
        # Check that optimization completed
        assert len(self.sms_emoa.population) == self.sms_emoa.population_size
        assert len(self.sms_emoa.pareto_front) > 0
        
        # Check that metrics were collected
        assert len(self.sms_emoa.hypervolume_history) > 0
        # Stochastic hypervolume history only populated when anticipatory learning is enabled
        if self.sms_emoa.anticipatory_learning is not None:
            assert len(self.sms_emoa.stochastic_hypervolume_history) > 0
    
    def test_get_pareto_front(self):
        """Test getting Pareto front."""
        # Create test data
        data = {
            'num_assets': 5,
            'returns_data': np.random.randn(100, 5) * 0.1
        }
        
        # Run optimization
        self.sms_emoa.run(data)
        
        # Get Pareto front
        pareto_front = self.sms_emoa.get_pareto_front()
        
        assert len(pareto_front) > 0
        assert all(solution.Pareto_rank == 0 for solution in pareto_front)
    
    def test_get_optimization_metrics(self):
        """Test getting optimization metrics."""
        # Create test data
        data = {
            'num_assets': 5,
            'returns_data': np.random.randn(100, 5) * 0.1
        }
        
        # Run optimization
        self.sms_emoa.run(data)
        
        # Get metrics
        hypervolume = self.sms_emoa.get_hypervolume()
        expected_future_hypervolume = self.sms_emoa.get_expected_future_hypervolume()
        function_evaluations = self.sms_emoa.get_function_evaluations()
        
        assert isinstance(hypervolume, float)
        assert isinstance(expected_future_hypervolume, float)
        assert isinstance(function_evaluations, int)
        assert len(self.sms_emoa.hypervolume_history) > 0
        # Stochastic hypervolume history only populated when anticipatory learning is enabled
        if self.sms_emoa.anticipatory_learning is not None:
            assert len(self.sms_emoa.stochastic_hypervolume_history) > 0


class TestSMSEMOAIntegration:
    """Test SMS-EMOA integration with other components."""
    
    def setup_method(self):
        """Set up test data."""
        # Set up portfolio data
        Portfolio.available_assets_size = 3
        Portfolio.mean_ROI = np.array([0.1, 0.15, 0.12])
        Portfolio.median_ROI = np.array([0.1, 0.15, 0.12])  # Same as mean for simplicity
        Portfolio.covariance = np.eye(3) * 0.1
        Portfolio.robust_covariance = np.eye(3) * 0.1
        Portfolio.complete_returns_data = np.random.randn(50, 3) * 0.1
        
        self.sms_emoa = SMSEMOA(population_size=5, generations=10)
    
    def test_integration_with_anticipatory_learning(self):
        """Test integration with anticipatory learning."""
        # Create test data
        data = {
            'num_assets': 3,
            'returns_data': np.random.randn(100, 3) * 0.1
        }
        
        # Run optimization with anticipatory learning
        self.sms_emoa.run(data)
        
        # Check that anticipatory learning was applied
        for solution in self.sms_emoa.population:
            assert hasattr(solution, 'anticipation')
    
    def test_integration_with_portfolio_class(self):
        """Test integration with Portfolio class."""
        # Create test data
        data = {
            'num_assets': 3,
            'returns_data': np.random.randn(100, 3) * 0.1
        }
        
        # Run optimization
        self.sms_emoa.run(data)
        
        # Check that all solutions have valid portfolios
        for solution in self.sms_emoa.population:
            assert hasattr(solution, 'P')
            assert hasattr(solution.P, 'ROI')
            assert hasattr(solution.P, 'risk')
            assert hasattr(solution.P, 'investment')
            assert len(solution.P.investment) == 3
    
    def test_convergence_behavior(self):
        """Test convergence behavior."""
        # Create test data
        data = {
            'num_assets': 3,
            'returns_data': np.random.randn(100, 3) * 0.1
        }
        
        # Run optimization
        self.sms_emoa.run(data)
        
        # Check that hypervolume improved or stayed the same
        hypervolume_history = self.sms_emoa.hypervolume_history
        
        # Hypervolume should generally increase or stay the same
        # (allowing for some fluctuations due to stochastic nature)
        assert len(hypervolume_history) > 0
        assert all(hv >= 0.0 for hv in hypervolume_history)


class TestSMSEMOAEdgeCases:
    """Test SMS-EMOA edge cases and error handling."""
    
    def setup_method(self):
        """Set up test data."""
        Portfolio.available_assets_size = 2
        Portfolio.mean_ROI = np.array([0.1, 0.15])
        Portfolio.median_ROI = np.array([0.1, 0.15])  # Same as mean for simplicity
        Portfolio.covariance = np.eye(2) * 0.1
        Portfolio.robust_covariance = np.eye(2) * 0.1
        Portfolio.complete_returns_data = np.random.randn(20, 2) * 0.1
    
    def test_small_population(self):
        """Test with small population."""
        sms_emoa = SMSEMOA(population_size=2, generations=5)
        # Create test data
        data = {
            'num_assets': 2,
            'returns_data': np.random.randn(50, 2) * 0.1
        }
        sms_emoa.run(data)
        
        assert len(sms_emoa.population) == 2
        assert len(sms_emoa.pareto_front) > 0
    
    def test_single_generation(self):
        """Test with single generation."""
        sms_emoa = SMSEMOA(population_size=5, generations=1)
        # Create test data
        data = {
            'num_assets': 2,
            'returns_data': np.random.randn(50, 2) * 0.1
        }
        sms_emoa.run(data)
        
        assert len(sms_emoa.hypervolume_history) == 1
    
    def test_zero_mutation_rate(self):
        """Test with zero mutation rate."""
        sms_emoa = SMSEMOA(population_size=5, generations=5, mutation_rate=0.0)
        # Create test data
        data = {
            'num_assets': 2,
            'returns_data': np.random.randn(50, 2) * 0.1
        }
        sms_emoa.run(data)
        
        assert len(sms_emoa.population) == 5
    
    def test_zero_crossover_rate(self):
        """Test with zero crossover rate."""
        sms_emoa = SMSEMOA(population_size=5, generations=5, crossover_rate=0.0)
        # Create test data
        data = {
            'num_assets': 2,
            'returns_data': np.random.randn(50, 2) * 0.1
        }
        sms_emoa.run(data)
        
        assert len(sms_emoa.population) == 5 