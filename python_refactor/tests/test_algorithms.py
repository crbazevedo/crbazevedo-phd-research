"""
Tests for algorithms module functionality.
"""

import pytest
import numpy as np
from typing import List

from src.algorithms.solution import Solution, create_random_solution
from src.algorithms.operators import (
    crossover, mutation, tournament_selection, binary_tournament_selection,
    create_offspring_population
)
from src.algorithms.nsga2 import (
    fast_non_dominated_sort, calculate_crowding_distance,
    select_next_generation, run_nsga2_generation, run_nsga2,
    get_pareto_front, evaluate_population_statistics
)
from src.portfolio.portfolio import Portfolio


class TestSolution:
    """Test cases for Solution class."""
    
    def test_solution_creation(self):
        """Test Solution object creation."""
        solution = Solution(5)
        assert solution.P.num_assets == 5
        assert solution.cd == 0.0
        assert solution.Pareto_rank == 0
        assert 0.0 <= solution.stability <= 1.0  # Stability is now calculated from Kalman filter
    
    def test_solution_dominance_without_constraints(self):
        """Test dominance checking without constraints."""
        # Create two solutions with different ROI and risk
        solution1 = Solution(3)
        solution1.P.ROI = 0.1
        solution1.P.risk = 0.05
        
        solution2 = Solution(3)
        solution2.P.ROI = 0.08
        solution2.P.risk = 0.06
        
        # solution1 should dominate solution2 (higher ROI, lower risk)
        assert solution1.dominates_without_constraints(solution2) == True
        assert solution2.dominates_without_constraints(solution1) == False
    
    def test_solution_dominance_with_constraints(self):
        """Test dominance checking with cardinality constraints."""
        Portfolio.max_cardinality = 2
        
        # Create solutions with different cardinalities
        solution1 = Solution(3)
        solution1.P.ROI = 0.1
        solution1.P.risk = 0.05
        solution1.P.cardinality = 1  # Within constraint
        
        solution2 = Solution(3)
        solution2.P.ROI = 0.08
        solution2.P.risk = 0.06
        solution2.P.cardinality = 3  # Exceeds constraint
        
        # solution1 should dominate solution2 due to constraint violation
        assert solution1.dominates_with_constraints(solution2) == True
        assert solution2.dominates_with_constraints(solution1) == False
    
    def test_solution_comparison(self):
        """Test solution comparison for sorting."""
        solution1 = Solution(3)
        solution1.Pareto_rank = 0
        solution1.cd = 0.5
        
        solution2 = Solution(3)
        solution2.Pareto_rank = 1
        solution2.cd = 1.0
        
        # solution1 should be "less than" solution2 (better rank)
        assert solution1 < solution2
        
        # Same rank, different crowding distance
        solution3 = Solution(3)
        solution3.Pareto_rank = 0
        solution3.cd = 0.3
        
        assert solution1 < solution3  # Higher crowding distance is better


class TestGeneticOperators:
    """Test cases for genetic operators."""
    
    def test_crossover(self):
        """Test crossover operation."""
        parent1 = Solution(3)
        parent1.P.investment = np.array([0.5, 0.3, 0.2])
        
        parent2 = Solution(3)
        parent2.P.investment = np.array([0.2, 0.5, 0.3])
        
        # Test multiple times to account for randomness
        crossover_occurred = False
        for _ in range(10):
            offspring1, offspring2 = crossover(parent1, parent2, crossover_rate=1.0)
            
            # Check that weights sum to 1
            assert abs(np.sum(offspring1.P.investment) - 1.0) < 1e-6
            assert abs(np.sum(offspring2.P.investment) - 1.0) < 1e-6
            
            # Check if crossover actually occurred (offspring different from parents)
            if (not np.array_equal(offspring1.P.investment, parent1.P.investment) or 
                not np.array_equal(offspring2.P.investment, parent2.P.investment)):
                crossover_occurred = True
                break
        
        # At least one crossover should have occurred
        assert crossover_occurred, "Crossover should produce different offspring at least once"
    
    def test_mutation(self):
        """Test mutation operation."""
        solution = Solution(3)
        original_weights = np.copy(solution.P.investment)
        
        mutated = mutation(solution, mutation_rate=1.0)
        
        # Check that mutation occurred
        assert not np.array_equal(mutated.P.investment, original_weights)
        
        # Check that weights sum to 1
        assert abs(np.sum(mutated.P.investment) - 1.0) < 1e-6
    
    def test_tournament_selection(self):
        """Test tournament selection."""
        population = [Solution(3) for _ in range(10)]
        
        # Set different ROI values for testing
        for i, solution in enumerate(population):
            solution.P.ROI = i * 0.01
        
        selected_idx = tournament_selection(population, tournament_size=3)
        
        # Check that a valid index was selected
        assert 0 <= selected_idx < len(population)
        assert population[selected_idx] in population
    
    def test_binary_tournament_selection(self):
        """Test binary tournament selection."""
        population = [Solution(3) for _ in range(10)]
        
        # Set different ROI values for testing
        for i, solution in enumerate(population):
            solution.P.ROI = i * 0.01
        
        selected = binary_tournament_selection(population)
        
        # Check that a solution was selected
        assert selected in population
    
    def test_create_offspring_population(self):
        """Test offspring population creation."""
        parent_population = [Solution(3) for _ in range(10)]
        offspring_size = 8
        
        offspring = create_offspring_population(
            parent_population, offspring_size, crossover_rate=0.8, mutation_rate=0.1
        )
        
        assert len(offspring) == offspring_size
        
        # Check that all offspring have valid weights
        for solution in offspring:
            assert abs(np.sum(solution.P.investment) - 1.0) < 1e-6


class TestNSGA2:
    """Test cases for NSGA-II algorithm."""
    
    def test_fast_non_dominated_sort(self):
        """Test fast non-dominated sorting."""
        # Create a simple population with known dominance relationships
        population = [Solution(3) for _ in range(4)]
        
        # Set objectives to create clear dominance relationships
        population[0].P.ROI = 0.1
        population[0].P.risk = 0.05
        
        population[1].P.ROI = 0.08
        population[1].P.risk = 0.06
        
        population[2].P.ROI = 0.12
        population[2].P.risk = 0.04
        
        population[3].P.ROI = 0.09
        population[3].P.risk = 0.07
        
        fronts = fast_non_dominated_sort(population)
        
        # Should have at least one front
        assert len(fronts) > 0
        
        # First front should contain non-dominated solutions
        first_front = fronts[0]
        assert len(first_front) > 0
        
        # Check that solutions in first front are not dominated by each other
        for i, sol1 in enumerate(first_front):
            for j, sol2 in enumerate(first_front):
                if i != j:
                    assert not sol1.dominates_without_constraints(sol2)
    
    def test_calculate_crowding_distance(self):
        """Test crowding distance calculation."""
        front = [Solution(3) for _ in range(4)]
        
        # Set different objectives
        front[0].P.ROI = 0.1
        front[0].P.risk = 0.05
        
        front[1].P.ROI = 0.08
        front[1].P.risk = 0.06
        
        front[2].P.ROI = 0.12
        front[2].P.risk = 0.04
        
        front[3].P.ROI = 0.09
        front[3].P.risk = 0.07
        
        calculate_crowding_distance(front)
        
        # Check that crowding distances are calculated
        for solution in front:
            assert solution.cd >= 0.0
        
        # Boundary solutions should have infinite crowding distance
        # (This depends on the sorting, but at least some should be infinite)
        infinite_cds = [s.cd for s in front if s.cd == float('inf')]
        assert len(infinite_cds) >= 2  # At least 2 boundary solutions
    
    def test_select_next_generation(self):
        """Test next generation selection."""
        parent_population = [Solution(3) for _ in range(5)]
        offspring_population = [Solution(3) for _ in range(5)]
        population_size = 5
        
        next_generation = select_next_generation(
            parent_population, offspring_population, population_size
        )
        
        assert len(next_generation) == population_size
        
        # All solutions should be from either parent or offspring population
        all_solutions = parent_population + offspring_population
        for solution in next_generation:
            assert solution in all_solutions
    
    def test_run_nsga2_generation(self):
        """Test running one generation of NSGA-II."""
        population = [Solution(3) for _ in range(10)]
        population_size = 10
        
        new_population = run_nsga2_generation(
            population, population_size, mutation_rate=0.1, crossover_rate=0.9
        )
        
        assert len(new_population) == population_size
        
        # Check that all solutions have valid weights
        for solution in new_population:
            assert abs(np.sum(solution.P.investment) - 1.0) < 1e-6
    
    def test_run_nsga2(self):
        """Test running NSGA-II algorithm."""
        num_generations = 5
        population_size = 10
        num_assets = 3
        
        final_population = run_nsga2(
            num_generations, population_size, num_assets,
            mutation_rate=0.1, crossover_rate=0.9, random_seed=42
        )
        
        assert len(final_population) == population_size
        
        # Check that all solutions have valid weights
        for solution in final_population:
            assert abs(np.sum(solution.P.investment) - 1.0) < 1e-6
    
    def test_get_pareto_front(self):
        """Test extracting Pareto front."""
        population = [Solution(3) for _ in range(10)]
        
        # Set some solutions to Pareto rank 0, others to rank 1
        for i in range(3):
            population[i].Pareto_rank = 0
        for i in range(3, 10):
            population[i].Pareto_rank = 1
        
        pareto_front = get_pareto_front(population)
        
        assert len(pareto_front) == 3
        
        # All solutions in Pareto front should have rank 0
        for solution in pareto_front:
            assert solution.Pareto_rank == 0
    
    def test_evaluate_population_statistics(self):
        """Test population statistics evaluation."""
        population = [Solution(3) for _ in range(5)]
        
        # Set some values for testing
        for i, solution in enumerate(population):
            solution.P.ROI = i * 0.01
            solution.P.risk = i * 0.005
            solution.P.cardinality = i + 1
            solution.Pareto_rank = i % 2
            solution.cd = i * 0.1
        
        stats = evaluate_population_statistics(population)
        
        # Check that statistics are calculated
        assert 'population_size' in stats
        assert 'pareto_front_size' in stats
        assert 'roi_mean' in stats
        assert 'risk_mean' in stats
        assert 'cardinality_mean' in stats
        
        assert stats['population_size'] == 5
        assert stats['roi_mean'] > 0
        assert stats['risk_mean'] > 0


class TestIntegration:
    """Integration tests for algorithms."""
    
    def test_full_optimization_workflow(self):
        """Test complete optimization workflow."""
        # Set up portfolio data
        Portfolio.available_assets_size = 3
        Portfolio.mean_ROI = np.array([0.01, 0.02, 0.015])
        Portfolio.median_ROI = np.array([0.008, 0.018, 0.012])
        Portfolio.covariance = np.array([
            [0.04, 0.01, 0.02],
            [0.01, 0.09, 0.03],
            [0.02, 0.03, 0.16]
        ])
        Portfolio.robust_covariance = Portfolio.covariance.copy()
        Portfolio.robustness = False
        
        # Run optimization
        population = run_nsga2(
            num_generations=3,
            population_size=8,
            num_assets=3,
            random_seed=42
        )
        
        # Check results
        assert len(population) == 8
        
        # Check that solutions have been evaluated
        for solution in population:
            assert solution.P.ROI != 0.0
            assert solution.P.risk != 0.0
            assert solution.P.cardinality > 0
        
        # Check that Pareto front exists
        pareto_front = get_pareto_front(population)
        assert len(pareto_front) > 0 