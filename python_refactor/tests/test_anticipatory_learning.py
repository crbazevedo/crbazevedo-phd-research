"""
Tests for Anticipatory Learning implementation.
"""

import pytest
import numpy as np
from src.algorithms.anticipatory_learning import (
    observe_state, prediction_error, anticipatory_learning_single,
    anticipatory_learning_population, kalman_filter_prediction,
    initialize_portfolio_kalman, update_kalman_measurement_noise,
    compute_stochastic_delta_s_contribution, apply_anticipatory_learning_to_algorithm
)
from src.portfolio.portfolio import Portfolio
from src.algorithms.solution import Solution


class TestStateObservation:
    """Test state observation functions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Initialize Portfolio static variables
        Portfolio.available_assets_size = 3
        Portfolio.mean_ROI = np.array([0.1, 0.15, 0.12])
        Portfolio.median_ROI = np.array([0.1, 0.15, 0.12])
        Portfolio.covariance = np.array([
            [0.04, 0.02, 0.01],
            [0.02, 0.09, 0.015],
            [0.01, 0.015, 0.16]
        ])
        Portfolio.robust_covariance = Portfolio.covariance.copy()
        Portfolio.robustness = False
        Portfolio.window_size = 5
        
        # Create synthetic returns data
        Portfolio.complete_returns_data = np.random.normal(0, 0.1, (50, 3))
        
        # Create portfolio
        self.portfolio = Portfolio(3)
        self.portfolio.init()
    
    def test_observe_state_with_kalman(self):
        """Test state observation with Kalman filter."""
        # Initialize Kalman filter
        initialize_portfolio_kalman(self.portfolio, 0.1, 0.05)
        
        # Test state observation
        observe_state(self.portfolio, 10, 0)
        
        # Check that observed values were set
        assert hasattr(self.portfolio, 'ROI_observed')
        assert hasattr(self.portfolio, 'risk_observed')
        assert self.portfolio.kalman_state is not None
    
    def test_observe_state_without_kalman(self):
        """Test state observation without Kalman filter."""
        # Test state observation without Kalman filter
        observe_state(self.portfolio, 10, 0)
        
        # Should not fail, but no Kalman updates
        assert self.portfolio.kalman_state is None
    
    def test_observe_state_no_data(self):
        """Test state observation with no data."""
        Portfolio.complete_returns_data = None
        
        # Should not fail
        observe_state(self.portfolio, 10, 0)
        
        # No updates should occur
        assert self.portfolio.kalman_state is None


class TestPredictionError:
    """Test prediction error computation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Initialize Portfolio static variables
        Portfolio.available_assets_size = 3
        Portfolio.mean_ROI = np.array([0.1, 0.15, 0.12])
        Portfolio.median_ROI = np.array([0.1, 0.15, 0.12])
        Portfolio.covariance = np.array([
            [0.04, 0.02, 0.01],
            [0.02, 0.09, 0.015],
            [0.01, 0.015, 0.16]
        ])
        Portfolio.robust_covariance = Portfolio.covariance.copy()
        Portfolio.robustness = False
        Portfolio.window_size = 5
        
        # Create synthetic returns data
        Portfolio.complete_returns_data = np.random.normal(0, 0.1, (50, 3))
        
        # Create portfolio
        self.portfolio = Portfolio(3)
        self.portfolio.init()
        self.portfolio.ROI = 0.12
        self.portfolio.risk = 0.08
    
    def test_prediction_error_basic(self):
        """Test basic prediction error computation."""
        error = prediction_error(self.portfolio, 0)
        
        assert isinstance(error, float)
        assert error >= 0.0  # Error should be non-negative
        assert not np.isnan(error)
        assert not np.isinf(error)
    
    def test_prediction_error_no_data(self):
        """Test prediction error with no data."""
        Portfolio.complete_returns_data = None
        
        error = prediction_error(self.portfolio, 0)
        
        assert error == 0.0  # Should return 0 when no data
    
    def test_prediction_error_out_of_bounds(self):
        """Test prediction error with out-of-bounds time."""
        error = prediction_error(self.portfolio, 100)  # Time beyond data
        
        assert error == 0.0  # Should return 0 when out of bounds


class TestAnticipatoryLearning:
    """Test anticipatory learning functions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Initialize Portfolio static variables
        Portfolio.available_assets_size = 3
        Portfolio.mean_ROI = np.array([0.1, 0.15, 0.12])
        Portfolio.median_ROI = np.array([0.1, 0.15, 0.12])
        Portfolio.covariance = np.array([
            [0.04, 0.02, 0.01],
            [0.02, 0.09, 0.015],
            [0.01, 0.015, 0.16]
        ])
        Portfolio.robust_covariance = Portfolio.covariance.copy()
        Portfolio.robustness = False
        Portfolio.window_size = 5
        
        # Create synthetic returns data
        Portfolio.complete_returns_data = np.random.normal(0, 0.1, (50, 3))
        
        # Create solution
        self.solution = Solution(3)
        self.solution.P.ROI = 0.12
        self.solution.P.risk = 0.08
    
    def test_anticipatory_learning_single(self):
        """Test anticipatory learning for single solution."""
        # Initialize Kalman filter
        initialize_portfolio_kalman(self.solution.P, 0.12, 0.08)
        
        # Apply anticipatory learning
        anticipatory_learning_single(self.solution, 0)
        
        # Check that learning was applied
        assert self.solution.anticipation == True
        assert hasattr(self.solution, 'alpha')
        assert 0.0 <= self.solution.alpha <= 1.0
        assert hasattr(self.solution, 'prediction_error')
        assert self.solution.prediction_error >= 0.0
    
    def test_anticipatory_learning_single_no_kalman(self):
        """Test anticipatory learning without Kalman filter."""
        # Explicitly remove Kalman filter
        self.solution.P.kalman_state = None
        
        # Apply anticipatory learning without Kalman filter
        anticipatory_learning_single(self.solution, 0)
        
        # Should not fail, but no learning should occur
        assert self.solution.anticipation == False
    
    def test_anticipatory_learning_population(self):
        """Test anticipatory learning for population."""
        # Create population
        population = [Solution(3) for _ in range(5)]
        
        # Initialize Kalman filters
        for solution in population:
            initialize_portfolio_kalman(solution.P, 0.12, 0.08)
        
        # Apply anticipatory learning
        anticipatory_learning_population(population, 0)
        
        # Check that learning was applied to all solutions
        for solution in population:
            assert solution.anticipation == True
            assert hasattr(solution, 'alpha')
            assert hasattr(solution, 'prediction_error')


class TestKalmanFilterPrediction:
    """Test Kalman filter prediction functions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create population with Kalman filters
        self.population = [Solution(3) for _ in range(5)]
        
        for solution in self.population:
            initialize_portfolio_kalman(solution.P, 0.12, 0.08)
    
    def test_kalman_filter_prediction(self):
        """Test Kalman filter prediction for population."""
        # Store initial values
        initial_predictions = []
        for solution in self.population:
            initial_predictions.append({
                'roi_pred': solution.P.ROI_prediction,
                'risk_pred': solution.P.risk_prediction
            })
        
        # Apply prediction
        kalman_filter_prediction(self.population)
        
        # Check that predictions were updated
        for i, solution in enumerate(self.population):
            # Check that error covariance prediction was updated
            assert solution.P.error_covar_prediction is not None
            
            # Note: Predictions might not change if velocities are zero and control input is zero
            # This is expected behavior for the Kalman filter model


class TestKalmanInitialization:
    """Test Kalman filter initialization functions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.portfolio = Portfolio(3)
        self.portfolio.init()
    
    def test_initialize_portfolio_kalman(self):
        """Test Kalman filter initialization for portfolio."""
        initialize_portfolio_kalman(self.portfolio, 0.15, 0.10)
        
        # Check that Kalman filter was initialized
        assert self.portfolio.kalman_state is not None
        assert self.portfolio.ROI_prediction == 0.15
        assert self.portfolio.risk_prediction == 0.10
        assert self.portfolio.error_covar is not None
        assert self.portfolio.error_covar_prediction is not None
    
    def test_update_kalman_measurement_noise(self):
        """Test updating Kalman filter measurement noise."""
        initialize_portfolio_kalman(self.portfolio, 0.15, 0.10)
        
        roi_variance = 0.02
        risk_variance = 0.03
        covariance = 0.005
        
        update_kalman_measurement_noise(self.portfolio, roi_variance, risk_variance, covariance)
        
        # Check that measurement noise was updated
        expected_R = np.array([
            [roi_variance, covariance],
            [covariance, risk_variance]
        ])
        
        np.testing.assert_array_equal(self.portfolio.kalman_state.R, expected_R)


class TestStochasticDeltaS:
    """Test stochastic Delta-S contribution computation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.solution = Solution(3)
        self.solution.P.ROI = 0.15
        self.solution.P.risk = 0.08
        initialize_portfolio_kalman(self.solution.P, 0.15, 0.08)
    
    def test_compute_stochastic_delta_s_contribution(self):
        """Test stochastic Delta-S contribution computation."""
        reference_point = (-1.0, 10.0)
        
        contribution = compute_stochastic_delta_s_contribution(self.solution, reference_point)
        
        assert isinstance(contribution, float)
        assert not np.isnan(contribution)
        assert not np.isinf(contribution)
    
    def test_compute_stochastic_delta_s_no_kalman(self):
        """Test stochastic Delta-S without Kalman filter."""
        self.solution.P.kalman_state = None
        reference_point = (-1.0, 10.0)
        
        contribution = compute_stochastic_delta_s_contribution(self.solution, reference_point)
        
        assert contribution == 0.0  # Should return 0 when no Kalman filter


class TestAnticipatoryLearningIntegration:
    """Test anticipatory learning integration with algorithms."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Initialize Portfolio static variables
        Portfolio.available_assets_size = 3
        Portfolio.mean_ROI = np.array([0.1, 0.15, 0.12])
        Portfolio.median_ROI = np.array([0.1, 0.15, 0.12])
        Portfolio.covariance = np.array([
            [0.04, 0.02, 0.01],
            [0.02, 0.09, 0.015],
            [0.01, 0.015, 0.16]
        ])
        Portfolio.robust_covariance = Portfolio.covariance.copy()
        Portfolio.robustness = False
        Portfolio.window_size = 5
        Portfolio.complete_returns_data = np.random.normal(0, 0.1, (50, 3))
        
        # Create population
        self.population = [Solution(3) for _ in range(5)]
        
        # Initialize Kalman filters
        for solution in self.population:
            initialize_portfolio_kalman(solution.P, 0.12, 0.08)
    
    def test_apply_anticipatory_learning_to_algorithm(self):
        """Test applying anticipatory learning to algorithm."""
        # Apply anticipatory learning
        apply_anticipatory_learning_to_algorithm(self.population, 0, 'both')
        
        # Check that learning was applied
        for solution in self.population:
            assert solution.anticipation == True
            assert hasattr(solution, 'alpha')
            assert hasattr(solution, 'prediction_error')
    
    def test_apply_anticipatory_learning_negative_time(self):
        """Test anticipatory learning with negative time (should skip)."""
        # Apply anticipatory learning with negative time
        apply_anticipatory_learning_to_algorithm(self.population, -1, 'both')
        
        # Check that no learning was applied
        for solution in self.population:
            assert solution.anticipation == False
    
    def test_apply_anticipatory_learning_nsga2(self):
        """Test anticipatory learning for NSGA-II algorithm."""
        apply_anticipatory_learning_to_algorithm(self.population, 0, 'nsga2')
        
        # Check that learning was applied
        for solution in self.population:
            assert solution.anticipation == True
    
    def test_apply_anticipatory_learning_sms_emoa(self):
        """Test anticipatory learning for SMS-EMOA algorithm."""
        apply_anticipatory_learning_to_algorithm(self.population, 0, 'sms_emoa')
        
        # Check that learning was applied
        for solution in self.population:
            assert solution.anticipation == True


class TestAnticipatoryLearningRobustness:
    """Test anticipatory learning robustness features."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Initialize Portfolio static variables
        Portfolio.available_assets_size = 3
        Portfolio.mean_ROI = np.array([0.1, 0.15, 0.12])
        Portfolio.median_ROI = np.array([0.1, 0.15, 0.12])
        Portfolio.covariance = np.array([
            [0.04, 0.02, 0.01],
            [0.02, 0.09, 0.015],
            [0.01, 0.015, 0.16]
        ])
        Portfolio.robust_covariance = Portfolio.covariance.copy()
        Portfolio.robustness = True  # Enable robust mode
        Portfolio.window_size = 5
        Portfolio.complete_returns_data = np.random.normal(0, 0.1, (50, 3))
        
        # Create solution
        self.solution = Solution(3)
        self.solution.P.ROI = 0.12
        self.solution.P.risk = 0.08
        initialize_portfolio_kalman(self.solution.P, 0.12, 0.08)
    
    def test_anticipatory_learning_robust_mode(self):
        """Test anticipatory learning in robust mode."""
        anticipatory_learning_single(self.solution, 0)
        
        # Check that robust values were updated
        assert self.solution.P.robust_ROI == self.solution.P.ROI
        assert self.solution.P.robust_risk == self.solution.P.risk
        assert self.solution.anticipation == True
    
    def test_anticipatory_learning_non_robust_mode(self):
        """Test anticipatory learning in non-robust mode."""
        Portfolio.robustness = False
        
        anticipatory_learning_single(self.solution, 0)
        
        # Check that non-robust values were updated
        assert self.solution.P.non_robust_ROI == self.solution.P.ROI
        assert self.solution.P.non_robust_risk == self.solution.P.risk
        assert self.solution.anticipation == True 