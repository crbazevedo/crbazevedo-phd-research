"""
Unit tests for Temporal Incomparability Probability (TIP) Implementation

Tests the implementation of Definition 6.1 and related functionality.
"""

import unittest
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from algorithms.temporal_incomparability_probability import TemporalIncomparabilityCalculator


class TestTemporalIncomparabilityProbability(unittest.TestCase):
    """Test cases for TemporalIncomparabilityCalculator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tip_calculator = TemporalIncomparabilityCalculator(monte_carlo_samples=100)
        
        # Create mock solutions for testing
        self.current_solution = self._create_mock_solution(0.1, 0.05)  # 10% ROI, 5% risk
        self.predicted_solution = self._create_mock_solution(0.12, 0.06)  # 12% ROI, 6% risk
        
    def _create_mock_solution(self, roi: float, risk: float):
        """Create a mock solution for testing."""
        class MockPortfolio:
            def __init__(self, roi, risk):
                self.ROI = roi
                self.risk = risk
                self.kalman_state = None
        
        class MockSolution:
            def __init__(self, roi, risk):
                self.P = MockPortfolio(roi, risk)
        
        return MockSolution(roi, risk)
    
    def test_initialization(self):
        """Test TIP calculator initialization."""
        self.assertEqual(self.tip_calculator.monte_carlo_samples, 100)
        self.assertEqual(len(self.tip_calculator.historical_tips), 0)
        
    def test_calculate_tip_basic(self):
        """Test basic TIP calculation."""
        tip = self.tip_calculator.calculate_tip(
            self.current_solution, self.predicted_solution
        )
        
        # TIP should be between 0 and 1
        self.assertGreaterEqual(tip, 0.0)
        self.assertLessEqual(tip, 1.0)
        
        # Should be stored in history
        self.assertEqual(len(self.tip_calculator.historical_tips), 1)
        self.assertEqual(self.tip_calculator.historical_tips[0], tip)
        
    def test_calculate_tip_with_uncertainty(self):
        """Test TIP calculation with prediction uncertainty."""
        tip = self.tip_calculator.calculate_tip(
            self.current_solution, self.predicted_solution,
            prediction_uncertainty=0.05
        )
        
        self.assertGreaterEqual(tip, 0.0)
        self.assertLessEqual(tip, 1.0)
        
    def test_tip_with_dominance_relationships(self):
        """Test TIP calculation with different dominance relationships."""
        # Case 1: Current dominates predicted
        current_dominates = self._create_mock_solution(0.15, 0.03)  # Higher ROI, lower risk
        predicted_dominated = self._create_mock_solution(0.10, 0.05)
        
        tip1 = self.tip_calculator.calculate_tip(current_dominates, predicted_dominated)
        
        # Case 2: Predicted dominates current
        current_dominated = self._create_mock_solution(0.10, 0.05)
        predicted_dominates = self._create_mock_solution(0.15, 0.03)
        
        tip2 = self.tip_calculator.calculate_tip(current_dominated, predicted_dominates)
        
        # Case 3: Mutually non-dominated
        current_non_dom = self._create_mock_solution(0.12, 0.04)  # Higher ROI, higher risk
        predicted_non_dom = self._create_mock_solution(0.10, 0.03)  # Lower ROI, lower risk
        
        tip3 = self.tip_calculator.calculate_tip(current_non_dom, predicted_non_dom)
        
        # TIP should be higher for non-dominated case
        self.assertGreater(tip3, tip1)
        self.assertGreater(tip3, tip2)
        
    def test_binary_entropy(self):
        """Test binary entropy function."""
        # Test known values
        self.assertAlmostEqual(self.tip_calculator.binary_entropy(0.5), 1.0, places=5)
        self.assertAlmostEqual(self.tip_calculator.binary_entropy(0.0), 0.0, places=5)
        self.assertAlmostEqual(self.tip_calculator.binary_entropy(1.0), 0.0, places=5)
        
        # Test symmetry
        p = 0.3
        entropy_p = self.tip_calculator.binary_entropy(p)
        entropy_1_minus_p = self.tip_calculator.binary_entropy(1.0 - p)
        self.assertAlmostEqual(entropy_p, entropy_1_minus_p, places=5)
        
    def test_anticipatory_learning_rate_calculation(self):
        """Test anticipatory learning rate calculation (Equation 6.6)."""
        tip = 0.5  # Maximum uncertainty
        horizon = 3
        
        learning_rate = self.tip_calculator.calculate_anticipatory_learning_rate_tip(tip, horizon)
        
        # Should be between 0 and 1
        self.assertGreaterEqual(learning_rate, 0.0)
        self.assertLessEqual(learning_rate, 1.0)
        
        # Test with different TIP values
        tip_low = 0.1  # Low uncertainty (same entropy as 0.9)
        tip_medium = 0.3  # Medium uncertainty
        tip_high = 0.5  # Maximum uncertainty
        
        lr_low = self.tip_calculator.calculate_anticipatory_learning_rate_tip(tip_low, horizon)
        lr_medium = self.tip_calculator.calculate_anticipatory_learning_rate_tip(tip_medium, horizon)
        lr_high = self.tip_calculator.calculate_anticipatory_learning_rate_tip(tip_high, horizon)
        
        # Higher TIP (closer to 0.5) should result in lower learning rate
        self.assertLess(lr_high, lr_medium)
        self.assertLess(lr_medium, lr_low)
        
    def test_anticipatory_learning_rate_edge_cases(self):
        """Test edge cases for learning rate calculation."""
        # Horizon = 1 should return 0
        lr_horizon_1 = self.tip_calculator.calculate_anticipatory_learning_rate_tip(0.5, 1)
        self.assertEqual(lr_horizon_1, 0.0)
        
        # Horizon = 2 should work
        lr_horizon_2 = self.tip_calculator.calculate_anticipatory_learning_rate_tip(0.5, 2)
        self.assertGreaterEqual(lr_horizon_2, 0.0)
        self.assertLessEqual(lr_horizon_2, 1.0)
        
    def test_historical_tip_trend(self):
        """Test historical TIP trend calculation."""
        # Add some historical TIP values
        self.tip_calculator.historical_tips = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
        trend = self.tip_calculator.get_historical_tip_trend()
        
        # Should be positive (increasing trend)
        self.assertGreater(trend, 0.0)
        
        # Test with decreasing trend
        self.tip_calculator.historical_tips = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
        trend_decreasing = self.tip_calculator.get_historical_tip_trend()
        self.assertLess(trend_decreasing, 0.0)
        
    def test_average_tip(self):
        """Test average TIP calculation."""
        # Add some historical TIP values
        self.tip_calculator.historical_tips = [0.3, 0.4, 0.5, 0.6, 0.7]
        
        avg_tip = self.tip_calculator.get_average_tip(window_size=5)
        expected_avg = np.mean([0.3, 0.4, 0.5, 0.6, 0.7])
        
        self.assertAlmostEqual(avg_tip, expected_avg, places=5)
        
        # Test with smaller window
        avg_tip_small = self.tip_calculator.get_average_tip(window_size=3)
        expected_avg_small = np.mean([0.5, 0.6, 0.7])
        
        self.assertAlmostEqual(avg_tip_small, expected_avg_small, places=5)
        
    def test_tip_statistics(self):
        """Test TIP statistics calculation."""
        # Add some historical TIP values
        self.tip_calculator.historical_tips = [0.3, 0.4, 0.5, 0.6, 0.7]
        
        stats = self.tip_calculator.get_tip_statistics()
        
        self.assertEqual(stats['count'], 5)
        self.assertAlmostEqual(stats['mean'], 0.5, places=5)
        self.assertGreaterEqual(stats['min'], 0.0)
        self.assertLessEqual(stats['max'], 1.0)
        self.assertGreaterEqual(stats['std'], 0.0)
        
    def test_reset_history(self):
        """Test history reset functionality."""
        # Add some historical TIP values
        self.tip_calculator.historical_tips = [0.3, 0.4, 0.5]
        
        # Reset
        self.tip_calculator.reset_history()
        
        self.assertEqual(len(self.tip_calculator.historical_tips), 0)
        
    def test_tip_with_kalman_covariance(self):
        """Test TIP calculation with Kalman filter covariance."""
        # Create solutions with mock Kalman states
        current_sol = self._create_mock_solution(0.1, 0.05)
        predicted_sol = self._create_mock_solution(0.12, 0.06)
        
        # Mock Kalman state with covariance
        class MockKalmanState:
            def __init__(self):
                self.P = np.eye(4) * 0.01
        
        current_sol.P.kalman_state = MockKalmanState()
        predicted_sol.P.kalman_state = MockKalmanState()
        predicted_sol.P.kalman_state.P = np.eye(4) * 0.02
        
        tip = self.tip_calculator.calculate_tip(current_sol, predicted_sol)
        
        self.assertGreaterEqual(tip, 0.0)
        self.assertLessEqual(tip, 1.0)
        
    def test_tip_simple_fallback(self):
        """Test simple TIP calculation fallback."""
        # Test the simple calculation directly
        tip = self.tip_calculator._calculate_tip_simple(0.1, 0.05, 0.12, 0.06)
        
        self.assertGreaterEqual(tip, 0.0)
        self.assertLessEqual(tip, 1.0)
        
    def test_tip_monte_carlo(self):
        """Test Monte Carlo TIP calculation."""
        tip = self.tip_calculator._calculate_tip_monte_carlo(
            0.1, 0.05, 0.12, 0.06, prediction_uncertainty=0.05
        )
        
        self.assertGreaterEqual(tip, 0.0)
        self.assertLessEqual(tip, 1.0)
        
    def test_tip_with_covariance_error_handling(self):
        """Test TIP calculation with invalid covariance matrix."""
        # Create solutions with invalid covariance
        current_sol = self._create_mock_solution(0.1, 0.05)
        predicted_sol = self._create_mock_solution(0.12, 0.06)
        
        class MockKalmanState:
            def __init__(self):
                # Set invalid covariance matrix (not positive definite)
                self.P = np.array([[1.0, 2.0], [2.0, 1.0], [0.0, 0.0], [0.0, 0.0]])  # Not positive definite
        
        current_sol.P.kalman_state = MockKalmanState()
        predicted_sol.P.kalman_state = MockKalmanState()
        predicted_sol.P.kalman_state.P = np.eye(4) * 0.01
        
        # Should fallback to simple calculation
        tip = self.tip_calculator.calculate_tip(current_sol, predicted_sol)
        
        self.assertGreaterEqual(tip, 0.0)
        self.assertLessEqual(tip, 1.0)
        
    def test_tip_constraints(self):
        """Test that TIP values are properly constrained."""
        # Test with extreme values
        extreme_current = self._create_mock_solution(0.0, 0.0)
        extreme_predicted = self._create_mock_solution(1.0, 1.0)
        
        tip = self.tip_calculator.calculate_tip(extreme_current, extreme_predicted)
        
        # Should be constrained between 0.05 and 0.95
        self.assertGreaterEqual(tip, 0.05)
        self.assertLessEqual(tip, 0.95)


if __name__ == '__main__':
    unittest.main()
