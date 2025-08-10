"""
Tests for Kalman Filter implementation.
"""

import pytest
import numpy as np
from src.algorithms.kalman_filter import (
    KalmanParams, kalman_prediction, kalman_update, kalman_filter,
    initialize_kalman_matrices, create_kalman_params,
    update_measurement_noise, get_portfolio_state, get_portfolio_prediction,
    get_error_covariance, get_prediction_error_covariance
)


class TestKalmanParams:
    """Test KalmanParams dataclass."""
    
    def test_kalman_params_initialization(self):
        """Test KalmanParams initialization with default values."""
        params = KalmanParams()
        
        assert params.x is not None
        assert params.x_next is not None
        assert params.u is not None
        assert params.P is not None
        assert params.P_next is not None
        
        assert params.x.shape == (4,)
        assert params.x_next.shape == (4,)
        assert params.u.shape == (4,)
        assert params.P.shape == (4, 4)
        assert params.P_next.shape == (4, 4)
    
    def test_kalman_params_custom_initialization(self):
        """Test KalmanParams initialization with custom values."""
        x = np.array([1.0, 0.1, 0.5, 0.05])
        P = np.eye(4) * 0.01
        
        params = KalmanParams(x=x, P=P)
        
        np.testing.assert_array_equal(params.x, x)
        np.testing.assert_array_equal(params.P, P)


class TestKalmanMatrices:
    """Test Kalman filter matrix initialization."""
    
    def test_initialize_kalman_matrices(self):
        """Test initialization of Kalman filter matrices."""
        F, H, R = initialize_kalman_matrices()
        
        # Check state transition matrix F (4x4)
        assert F.shape == (4, 4)
        assert F[0, 0] == 1.0  # ROI_next = ROI + ROI_velocity
        assert F[0, 1] == 1.0
        assert F[2, 2] == 1.0  # risk_next = risk + risk_velocity
        assert F[2, 3] == 1.0
        
        # Check measurement matrix H (2x4)
        assert H.shape == (2, 4)
        assert H[0, 0] == 1.0  # Observe ROI
        assert H[1, 2] == 1.0  # Observe risk
        
        # Check measurement noise covariance R (2x2)
        assert R.shape == (2, 2)
        assert R[0, 0] > 0  # ROI measurement noise
        assert R[1, 1] > 0  # Risk measurement noise


class TestKalmanPrediction:
    """Test Kalman filter prediction step."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.params = create_kalman_params(initial_roi=0.1, initial_risk=0.05)
    
    def test_kalman_prediction(self):
        """Test Kalman filter prediction step."""
        # Store initial values
        initial_x = self.params.x.copy()
        initial_P = self.params.P.copy()
        
        # Perform prediction
        kalman_prediction(self.params)
        
        # Check that prediction follows state transition model
        expected_x_next = self.params.F @ initial_x + self.params.u
        np.testing.assert_array_almost_equal(self.params.x_next, expected_x_next)
        
        # Check that covariance prediction is correct
        expected_P_next = self.params.F @ initial_P @ self.params.F.T
        np.testing.assert_array_almost_equal(self.params.P_next, expected_P_next)
        
        # Check that covariance was updated (should be different)
        assert not np.array_equal(self.params.P_next, initial_P)


class TestKalmanUpdate:
    """Test Kalman filter update step."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.params = create_kalman_params(initial_roi=0.1, initial_risk=0.05)
        # Perform initial prediction
        kalman_prediction(self.params)
    
    def test_kalman_update(self):
        """Test Kalman filter update step."""
        # Store prediction values
        x_next = self.params.x_next.copy()
        P_next = self.params.P_next.copy()
        
        # Create measurement
        measurement = np.array([0.12, 0.06])  # [ROI, risk]
        
        # Perform update
        kalman_update(self.params, measurement)
        
        # Check that state was updated
        assert not np.array_equal(self.params.x, x_next)
        assert not np.array_equal(self.params.P, P_next)
        
        # Check that state is reasonable
        assert 0.0 <= self.params.x[0] <= 1.0  # ROI should be positive
        assert 0.0 <= self.params.x[2] <= 1.0  # Risk should be positive


class TestKalmanFilter:
    """Test complete Kalman filter."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.params = create_kalman_params(initial_roi=0.1, initial_risk=0.05)
    
    def test_kalman_filter_complete(self):
        """Test complete Kalman filter step."""
        # Store initial values
        initial_x = self.params.x.copy()
        initial_P = self.params.P.copy()
        
        # Create measurement
        measurement = np.array([0.12, 0.06])
        
        # Perform complete filter step
        kalman_filter(self.params, measurement)
        
        # Check that both prediction and update occurred
        assert not np.array_equal(self.params.x, initial_x)
        assert not np.array_equal(self.params.P, initial_P)
        
        # Check that next state was computed
        assert self.params.x_next is not None
        assert self.params.P_next is not None


class TestKalmanUtilities:
    """Test Kalman filter utility functions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.params = create_kalman_params(initial_roi=0.1, initial_risk=0.05)
    
    def test_create_kalman_params(self):
        """Test creation of Kalman parameters."""
        params = create_kalman_params(initial_roi=0.15, initial_risk=0.08)
        
        # Check initial state
        assert params.x[0] == 0.15  # ROI
        assert params.x[2] == 0.08  # Risk
        assert params.x[1] == 0.0   # ROI velocity
        assert params.x[3] == 0.0   # Risk velocity
        
        # Check that matrices are set
        assert params.F is not None
        assert params.H is not None
        assert params.R is not None
    
    def test_update_measurement_noise(self):
        """Test updating measurement noise covariance."""
        roi_variance = 0.02
        risk_variance = 0.03
        covariance = 0.005
        
        update_measurement_noise(self.params, roi_variance, risk_variance, covariance)
        
        expected_R = np.array([
            [roi_variance, covariance],
            [covariance, risk_variance]
        ])
        
        np.testing.assert_array_equal(self.params.R, expected_R)
    
    def test_get_portfolio_state(self):
        """Test extracting portfolio state from Kalman filter."""
        roi, risk = get_portfolio_state(self.params)
        
        assert roi == self.params.x[0]
        assert risk == self.params.x[2]
    
    def test_get_portfolio_prediction(self):
        """Test extracting portfolio prediction from Kalman filter."""
        # Perform prediction first
        kalman_prediction(self.params)
        
        roi_pred, risk_pred = get_portfolio_prediction(self.params)
        
        assert roi_pred == self.params.x_next[0]
        assert risk_pred == self.params.x_next[2]
    
    def test_get_error_covariance(self):
        """Test getting error covariance matrix."""
        covar = get_error_covariance(self.params)
        
        np.testing.assert_array_equal(covar, self.params.P)
        assert covar is not self.params.P  # Should be a copy
    
    def test_get_prediction_error_covariance(self):
        """Test getting prediction error covariance matrix."""
        # Perform prediction first
        kalman_prediction(self.params)
        
        covar = get_prediction_error_covariance(self.params)
        
        np.testing.assert_array_equal(covar, self.params.P_next)
        assert covar is not self.params.P_next  # Should be a copy


class TestKalmanFilterIntegration:
    """Test Kalman filter integration scenarios."""
    
    def test_multiple_filter_steps(self):
        """Test multiple Kalman filter steps."""
        params = create_kalman_params(initial_roi=0.1, initial_risk=0.05)
        
        measurements = [
            np.array([0.12, 0.06]),
            np.array([0.11, 0.07]),
            np.array([0.13, 0.05])
        ]
        
        for measurement in measurements:
            kalman_filter(params, measurement)
            
            # Check that state is reasonable
            assert 0.0 <= params.x[0] <= 1.0  # ROI
            assert 0.0 <= params.x[2] <= 1.0  # Risk
            assert params.P[0, 0] > 0  # Positive variance
            assert params.P[2, 2] > 0  # Positive variance
    
    def test_kalman_filter_convergence(self):
        """Test that Kalman filter converges with consistent measurements."""
        params = create_kalman_params(initial_roi=0.1, initial_risk=0.05)
        
        # Consistent measurement
        measurement = np.array([0.12, 0.06])
        
        # Multiple filter steps
        for _ in range(10):
            kalman_filter(params, measurement)
        
        # Check convergence (state should be close to measurement)
        np.testing.assert_array_almost_equal(
            params.x[[0, 2]], measurement, decimal=2
        ) 