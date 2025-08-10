"""
Tests for portfolio module functionality.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime

from src.portfolio.portfolio import Portfolio, KalmanParams


class TestPortfolio:
    """Test cases for Portfolio class."""
    
    def test_portfolio_creation(self):
        """Test Portfolio object creation."""
        portfolio = Portfolio(5)
        assert portfolio.num_assets == 5
        assert len(portfolio.investment) == 5
        assert np.all(portfolio.investment == 0)
        assert portfolio.ROI == 0.0
        assert portfolio.risk == 0.0
        assert portfolio.cardinality == 0.0
    
    def test_portfolio_init(self):
        """Test portfolio initialization with random weights."""
        portfolio = Portfolio(3)
        portfolio.init()
        
        # Check that weights sum to 1
        assert abs(np.sum(portfolio.investment) - 1.0) < 1e-6
        # Check that all weights are positive
        assert np.all(portfolio.investment >= 0)
    
    def test_portfolio_repr(self):
        """Test Portfolio string representation."""
        portfolio = Portfolio(3)
        portfolio.ROI = 0.05
        portfolio.risk = 0.02
        portfolio.cardinality = 2.0
        
        repr_str = repr(portfolio)
        assert "Portfolio" in repr_str
        assert "0.0500" in repr_str
        assert "0.0200" in repr_str
        assert "2.0" in repr_str


class TestPortfolioCalculations:
    """Test cases for portfolio calculation methods."""
    
    @pytest.fixture
    def sample_returns_data(self):
        """Create sample returns data for testing."""
        np.random.seed(42)
        # Generate realistic returns data
        n_obs = 100
        n_assets = 5
        returns = np.random.normal(0.001, 0.02, (n_obs, n_assets))
        return returns
    
    def test_estimate_assets_mean_ROI(self, sample_returns_data):
        """Test mean ROI estimation."""
        mean_ROI = Portfolio.estimate_assets_mean_ROI(sample_returns_data)
        
        assert len(mean_ROI) == sample_returns_data.shape[1]
        assert np.allclose(mean_ROI, np.mean(sample_returns_data, axis=0))
    
    def test_estimate_assets_median_ROI(self, sample_returns_data):
        """Test median ROI estimation."""
        median_ROI = Portfolio.estimate_assets_median_ROI(sample_returns_data)
        
        assert len(median_ROI) == sample_returns_data.shape[1]
        assert np.allclose(median_ROI, np.median(sample_returns_data, axis=0))
    
    def test_estimate_mean_ROI_single_asset(self, sample_returns_data):
        """Test mean ROI estimation for single asset."""
        asset_idx = 2
        mean_ROI = Portfolio.estimate_mean_ROI(asset_idx, sample_returns_data)
        
        expected = np.mean(sample_returns_data[:, asset_idx])
        assert abs(mean_ROI - expected) < 1e-10
    
    def test_estimate_median_ROI_single_asset(self, sample_returns_data):
        """Test median ROI estimation for single asset."""
        asset_idx = 2
        median_ROI = Portfolio.estimate_median_ROI(asset_idx, sample_returns_data)
        
        expected = np.median(sample_returns_data[:, asset_idx])
        assert abs(median_ROI - expected) < 1e-10
    
    def test_estimate_covariance(self, sample_returns_data):
        """Test covariance estimation."""
        mean_ROI = np.mean(sample_returns_data, axis=0)
        covariance = Portfolio.estimate_covariance(mean_ROI, sample_returns_data)
        
        # Check dimensions
        assert covariance.shape == (sample_returns_data.shape[1], sample_returns_data.shape[1])
        
        # Check symmetry
        assert np.allclose(covariance, covariance.T)
        
        # Check positive definiteness (eigenvalues should be positive)
        eigenvals = np.linalg.eigvals(covariance)
        assert np.all(eigenvals > -1e-10)  # Allow small numerical errors
    
    def test_estimate_robust_covariance(self, sample_returns_data):
        """Test robust covariance estimation."""
        mean_ROI = np.mean(sample_returns_data, axis=0)
        robust_covariance = Portfolio.estimate_robust_covariance(mean_ROI, sample_returns_data)
        
        # Check dimensions
        assert robust_covariance.shape == (sample_returns_data.shape[1], sample_returns_data.shape[1])
        
        # Check symmetry
        assert np.allclose(robust_covariance, robust_covariance.T)
        
        # Check positive definiteness
        eigenvals = np.linalg.eigvals(robust_covariance)
        assert np.all(eigenvals > -1e-10)
    
    def test_compute_ROI(self, sample_returns_data):
        """Test ROI computation."""
        portfolio = Portfolio(5)
        portfolio.investment = np.array([0.2, 0.3, 0.1, 0.2, 0.2])
        
        mean_ROI = np.array([0.01, 0.02, 0.015, 0.025, 0.018])
        roi = Portfolio.compute_ROI(portfolio, mean_ROI)
        
        expected = 0.2 * 0.01 + 0.3 * 0.02 + 0.1 * 0.015 + 0.2 * 0.025 + 0.2 * 0.018
        assert abs(roi - expected) < 1e-10
    
    def test_compute_risk(self, sample_returns_data):
        """Test risk computation."""
        portfolio = Portfolio(3)
        portfolio.investment = np.array([0.4, 0.3, 0.3])
        
        # Create a simple covariance matrix
        covariance = np.array([
            [0.04, 0.01, 0.02],
            [0.01, 0.09, 0.03],
            [0.02, 0.03, 0.16]
        ])
        
        risk = Portfolio.compute_risk(portfolio, covariance)
        
        # Manual calculation
        expected = portfolio.investment @ covariance @ portfolio.investment
        assert abs(risk - expected) < 1e-10
    
    def test_card(self):
        """Test portfolio cardinality calculation."""
        portfolio = Portfolio(5)
        portfolio.investment = np.array([0.3, 0.0, 0.4, 0.0, 0.3])
        
        cardinality = Portfolio.card(portfolio)
        assert cardinality == 3.0  # Only 3 non-zero weights
    
    def test_card_all_zero(self):
        """Test cardinality with all zero weights."""
        portfolio = Portfolio(3)
        portfolio.investment = np.array([0.0, 0.0, 0.0])
        
        cardinality = Portfolio.card(portfolio)
        assert cardinality == 0.0


class TestPortfolioEfficiency:
    """Test cases for portfolio efficiency computations."""
    
    @pytest.fixture
    def setup_portfolio_data(self):
        """Setup portfolio data for efficiency tests."""
        # Set up static data
        Portfolio.available_assets_size = 3
        Portfolio.mean_ROI = np.array([0.01, 0.02, 0.015])
        Portfolio.median_ROI = np.array([0.008, 0.018, 0.012])
        Portfolio.covariance = np.array([
            [0.04, 0.01, 0.02],
            [0.01, 0.09, 0.03],
            [0.02, 0.03, 0.16]
        ])
        Portfolio.robust_covariance = np.array([
            [0.05, 0.015, 0.025],
            [0.015, 0.12, 0.035],
            [0.025, 0.035, 0.20]
        ])
    
    def test_compute_efficiency(self, setup_portfolio_data):
        """Test non-robust efficiency computation."""
        portfolio = Portfolio(3)
        portfolio.investment = np.array([0.4, 0.3, 0.3])
        
        Portfolio.compute_efficiency(portfolio)
        
        # Check that ROI and risk are computed
        assert portfolio.ROI != 0.0
        assert portfolio.risk != 0.0
        assert portfolio.cardinality == 3.0
        
        # Check that robust metrics are also computed
        assert portfolio.robust_ROI != 0.0
        assert portfolio.robust_risk != 0.0
    
    def test_compute_robust_efficiency(self, setup_portfolio_data):
        """Test robust efficiency computation."""
        portfolio = Portfolio(3)
        portfolio.investment = np.array([0.4, 0.3, 0.3])
        
        Portfolio.compute_robust_efficiency(portfolio)
        
        # Check that ROI and risk are computed using robust methods
        assert portfolio.ROI != 0.0
        assert portfolio.risk != 0.0
        assert portfolio.cardinality == 3.0
        
        # Check that non-robust metrics are also computed
        assert portfolio.non_robust_ROI != 0.0
        assert portfolio.non_robust_risk != 0.0


class TestMovingAverages:
    """Test cases for moving average calculations."""
    
    @pytest.fixture
    def setup_autocorrelation(self):
        """Setup autocorrelation data for moving average tests."""
        Portfolio.window_size = 5
        Portfolio.available_assets_size = 3
        
        # Create sample autocorrelation data
        lags = 2 * Portfolio.window_size
        Portfolio.autocorrelation = np.random.random((lags, 3))
        # Ensure some positive weights
        Portfolio.autocorrelation[0, :] = 0.1
    
    def test_moving_average(self, setup_autocorrelation):
        """Test moving average computation."""
        # Create sample returns data
        n_obs = 50
        returns_data = np.random.normal(0.001, 0.02, (n_obs, 3))
        
        ma = Portfolio.moving_average(returns_data)
        
        assert ma.shape == (Portfolio.window_size, 3)
        assert not np.any(np.isnan(ma))
    
    def test_moving_median(self, setup_autocorrelation):
        """Test moving median computation."""
        # Create sample returns data
        n_obs = 50
        returns_data = np.random.normal(0.001, 0.02, (n_obs, 3))
        
        mm = Portfolio.moving_median(returns_data)
        
        assert mm.shape == (Portfolio.window_size, 3)
        assert not np.any(np.isnan(mm))


class TestAutocorrelation:
    """Test cases for autocorrelation calculations."""
    
    def test_sample_autocorrelation(self):
        """Test sample autocorrelation computation."""
        # Create sample returns data
        n_obs = 100
        n_assets = 3
        returns_data = np.random.normal(0.001, 0.02, (n_obs, n_assets))
        
        lags = 10
        Portfolio.sample_autocorrelation(returns_data, lags)
        
        assert Portfolio.autocorrelation.shape == (lags, n_assets)
        assert not np.any(np.isnan(Portfolio.autocorrelation))


class TestStatistics:
    """Test cases for statistics computation."""
    
    def test_compute_statistics(self):
        """Test complete statistics computation."""
        # Create sample returns data
        n_obs = 100
        n_assets = 3
        returns_data = np.random.normal(0.001, 0.02, (n_obs, n_assets))
        
        Portfolio.window_size = 5
        Portfolio.compute_statistics(returns_data)
        
        # Check that all statistics are computed
        assert Portfolio.current_returns_data is not None
        assert Portfolio.mean_ROI is not None
        assert Portfolio.median_ROI is not None
        assert Portfolio.covariance is not None
        assert Portfolio.robust_covariance is not None
        assert Portfolio.autocorrelation is not None
        
        # Check dimensions
        assert Portfolio.mean_ROI.shape == (n_assets,)
        assert Portfolio.median_ROI.shape == (n_assets,)
        assert Portfolio.covariance.shape == (n_assets, n_assets)
        assert Portfolio.robust_covariance.shape == (n_assets, n_assets)
        assert Portfolio.autocorrelation.shape == (2 * Portfolio.window_size, n_assets)


class TestKalmanParams:
    """Test cases for KalmanParams dataclass."""
    
    def test_kalman_params_creation(self):
        """Test KalmanParams object creation."""
        x = np.array([1.0, 2.0])
        P = np.eye(2)
        F = np.eye(2)
        Q = np.eye(2) * 0.1
        R = np.eye(2) * 0.01
        H = np.eye(2)
        
        kalman = KalmanParams(x=x, P=P, F=F, Q=Q, R=R, H=H)
        
        assert np.array_equal(kalman.x, x)
        assert np.array_equal(kalman.P, P)
        assert np.array_equal(kalman.F, F)
        assert np.array_equal(kalman.Q, Q)
        assert np.array_equal(kalman.R, R)
        assert np.array_equal(kalman.H, H) 