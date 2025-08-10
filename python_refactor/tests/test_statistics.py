"""
Tests for advanced statistics functions.
"""

import pytest
import numpy as np
from src.algorithms.statistics import (
    multi_norm, normal_cdf, entropy, linear_entropy,
    compute_correlation_matrix, compute_conditional_statistics,
    compute_stochastic_parameters, compute_non_dominance_probability,
    compute_confidence_alpha
)


class TestMultivariateNormal:
    """Test multivariate normal distribution functions."""
    
    def test_multi_norm_basic(self):
        """Test basic multivariate normal sampling."""
        mu = np.array([0.0, 0.0])
        Sigma = np.array([[1.0, 0.5], [0.5, 1.0]])
        num_samples = 100
        
        samples = multi_norm(mu, Sigma, num_samples)
        
        assert samples.shape == (2, num_samples)
        assert not np.any(np.isnan(samples))
        assert not np.any(np.isinf(samples))
    
    def test_multi_norm_single_sample(self):
        """Test multivariate normal sampling with single sample."""
        mu = np.array([1.0, 2.0])
        Sigma = np.array([[2.0, 0.0], [0.0, 3.0]])
        
        samples = multi_norm(mu, Sigma, 1)
        
        assert samples.shape == (2, 1)
        assert not np.any(np.isnan(samples))
    
    def test_multi_norm_positive_definite(self):
        """Test multivariate normal with positive definite covariance."""
        mu = np.array([0.1, 0.2])
        Sigma = np.array([[0.04, 0.02], [0.02, 0.09]])
        
        samples = multi_norm(mu, Sigma, 50)
        
        assert samples.shape == (2, 50)
        # Check that samples are reasonable
        assert np.all(samples[0, :] > -1.0)  # ROI should be reasonable
        assert np.all(samples[1, :] > -1.0)  # Risk should be reasonable


class TestNormalCDF:
    """Test multivariate normal CDF functions."""
    
    def test_normal_cdf_2d(self):
        """Test 2D multivariate normal CDF."""
        z = np.array([0.0, 0.0])
        Sigma = np.array([[1.0, 0.5], [0.5, 1.0]])
        
        prob = normal_cdf(z, Sigma)
        
        assert 0.0 <= prob <= 1.0
        assert not np.isnan(prob)
        assert not np.isinf(prob)
    
    def test_normal_cdf_different_bounds(self):
        """Test CDF with different upper bounds."""
        Sigma = np.array([[1.0, 0.0], [0.0, 1.0]])
        
        # Test different bounds
        z1 = np.array([1.0, 1.0])
        z2 = np.array([2.0, 2.0])
        
        prob1 = normal_cdf(z1, Sigma)
        prob2 = normal_cdf(z2, Sigma)
        
        assert 0.0 <= prob1 <= 1.0
        assert 0.0 <= prob2 <= 1.0
        assert prob2 > prob1  # Larger bounds should give higher probability


class TestEntropyFunctions:
    """Test entropy calculation functions."""
    
    def test_entropy_basic(self):
        """Test basic entropy calculations."""
        # Test extreme values
        assert entropy(0.0) == 0.0
        assert entropy(1.0) == 0.0
        
        # Test middle value
        mid_entropy = entropy(0.5)
        assert mid_entropy > 0.0
        assert mid_entropy == 1.0  # Maximum entropy for binary case
    
    def test_entropy_values(self):
        """Test entropy for various probability values."""
        # Test symmetry around 0.5
        assert abs(entropy(0.3) - entropy(0.7)) < 1e-10
        
        # Test that entropy is maximum at 0.5
        assert entropy(0.5) > entropy(0.3)
        assert entropy(0.5) > entropy(0.7)
    
    def test_linear_entropy_basic(self):
        """Test basic linear entropy calculations."""
        # Test extreme values
        assert linear_entropy(0.0) == 0.0
        assert linear_entropy(1.0) == 0.0
        
        # Test middle value
        assert linear_entropy(0.5) == 1.0
    
    def test_linear_entropy_values(self):
        """Test linear entropy for various probability values."""
        # Test symmetry around 0.5
        assert abs(linear_entropy(0.3) - linear_entropy(0.7)) < 1e-10
        
        # Test linear behavior
        assert linear_entropy(0.25) == 0.5
        assert linear_entropy(0.75) == 0.5


class TestCorrelationMatrix:
    """Test correlation matrix computation."""
    
    def test_compute_correlation_matrix(self):
        """Test correlation matrix computation."""
        covariance = np.array([[4.0, 2.0], [2.0, 9.0]])
        
        correlation = compute_correlation_matrix(covariance)
        
        assert correlation.shape == (2, 2)
        assert correlation[0, 0] == 1.0  # Diagonal should be 1
        assert correlation[1, 1] == 1.0
        assert correlation[0, 1] == correlation[1, 0]  # Symmetric
        assert abs(correlation[0, 1]) <= 1.0  # Correlation should be between -1 and 1
    
    def test_compute_correlation_matrix_zero_variance(self):
        """Test correlation matrix with zero variance."""
        covariance = np.array([[0.0, 0.0], [0.0, 1.0]])
        
        correlation = compute_correlation_matrix(covariance)
        
        assert correlation.shape == (2, 2)
        assert correlation[0, 0] == 1.0
        assert correlation[1, 1] == 1.0


class TestConditionalStatistics:
    """Test conditional statistics computation."""
    
    def test_compute_conditional_statistics_2d(self):
        """Test conditional statistics for 2D case."""
        covariance = np.array([[4.0, 2.0], [2.0, 9.0]])
        mean = np.array([0.1, 0.2])
        
        # Condition on first variable
        cond_mean, cond_var = compute_conditional_statistics(covariance, mean, 0)
        
        assert isinstance(cond_mean, float)
        assert isinstance(cond_var, float)
        assert cond_var > 0.0
    
    def test_compute_conditional_statistics_1d(self):
        """Test conditional statistics for 1D case."""
        covariance = np.array([[4.0]])
        mean = np.array([0.1])
        
        cond_mean, cond_var = compute_conditional_statistics(covariance, mean, 0)
        
        assert cond_mean == 0.1
        assert cond_var == 4.0


class TestStochasticParameters:
    """Test stochastic parameters computation."""
    
    def test_compute_stochastic_parameters(self):
        """Test stochastic parameters computation."""
        roi = 0.12
        risk = 0.08
        error_covariance = np.array([[0.04, 0.02], [0.02, 0.09]])
        
        params = compute_stochastic_parameters(roi, risk, error_covariance)
        
        # Check required keys
        required_keys = [
            'cov', 'var_roi', 'var_risk', 'corr', 'var_ratio',
            'conditional_mean_roi', 'conditional_var_roi',
            'conditional_mean_risk', 'conditional_var_risk'
        ]
        
        for key in required_keys:
            assert key in params
            assert isinstance(params[key], (int, float))
        
        # Check reasonable values
        assert params['var_roi'] > 0.0
        assert params['var_risk'] > 0.0
        assert abs(params['corr']) <= 1.0
        assert params['var_ratio'] > 0.0
    
    def test_compute_stochastic_parameters_zero_covariance(self):
        """Test stochastic parameters with zero covariance."""
        roi = 0.1
        risk = 0.05
        error_covariance = np.array([[0.01, 0.0], [0.0, 0.02]])
        
        params = compute_stochastic_parameters(roi, risk, error_covariance)
        
        assert params['cov'] == 0.0
        assert params['corr'] == 0.0


class TestNonDominanceProbability:
    """Test non-dominance probability computation."""
    
    def test_compute_non_dominance_probability(self):
        """Test non-dominance probability computation."""
        delta1 = np.array([0.02, 0.01])
        delta2 = np.array([-0.01, -0.02])
        covariance = np.array([[0.04, 0.02], [0.02, 0.09]])
        
        prob = compute_non_dominance_probability(delta1, delta2, covariance)
        
        assert 0.0 <= prob <= 1.0
        assert not np.isnan(prob)
        assert not np.isinf(prob)
    
    def test_compute_non_dominance_probability_extreme(self):
        """Test non-dominance probability with extreme deltas."""
        delta1 = np.array([10.0, 10.0])
        delta2 = np.array([-10.0, -10.0])
        covariance = np.array([[1.0, 0.0], [0.0, 1.0]])
        
        prob = compute_non_dominance_probability(delta1, delta2, covariance)
        
        assert 0.0 <= prob <= 1.0


class TestConfidenceAlpha:
    """Test confidence parameter alpha computation."""
    
    def test_compute_confidence_alpha(self):
        """Test confidence parameter alpha computation."""
        # Test with different probabilities
        prob1 = 0.5
        prob2 = 0.1
        prob3 = 0.9
        
        alpha1 = compute_confidence_alpha(prob1)
        alpha2 = compute_confidence_alpha(prob2)
        alpha3 = compute_confidence_alpha(prob3)
        
        assert 0.0 <= alpha1 <= 1.0
        assert 0.0 <= alpha2 <= 1.0
        assert 0.0 <= alpha3 <= 1.0
        
        # Higher probability should give higher confidence
        assert alpha3 > alpha1
        assert alpha1 > alpha2
        

    
    def test_compute_confidence_alpha_extreme(self):
        """Test confidence alpha with extreme probabilities."""
        alpha0 = compute_confidence_alpha(0.0)
        alpha1 = compute_confidence_alpha(1.0)
        
        assert 0.0 <= alpha0 <= 1.0
        assert 0.0 <= alpha1 <= 1.0


class TestStatisticsIntegration:
    """Test integration of statistics functions."""
    
    def test_stochastic_analysis_workflow(self):
        """Test complete stochastic analysis workflow."""
        # Simulate portfolio data
        roi = 0.15
        risk = 0.08
        error_covariance = np.array([[0.04, 0.02], [0.02, 0.09]])
        
        # Compute stochastic parameters
        params = compute_stochastic_parameters(roi, risk, error_covariance)
        
        # Compute deltas for dominance scenarios
        delta1 = np.array([0.02, 0.01])
        delta2 = np.array([-0.01, -0.02])
        
        # Compute non-dominance probability
        nd_prob = compute_non_dominance_probability(delta1, delta2, error_covariance)
        
        # Compute confidence
        alpha = compute_confidence_alpha(nd_prob)
        
        # Check that all values are reasonable
        assert 0.0 <= nd_prob <= 1.0
        assert 0.0 <= alpha <= 1.0
        assert params['var_roi'] > 0.0
        assert params['var_risk'] > 0.0 