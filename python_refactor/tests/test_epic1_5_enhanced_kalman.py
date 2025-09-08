"""
Unit tests for EPIC 1.5: Enhanced Kalman Filter

Tests the enhanced Kalman filter implementation including:
- Enhanced state space model
- Regime-integrated Kalman filter
- Advanced parameter estimation
- Enhanced uncertainty quantification
- Performance optimizations
"""

import unittest
import numpy as np
import pandas as pd
import sys
import os
from typing import Dict
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from algorithms.enhanced_kalman_filter import (
    EnhancedKalmanFilter, EnhancedStateSpaceModel, KalmanParameterEstimator,
    KalmanUncertaintyQuantifier, EnhancedPredictionResult, EnhancedUpdateResult,
    KalmanParameters, create_enhanced_kalman_filter
)
from algorithms.regime_integrated_kalman import (
    RegimeIntegratedKalmanFilter, RegimeSpecificKalmanModel, RegimeAwareResult,
    create_regime_integrated_kalman
)
from algorithms.regime_detection_bnn import RegimeDetectionResult
from algorithms.regime_detection_bnn import MarketRegimeDetectionBNN


class TestEnhancedStateSpaceModel(unittest.TestCase):
    """Test cases for EnhancedStateSpaceModel class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.state_model = EnhancedStateSpaceModel(state_dim=4, observation_dim=2)
    
    def test_initialization(self):
        """Test initialization of enhanced state space model."""
        self.assertEqual(self.state_model.state_dim, 4)
        self.assertEqual(self.state_model.observation_dim, 2)
        self.assertIn('mean_reversion', self.state_model.dynamics_params)
        self.assertIn('volatility_clustering', self.state_model.dynamics_params)
        self.assertIn('cross_correlation', self.state_model.dynamics_params)
        self.assertIn('momentum_persistence', self.state_model.dynamics_params)
    
    def test_create_enhanced_transition_matrix(self):
        """Test enhanced transition matrix creation."""
        F = self.state_model.create_enhanced_transition_matrix(dt=1.0)
        
        self.assertEqual(F.shape, (4, 4))
        self.assertFalse(np.any(np.isnan(F)))
        self.assertFalse(np.any(np.isinf(F)))
        
        # Check that it's a valid transition matrix
        self.assertGreater(np.linalg.det(F), 0)
    
    def test_create_enhanced_process_noise(self):
        """Test enhanced process noise creation."""
        for regime in ['bull_market', 'bear_market', 'sideways_market']:
            Q = self.state_model.create_enhanced_process_noise(dt=1.0, regime=regime)
            
            self.assertEqual(Q.shape, (4, 4))
            self.assertTrue(np.allclose(Q, Q.T))  # Symmetric
            self.assertTrue(np.all(np.linalg.eigvals(Q) >= 0))  # Positive semi-definite
    
    def test_create_enhanced_measurement_matrix(self):
        """Test enhanced measurement matrix creation."""
        H = self.state_model.create_enhanced_measurement_matrix()
        
        self.assertEqual(H.shape, (2, 4))
        self.assertFalse(np.any(np.isnan(H)))
        self.assertFalse(np.any(np.isinf(H)))
    
    def test_create_enhanced_measurement_noise(self):
        """Test enhanced measurement noise creation."""
        for regime in ['bull_market', 'bear_market', 'sideways_market']:
            R = self.state_model.create_enhanced_measurement_noise(regime=regime)
            
            self.assertEqual(R.shape, (2, 2))
            self.assertTrue(np.allclose(R, R.T))  # Symmetric
            self.assertTrue(np.all(np.linalg.eigvals(R) >= 0))  # Positive semi-definite


class TestKalmanParameterEstimator(unittest.TestCase):
    """Test cases for KalmanParameterEstimator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.estimator = KalmanParameterEstimator()
        
        # Create test parameters
        self.test_params = KalmanParameters(
            F=np.eye(4),
            H=np.array([[1, 0, 0, 0], [0, 1, 0, 0]]),
            Q=np.eye(4) * 0.01,
            R=np.eye(2) * 0.005,
            P0=np.eye(4) * 0.1
        )
        
        # Create test observations
        self.test_observations = np.random.randn(100, 2)
    
    def test_initialization(self):
        """Test initialization of parameter estimator."""
        self.assertEqual(len(self.estimator.estimation_history), 0)
        self.assertEqual(len(self.estimator.parameter_history), 0)
    
    def test_maximum_likelihood_estimation(self):
        """Test maximum likelihood parameter estimation."""
        estimated_params = self.estimator.maximum_likelihood_estimation(
            self.test_observations, self.test_params
        )
        
        self.assertIsInstance(estimated_params, KalmanParameters)
        self.assertEqual(estimated_params.F.shape, (4, 4))
        self.assertEqual(estimated_params.H.shape, (2, 4))
        self.assertEqual(estimated_params.Q.shape, (4, 4))
        self.assertEqual(estimated_params.R.shape, (2, 2))
    
    def test_bayesian_parameter_estimation(self):
        """Test Bayesian parameter estimation."""
        estimated_params = self.estimator.bayesian_parameter_estimation(
            self.test_observations, self.test_params
        )
        
        self.assertIsInstance(estimated_params, KalmanParameters)
        self.assertEqual(estimated_params.F.shape, (4, 4))
        self.assertEqual(estimated_params.H.shape, (2, 4))
        self.assertEqual(estimated_params.Q.shape, (4, 4))
        self.assertEqual(estimated_params.R.shape, (2, 2))
    
    def test_online_parameter_update(self):
        """Test online parameter update."""
        observation = np.array([0.1, 0.05])
        innovation = np.array([0.02, 0.01])
        
        updated_params = self.estimator.online_parameter_update(
            observation, self.test_params, innovation
        )
        
        self.assertIsInstance(updated_params, KalmanParameters)
        self.assertEqual(len(self.estimator.parameter_history), 1)


class TestKalmanUncertaintyQuantifier(unittest.TestCase):
    """Test cases for KalmanUncertaintyQuantifier class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.quantifier = KalmanUncertaintyQuantifier(num_ensemble_samples=50)
        
        # Test data
        self.test_prediction = np.array([0.1, 0.05, 0.001, 0.0005])
        self.test_covariance = np.eye(4) * 0.01
    
    def test_initialization(self):
        """Test initialization of uncertainty quantifier."""
        self.assertEqual(self.quantifier.num_ensemble_samples, 50)
        self.assertEqual(len(self.quantifier.uncertainty_history), 0)
    
    def test_ensemble_uncertainty(self):
        """Test ensemble uncertainty calculation."""
        uncertainty = self.quantifier.ensemble_uncertainty(
            self.test_prediction, self.test_covariance
        )
        
        self.assertEqual(len(uncertainty), 4)
        self.assertTrue(np.all(uncertainty >= 0))
        self.assertFalse(np.any(np.isnan(uncertainty)))
    
    def test_bootstrap_uncertainty(self):
        """Test bootstrap uncertainty calculation."""
        uncertainty = self.quantifier.bootstrap_uncertainty(
            self.test_prediction, self.test_covariance, num_bootstrap=20
        )
        
        self.assertEqual(len(uncertainty), 4)
        self.assertTrue(np.all(uncertainty >= 0))
        self.assertFalse(np.any(np.isnan(uncertainty)))
    
    def test_bayesian_uncertainty(self):
        """Test Bayesian uncertainty calculation."""
        prior_covariance = np.eye(4) * 0.02
        
        uncertainty = self.quantifier.bayesian_uncertainty(
            self.test_prediction, self.test_covariance, prior_covariance
        )
        
        self.assertEqual(len(uncertainty), 4)
        self.assertTrue(np.all(uncertainty >= 0))
        self.assertFalse(np.any(np.isnan(uncertainty)))
    
    def test_calculate_prediction_intervals(self):
        """Test prediction interval calculation."""
        uncertainty = np.array([0.01, 0.005, 0.001, 0.0005])
        confidence_levels = [0.68, 0.95, 0.99]
        
        intervals = self.quantifier.calculate_prediction_intervals(
            self.test_prediction, uncertainty, confidence_levels
        )
        
        self.assertEqual(len(intervals), 3)
        for confidence in ['68%', '95%', '99%']:
            self.assertIn(confidence, intervals)
            self.assertIn('lower', intervals[confidence])
            self.assertIn('upper', intervals[confidence])


class TestEnhancedKalmanFilter(unittest.TestCase):
    """Test cases for EnhancedKalmanFilter class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.kalman_filter = EnhancedKalmanFilter(state_dim=4, observation_dim=2)
        self.kalman_filter.initialize_parameters('sideways_market')
    
    def test_initialization(self):
        """Test initialization of enhanced Kalman filter."""
        self.assertEqual(self.kalman_filter.state_dim, 4)
        self.assertEqual(self.kalman_filter.observation_dim, 2)
        self.assertIsNotNone(self.kalman_filter.state_model)
        self.assertIsNotNone(self.kalman_filter.parameter_estimator)
        self.assertIsNotNone(self.kalman_filter.uncertainty_quantifier)
        self.assertIsNotNone(self.kalman_filter.parameters)
    
    def test_enhanced_prediction(self):
        """Test enhanced prediction."""
        current_state = np.array([0.1, 0.05, 0.001, 0.0005])
        
        result = self.kalman_filter.enhanced_prediction(current_state, 'sideways_market')
        
        self.assertIsInstance(result, EnhancedPredictionResult)
        self.assertEqual(len(result.prediction), 4)
        self.assertEqual(result.covariance.shape, (4, 4))
        self.assertEqual(len(result.uncertainty), 4)
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)
        self.assertIsNotNone(result.prediction_interval)
    
    def test_adaptive_update(self):
        """Test adaptive update."""
        current_state = np.array([0.1, 0.05, 0.001, 0.0005])
        observation = np.array([0.12, 0.06])
        
        # First make a prediction
        prediction = self.kalman_filter.enhanced_prediction(current_state, 'sideways_market')
        
        # Then update
        update_result = self.kalman_filter.adaptive_update(observation, prediction)
        
        self.assertIsInstance(update_result, EnhancedUpdateResult)
        self.assertEqual(len(update_result.updated_state), 4)
        self.assertEqual(update_result.updated_covariance.shape, (4, 4))
        self.assertEqual(len(update_result.innovation), 2)
        self.assertEqual(update_result.innovation_covariance.shape, (2, 2))
        self.assertEqual(update_result.kalman_gain.shape, (4, 2))
        self.assertIsInstance(update_result.log_likelihood, float)
    
    def test_get_performance_statistics(self):
        """Test performance statistics."""
        # Initially no statistics
        stats = self.kalman_filter.get_performance_statistics()
        self.assertIn('error', stats)
        
        # Make some predictions and updates
        current_state = np.array([0.1, 0.05, 0.001, 0.0005])
        observation = np.array([0.12, 0.06])
        
        for _ in range(5):
            prediction = self.kalman_filter.enhanced_prediction(current_state, 'sideways_market')
            self.kalman_filter.adaptive_update(observation, prediction)
        
        # Get statistics
        stats = self.kalman_filter.get_performance_statistics()
        
        self.assertIn('total_predictions', stats)
        self.assertIn('total_updates', stats)
        self.assertIn('average_confidence', stats)
        self.assertIn('average_uncertainty', stats)
        self.assertIn('current_regime', stats)
        
        self.assertEqual(stats['total_predictions'], 5)
        self.assertEqual(stats['total_updates'], 5)
    
    def test_reset_history(self):
        """Test history reset."""
        # Make some predictions
        current_state = np.array([0.1, 0.05, 0.001, 0.0005])
        
        for _ in range(3):
            self.kalman_filter.enhanced_prediction(current_state, 'sideways_market')
        
        # Check history exists
        self.assertGreater(len(self.kalman_filter.prediction_history), 0)
        
        # Reset history
        self.kalman_filter.reset_history()
        
        # Check history is cleared
        self.assertEqual(len(self.kalman_filter.prediction_history), 0)
        self.assertEqual(len(self.kalman_filter.update_history), 0)


class TestRegimeSpecificKalmanModel(unittest.TestCase):
    """Test cases for RegimeSpecificKalmanModel class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.bull_model = RegimeSpecificKalmanModel('bull_market')
        self.bear_model = RegimeSpecificKalmanModel('bear_market')
        self.sideways_model = RegimeSpecificKalmanModel('sideways_market')
        
        # Mock regime info
        from algorithms.regime_detection_bnn import RegimeDetectionResult
        self.mock_regime_info = RegimeDetectionResult(
            regime_probabilities=np.array([0.8, 0.1, 0.1]),
            regime_uncertainty=np.array([0.1, 0.05, 0.05]),
            predicted_regime='bull_market',
            confidence=0.8,
            timestamp=datetime.now().timestamp()
        )
    
    def test_initialization(self):
        """Test initialization of regime-specific models."""
        for model in [self.bull_model, self.bear_model, self.sideways_model]:
            self.assertEqual(model.state_dim, 4)
            self.assertEqual(model.observation_dim, 2)
            self.assertIn('mean_reversion', model.regime_params)
            self.assertIn('volatility_clustering', model.regime_params)
            self.assertIn('cross_correlation', model.regime_params)
            self.assertIn('momentum_persistence', model.regime_params)
    
    def test_regime_parameters_differences(self):
        """Test that different regimes have different parameters."""
        bull_params = self.bull_model.regime_params
        bear_params = self.bear_model.regime_params
        sideways_params = self.sideways_model.regime_params
        
        # Bull market should have lower mean reversion than bear market
        self.assertLess(bull_params['mean_reversion'], bear_params['mean_reversion'])
        
        # Bull market should have higher momentum persistence than bear market
        self.assertGreater(bull_params['momentum_persistence'], bear_params['momentum_persistence'])
        
        # Bull market should have lower process noise than bear market
        self.assertLess(bull_params['process_noise_scale'], bear_params['process_noise_scale'])
    
    def test_create_regime_specific_parameters(self):
        """Test creation of regime-specific parameters."""
        for model in [self.bull_model, self.bear_model, self.sideways_model]:
            parameters = model.create_regime_specific_parameters()
            
            self.assertIsInstance(parameters, KalmanParameters)
            self.assertEqual(parameters.F.shape, (4, 4))
            self.assertEqual(parameters.H.shape, (2, 4))
            self.assertEqual(parameters.Q.shape, (4, 4))
            self.assertEqual(parameters.R.shape, (2, 2))
    
    def test_predict(self):
        """Test regime-specific prediction."""
        current_state = np.array([0.1, 0.05, 0.001, 0.0005])
        
        for model in [self.bull_model, self.bear_model, self.sideways_model]:
            result = model.predict(current_state, self.mock_regime_info)
            
            self.assertIsInstance(result, EnhancedPredictionResult)
            self.assertEqual(len(result.prediction), 4)
            self.assertEqual(result.covariance.shape, (4, 4))
            self.assertGreaterEqual(result.confidence, 0.0)
            self.assertLessEqual(result.confidence, 1.0)
    
    def test_update_performance(self):
        """Test performance update."""
        current_state = np.array([0.1, 0.05, 0.001, 0.0005])
        observation = np.array([0.12, 0.06])
        
        # Make prediction
        prediction = self.bull_model.predict(current_state, self.mock_regime_info)
        
        # Update performance
        self.bull_model.update_performance(observation, prediction)
        
        # Check performance was recorded
        self.assertEqual(len(self.bull_model.prediction_accuracy), 1)
        self.assertGreater(self.bull_model.prediction_accuracy[0], 0)
    
    def test_get_regime_performance(self):
        """Test regime performance statistics."""
        # Initially no performance data
        stats = self.bull_model.get_regime_performance()
        self.assertIn('error', stats)
        
        # Add some performance data
        current_state = np.array([0.1, 0.05, 0.001, 0.0005])
        observation = np.array([0.12, 0.06])
        
        for _ in range(5):
            prediction = self.bull_model.predict(current_state, self.mock_regime_info)
            self.bull_model.update_performance(observation, prediction)
        
        # Get performance statistics
        stats = self.bull_model.get_regime_performance()
        
        self.assertIn('regime', stats)
        self.assertIn('average_prediction_error', stats)
        self.assertIn('total_predictions', stats)
        self.assertEqual(stats['regime'], 'bull_market')
        self.assertEqual(stats['total_predictions'], 5)


class TestRegimeIntegratedKalmanFilter(unittest.TestCase):
    """Test cases for RegimeIntegratedKalmanFilter class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock regime detector
        self.regime_detector = MarketRegimeDetectionBNN(input_dim=20, num_regimes=3)
        
        # Create mock data for regime detector
        self.mock_data = self._create_mock_market_data()
        self.regime_detector.fit(self.mock_data)
        
        # Create regime-integrated Kalman filter
        self.kalman_filter = RegimeIntegratedKalmanFilter(self.regime_detector)
    
    def _create_mock_market_data(self) -> Dict[str, pd.DataFrame]:
        """Create mock market data for testing."""
        dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
        
        # Create different market regimes
        bull_market_data = self._create_regime_data(dates[:365], trend=0.001, volatility=0.01)
        bear_market_data = self._create_regime_data(dates[365:730], trend=-0.001, volatility=0.015)
        sideways_market_data = self._create_regime_data(dates[730:], trend=0.0001, volatility=0.008)
        
        # Combine data
        combined_data = pd.concat([bull_market_data, bear_market_data, sideways_market_data])
        
        return {
            'test_asset': combined_data,
            'test_asset_2': combined_data * 1.1
        }
    
    def _create_regime_data(self, dates: pd.DatetimeIndex, trend: float, 
                          volatility: float) -> pd.DataFrame:
        """Create data for a specific market regime."""
        np.random.seed(42)
        
        returns = np.random.normal(trend, volatility, len(dates))
        prices = 100 * np.exp(np.cumsum(returns))
        
        data = pd.DataFrame({
            'Date': dates,
            'Open': prices * (1 + np.random.normal(0, 0.001, len(dates))),
            'High': prices * (1 + np.abs(np.random.normal(0, 0.005, len(dates)))),
            'Low': prices * (1 - np.abs(np.random.normal(0, 0.005, len(dates)))),
            'Close': prices,
            'Adj Close': prices,
            'Volume': np.random.randint(1000000, 10000000, len(dates))
        })
        
        return data.set_index('Date')
    
    def test_initialization(self):
        """Test initialization of regime-integrated Kalman filter."""
        self.assertIsNotNone(self.kalman_filter.regime_detector)
        self.assertEqual(len(self.kalman_filter.regime_models), 3)
        self.assertIn('bull_market', self.kalman_filter.regime_models)
        self.assertIn('bear_market', self.kalman_filter.regime_models)
        self.assertIn('sideways_market', self.kalman_filter.regime_models)
        self.assertEqual(self.kalman_filter.current_regime, 'sideways_market')
    
    def test_regime_aware_prediction(self):
        """Test regime-aware prediction."""
        current_state = np.array([0.1, 0.05, 0.001, 0.0005])
        market_features = np.random.randn(20)
        
        result = self.kalman_filter.regime_aware_prediction(current_state, market_features)
        
        self.assertIsInstance(result, RegimeAwareResult)
        self.assertEqual(len(result.prediction), 4)
        self.assertEqual(result.covariance.shape, (4, 4))
        self.assertEqual(len(result.uncertainty), 4)
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)
        self.assertIsInstance(result.regime_info, RegimeDetectionResult)
        self.assertEqual(len(result.regime_specific_prediction), 4)
        self.assertGreaterEqual(result.regime_transition_probability, 0.0)
        self.assertLessEqual(result.regime_transition_probability, 1.0)
        self.assertEqual(len(result.multi_regime_predictions), 3)
    
    def test_regime_aware_update(self):
        """Test regime-aware update."""
        current_state = np.array([0.1, 0.05, 0.001, 0.0005])
        market_features = np.random.randn(20)
        observation = np.array([0.12, 0.06])
        
        # First make a prediction
        prediction = self.kalman_filter.regime_aware_prediction(current_state, market_features)
        
        # Then update
        update_result = self.kalman_filter.regime_aware_update(observation, prediction)
        
        self.assertIsInstance(update_result, EnhancedUpdateResult)
        self.assertEqual(len(update_result.updated_state), 4)
        self.assertEqual(update_result.updated_covariance.shape, (4, 4))
        self.assertEqual(len(update_result.innovation), 2)
        self.assertIsInstance(update_result.log_likelihood, float)
    
    def test_predict_multiple_horizons(self):
        """Test multiple horizon prediction."""
        current_state = np.array([0.1, 0.05, 0.001, 0.0005])
        market_features = np.random.randn(20)
        horizons = [1, 2, 3]
        
        results = self.kalman_filter.predict_multiple_horizons(current_state, market_features, horizons)
        
        self.assertEqual(len(results), 3)
        for horizon in horizons:
            self.assertIn(horizon, results)
            self.assertIsInstance(results[horizon], RegimeAwareResult)
    
    def test_get_regime_statistics(self):
        """Test regime statistics."""
        # Initially no statistics
        stats = self.kalman_filter.get_regime_statistics()
        self.assertIn('error', stats)
        
        # Make some predictions
        current_state = np.array([0.1, 0.05, 0.001, 0.0005])
        market_features = np.random.randn(20)
        
        for _ in range(5):
            self.kalman_filter.regime_aware_prediction(current_state, market_features)
        
        # Get statistics
        stats = self.kalman_filter.get_regime_statistics()
        
        self.assertIn('total_predictions', stats)
        self.assertIn('regime_distribution', stats)
        self.assertIn('regime_transitions', stats)
        self.assertIn('average_confidence', stats)
        self.assertIn('current_regime', stats)
        self.assertIn('regime_stability', stats)
        self.assertIn('regime_performance', stats)
        
        self.assertEqual(stats['total_predictions'], 5)
    
    def test_reset_history(self):
        """Test history reset."""
        # Make some predictions
        current_state = np.array([0.1, 0.05, 0.001, 0.0005])
        market_features = np.random.randn(20)
        
        for _ in range(3):
            self.kalman_filter.regime_aware_prediction(current_state, market_features)
        
        # Check history exists
        self.assertGreater(len(self.kalman_filter.prediction_history), 0)
        self.assertGreater(len(self.kalman_filter.regime_history), 0)
        
        # Reset history
        self.kalman_filter.reset_history()
        
        # Check history is cleared
        self.assertEqual(len(self.kalman_filter.prediction_history), 0)
        self.assertEqual(len(self.kalman_filter.update_history), 0)
        self.assertEqual(len(self.kalman_filter.regime_history), 0)
        self.assertEqual(len(self.kalman_filter.regime_transition_history), 0)


class TestConvenienceFunctions(unittest.TestCase):
    """Test cases for convenience functions."""
    
    def test_create_enhanced_kalman_filter(self):
        """Test enhanced Kalman filter creation."""
        filter_instance = create_enhanced_kalman_filter(
            state_dim=4, observation_dim=2, regime='bull_market'
        )
        
        self.assertIsInstance(filter_instance, EnhancedKalmanFilter)
        self.assertEqual(filter_instance.state_dim, 4)
        self.assertEqual(filter_instance.observation_dim, 2)
        self.assertIsNotNone(filter_instance.parameters)
    
    def test_create_regime_integrated_kalman(self):
        """Test regime-integrated Kalman filter creation."""
        # Create mock regime detector
        regime_detector = MarketRegimeDetectionBNN(input_dim=20, num_regimes=3)
        
        # Create mock data
        dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
        bull_data = self._create_regime_data(dates[:365], trend=0.001, volatility=0.01)
        bear_data = self._create_regime_data(dates[365:730], trend=-0.001, volatility=0.015)
        sideways_data = self._create_regime_data(dates[730:], trend=0.0001, volatility=0.008)
        
        combined_data = pd.concat([bull_data, bear_data, sideways_data])
        mock_data = {'test_asset': combined_data}
        
        regime_detector.fit(mock_data)
        
        # Create regime-integrated filter
        filter_instance = create_regime_integrated_kalman(regime_detector)
        
        self.assertIsInstance(filter_instance, RegimeIntegratedKalmanFilter)
        self.assertIsNotNone(filter_instance.regime_detector)
        self.assertEqual(len(filter_instance.regime_models), 3)
    
    def _create_regime_data(self, dates: pd.DatetimeIndex, trend: float, 
                          volatility: float) -> pd.DataFrame:
        """Create data for a specific market regime."""
        np.random.seed(42)
        
        returns = np.random.normal(trend, volatility, len(dates))
        prices = 100 * np.exp(np.cumsum(returns))
        
        data = pd.DataFrame({
            'Date': dates,
            'Open': prices * (1 + np.random.normal(0, 0.001, len(dates))),
            'High': prices * (1 + np.abs(np.random.normal(0, 0.005, len(dates)))),
            'Low': prices * (1 - np.abs(np.random.normal(0, 0.005, len(dates)))),
            'Close': prices,
            'Adj Close': prices,
            'Volume': np.random.randint(1000000, 10000000, len(dates))
        })
        
        return data.set_index('Date')


class TestEPIC1_5Integration(unittest.TestCase):
    """Integration tests for EPIC 1.5 components."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock regime detector
        self.regime_detector = MarketRegimeDetectionBNN(input_dim=20, num_regimes=3)
        
        # Create mock data
        self.mock_data = self._create_mock_market_data()
        self.regime_detector.fit(self.mock_data)
        
        # Create enhanced Kalman filter
        self.enhanced_kalman = EnhancedKalmanFilter()
        self.enhanced_kalman.initialize_parameters('sideways_market')
        
        # Create regime-integrated Kalman filter
        self.regime_kalman = RegimeIntegratedKalmanFilter(self.regime_detector)
    
    def _create_mock_market_data(self) -> Dict[str, pd.DataFrame]:
        """Create mock market data for testing."""
        dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
        
        bull_data = self._create_regime_data(dates[:365], trend=0.001, volatility=0.01)
        bear_data = self._create_regime_data(dates[365:730], trend=-0.001, volatility=0.015)
        sideways_data = self._create_regime_data(dates[730:], trend=0.0001, volatility=0.008)
        
        combined_data = pd.concat([bull_data, bear_data, sideways_data])
        
        return {
            'test_asset': combined_data,
            'test_asset_2': combined_data * 1.1
        }
    
    def _create_regime_data(self, dates: pd.DatetimeIndex, trend: float, 
                          volatility: float) -> pd.DataFrame:
        """Create data for a specific market regime."""
        np.random.seed(42)
        
        returns = np.random.normal(trend, volatility, len(dates))
        prices = 100 * np.exp(np.cumsum(returns))
        
        data = pd.DataFrame({
            'Date': dates,
            'Open': prices * (1 + np.random.normal(0, 0.001, len(dates))),
            'High': prices * (1 + np.abs(np.random.normal(0, 0.005, len(dates)))),
            'Low': prices * (1 - np.abs(np.random.normal(0, 0.005, len(dates)))),
            'Close': prices,
            'Adj Close': prices,
            'Volume': np.random.randint(1000000, 10000000, len(dates))
        })
        
        return data.set_index('Date')
    
    def test_full_epic1_5_workflow(self):
        """Test complete EPIC 1.5 workflow."""
        # Step 1: Enhanced Kalman filter prediction
        current_state = np.array([0.1, 0.05, 0.001, 0.0005])
        enhanced_prediction = self.enhanced_kalman.enhanced_prediction(current_state, 'sideways_market')
        
        self.assertIsInstance(enhanced_prediction, EnhancedPredictionResult)
        self.assertEqual(len(enhanced_prediction.prediction), 4)
        self.assertIsNotNone(enhanced_prediction.prediction_interval)
        
        # Step 2: Enhanced Kalman filter update
        observation = np.array([0.12, 0.06])
        enhanced_update = self.enhanced_kalman.adaptive_update(observation, enhanced_prediction)
        
        self.assertIsInstance(enhanced_update, EnhancedUpdateResult)
        self.assertEqual(len(enhanced_update.updated_state), 4)
        self.assertIsInstance(enhanced_update.log_likelihood, float)
        
        # Step 3: Regime-aware prediction
        market_features = np.random.randn(20)
        regime_prediction = self.regime_kalman.regime_aware_prediction(current_state, market_features)
        
        self.assertIsInstance(regime_prediction, RegimeAwareResult)
        self.assertIsInstance(regime_prediction.regime_info, RegimeDetectionResult)
        self.assertEqual(len(regime_prediction.multi_regime_predictions), 3)
        
        # Step 4: Regime-aware update
        regime_update = self.regime_kalman.regime_aware_update(observation, regime_prediction)
        
        self.assertIsInstance(regime_update, EnhancedUpdateResult)
        self.assertEqual(len(regime_update.updated_state), 4)
        
        # Step 5: Performance comparison
        enhanced_stats = self.enhanced_kalman.get_performance_statistics()
        regime_stats = self.regime_kalman.get_regime_statistics()
        
        self.assertIn('total_predictions', enhanced_stats)
        self.assertIn('total_predictions', regime_stats)
    
    def test_epic1_5_performance_validation(self):
        """Test EPIC 1.5 performance validation."""
        # Create test data
        test_states = []
        test_observations = []
        test_features = []
        
        for _ in range(20):
            state = np.random.randn(4)
            observation = np.random.randn(2)
            features = np.random.randn(20)
            
            test_states.append(state)
            test_observations.append(observation)
            test_features.append(features)
        
        # Test enhanced Kalman filter
        enhanced_predictions = []
        for state, observation in zip(test_states, test_observations):
            prediction = self.enhanced_kalman.enhanced_prediction(state, 'sideways_market')
            enhanced_predictions.append(prediction.prediction)
            
            update = self.enhanced_kalman.adaptive_update(observation, prediction)
        
        # Test regime-integrated Kalman filter
        regime_predictions = []
        for state, observation, features in zip(test_states, test_observations, test_features):
            prediction = self.regime_kalman.regime_aware_prediction(state, features)
            regime_predictions.append(prediction.prediction)
            
            update = self.regime_kalman.regime_aware_update(observation, prediction)
        
        # Validate performance
        enhanced_stats = self.enhanced_kalman.get_performance_statistics()
        regime_stats = self.regime_kalman.get_regime_statistics()
        
        self.assertEqual(enhanced_stats['total_predictions'], 20)
        self.assertEqual(enhanced_stats['total_updates'], 20)
        self.assertEqual(regime_stats['total_predictions'], 20)
        
        # Check that both methods produce reasonable results
        self.assertGreater(enhanced_stats['average_confidence'], 0.0)
        self.assertLess(enhanced_stats['average_confidence'], 1.0)
        self.assertGreater(regime_stats['average_confidence'], 0.0)
        self.assertLess(regime_stats['average_confidence'], 1.0)


if __name__ == '__main__':
    unittest.main()
