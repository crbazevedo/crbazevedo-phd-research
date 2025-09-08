"""
Unit tests for EPIC 1: Market Regime Detection BNN

Tests the market regime detection and regime-switching Kalman filter
implementation based on EPIC 0 analysis results.
"""

import unittest
import numpy as np
import pandas as pd
import sys
import os
from typing import Dict
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from algorithms.regime_detection_bnn import (
    MarketRegimeDetector, MarketRegimeDetectionBNN, RegimeDetectionResult,
    create_regime_detector
)
from algorithms.regime_switching_kalman import (
    RegimeSwitchingKalmanFilter, AdaptiveRegimeSwitchingKalmanFilter,
    RegimeSwitchingResult, create_regime_switching_kalman
)


class TestMarketRegimeDetector(unittest.TestCase):
    """Test cases for MarketRegimeDetector class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = MarketRegimeDetector(input_dim=20, num_regimes=3)
        
        # Create mock market data
        self.mock_data = self._create_mock_market_data()
        
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
            'test_asset_2': combined_data * 1.1  # Slightly different scale
        }
    
    def _create_regime_data(self, dates: pd.DatetimeIndex, trend: float, 
                          volatility: float) -> pd.DataFrame:
        """Create data for a specific market regime."""
        np.random.seed(42)  # For reproducible tests
        
        # Generate price data with trend and volatility
        returns = np.random.normal(trend, volatility, len(dates))
        prices = 100 * np.exp(np.cumsum(returns))
        
        # Create OHLCV data
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
        """Test initialization of regime detector."""
        self.assertEqual(self.detector.input_dim, 20)
        self.assertEqual(self.detector.num_regimes, 3)
        self.assertEqual(len(self.detector.regime_labels), 3)
        self.assertFalse(self.detector.is_fitted)
    
    def test_create_market_features(self):
        """Test market feature creation."""
        features = self.detector._create_market_features(self.mock_data['test_asset'])
        
        self.assertEqual(len(features), 20)
        self.assertIsInstance(features, np.ndarray)
        self.assertFalse(np.any(np.isnan(features)))
    
    def test_label_regimes(self):
        """Test regime labeling."""
        regimes = self.detector._label_regimes(self.mock_data['test_asset'])
        
        self.assertEqual(len(regimes), len(self.mock_data['test_asset']))
        self.assertTrue(all(regime in self.detector.regime_labels for regime in regimes))
    
    def test_fit(self):
        """Test model fitting."""
        self.detector.fit(self.mock_data)
        
        self.assertTrue(self.detector.is_fitted)
        self.assertEqual(len(self.detector.models), 3)  # Ensemble of 3 models
    
    def test_detect_regime(self):
        """Test regime detection."""
        # Fit the model first
        self.detector.fit(self.mock_data)
        
        # Create test features
        test_features = np.random.randn(20)
        
        # Detect regime
        result = self.detector.detect_regime(test_features)
        
        self.assertIsInstance(result, RegimeDetectionResult)
        self.assertEqual(len(result.regime_probabilities), 3)
        self.assertEqual(len(result.regime_uncertainty), 3)
        self.assertIn(result.predicted_regime, self.detector.regime_labels)
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)
        
        # Check probabilities sum to 1
        self.assertAlmostEqual(np.sum(result.regime_probabilities), 1.0, places=5)
    
    def test_detect_regime_from_data(self):
        """Test regime detection from market data."""
        # Fit the model first
        self.detector.fit(self.mock_data)
        
        # Detect regime from data
        result = self.detector.detect_regime_from_data(self.mock_data['test_asset'])
        
        self.assertIsInstance(result, RegimeDetectionResult)
        self.assertIn(result.predicted_regime, self.detector.regime_labels)
    
    def test_detect_regime_not_fitted(self):
        """Test regime detection when model is not fitted."""
        test_features = np.random.randn(20)
        result = self.detector.detect_regime(test_features)
        
        # Should return default regime
        self.assertEqual(result.predicted_regime, 'sideways_market')
        self.assertEqual(result.confidence, 0.5)
    
    def test_get_regime_statistics(self):
        """Test regime statistics."""
        # Initially no statistics
        stats = self.detector.get_regime_statistics()
        self.assertIn('error', stats)
        
        # Fit and make some predictions
        self.detector.fit(self.mock_data)
        for _ in range(5):
            test_features = np.random.randn(20)
            self.detector.detect_regime(test_features)
        
        # Get statistics
        stats = self.detector.get_regime_statistics()
        
        self.assertIn('total_detections', stats)
        self.assertIn('regime_counts', stats)
        self.assertIn('average_confidence', stats)
        self.assertIn('regime_transitions', stats)
        self.assertIn('stability_ratio', stats)
        
        self.assertEqual(stats['total_detections'], 5)
    
    def test_reset_history(self):
        """Test history reset."""
        # Fit and make predictions
        self.detector.fit(self.mock_data)
        for _ in range(3):
            test_features = np.random.randn(20)
            self.detector.detect_regime(test_features)
        
        # Check history exists
        self.assertGreater(len(self.detector.regime_history), 0)
        
        # Reset history
        self.detector.reset_history()
        
        # Check history is cleared
        self.assertEqual(len(self.detector.regime_history), 0)
    
    def test_validate_regime_detection(self):
        """Test regime detection validation."""
        # Fit the model
        self.detector.fit(self.mock_data)
        
        # Create test data
        test_data = {'test_asset': self.mock_data['test_asset']}
        
        # Validate
        validation_results = self.detector.validate_regime_detection(test_data)
        
        self.assertIn('accuracy', validation_results)
        self.assertIn('classification_report', validation_results)
        self.assertIn('total_predictions', validation_results)
        
        self.assertGreaterEqual(validation_results['accuracy'], 0.0)
        self.assertLessEqual(validation_results['accuracy'], 1.0)


class TestMarketRegimeDetectionBNN(unittest.TestCase):
    """Test cases for MarketRegimeDetectionBNN class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.enhanced_detector = MarketRegimeDetectionBNN(input_dim=20, num_regimes=3)
        self.mock_data = self._create_mock_market_data()
    
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
        """Test initialization of enhanced regime detector."""
        self.assertEqual(self.enhanced_detector.input_dim, 20)
        self.assertEqual(self.enhanced_detector.num_regimes, 3)
        self.assertIsNotNone(self.enhanced_detector.base_detector)
    
    def test_fit(self):
        """Test enhanced model fitting."""
        self.enhanced_detector.fit(self.mock_data)
        
        self.assertTrue(self.enhanced_detector.base_detector.is_fitted)
        self.assertGreaterEqual(len(self.enhanced_detector.uncertainty_models), 0)
    
    def test_detect_regime_enhanced(self):
        """Test enhanced regime detection."""
        # Fit the model
        self.enhanced_detector.fit(self.mock_data)
        
        # Test features
        test_features = np.random.randn(20)
        
        # Detect regime
        result = self.enhanced_detector.detect_regime(test_features)
        
        self.assertIsInstance(result, RegimeDetectionResult)
        self.assertIn(result.predicted_regime, self.enhanced_detector.regime_labels)
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)
    
    def test_get_enhanced_statistics(self):
        """Test enhanced statistics."""
        # Initially no statistics
        stats = self.enhanced_detector.get_enhanced_statistics()
        self.assertIn('error', stats)
        
        # Fit and make predictions
        self.enhanced_detector.fit(self.mock_data)
        for _ in range(5):
            test_features = np.random.randn(20)
            self.enhanced_detector.detect_regime(test_features)
        
        # Get enhanced statistics
        stats = self.enhanced_detector.get_enhanced_statistics()
        
        self.assertIn('total_detections', stats)
        self.assertIn('average_uncertainty', stats)
        self.assertIn('uncertainty_std', stats)
        self.assertIn('confidence_std', stats)
        self.assertIn('uncertainty_trend', stats)
        self.assertIn('enhanced_detections', stats)


class TestRegimeSwitchingKalmanFilter(unittest.TestCase):
    """Test cases for RegimeSwitchingKalmanFilter class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock regime detector
        self.regime_detector = MarketRegimeDetectionBNN(input_dim=20, num_regimes=3)
        self.mock_data = self._create_mock_market_data()
        
        # Fit regime detector
        self.regime_detector.fit(self.mock_data)
        
        # Create regime-switching Kalman filter
        self.kalman_filter = RegimeSwitchingKalmanFilter(self.regime_detector)
    
    def _create_mock_market_data(self) -> Dict[str, pd.DataFrame]:
        """Create mock market data for testing."""
        dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
        
        bull_market_data = self._create_regime_data(dates[:365], trend=0.001, volatility=0.01)
        bear_market_data = self._create_regime_data(dates[365:730], trend=-0.001, volatility=0.015)
        sideways_market_data = self._create_regime_data(dates[730:], trend=0.0001, volatility=0.008)
        
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
        """Test initialization of regime-switching Kalman filter."""
        self.assertIsNotNone(self.kalman_filter.regime_detector)
        self.assertEqual(len(self.kalman_filter.kalman_filters), 3)
        self.assertEqual(self.kalman_filter.current_regime, 'sideways_market')
        self.assertEqual(self.kalman_filter.regime_confidence, 0.5)
    
    def test_predict_with_regime(self):
        """Test regime-aware prediction."""
        # Test state and features
        current_state = np.array([0.1, 0.05, 0.001, 0.0005])  # ROI, risk, ROI_vel, risk_vel
        market_features = np.random.randn(20)
        
        # Predict
        result = self.kalman_filter.predict_with_regime(current_state, market_features)
        
        self.assertIsInstance(result, RegimeSwitchingResult)
        self.assertEqual(len(result.prediction), 4)
        self.assertEqual(result.covariance.shape, (4, 4))
        self.assertIsInstance(result.regime_info, RegimeDetectionResult)
        self.assertTrue(result.regime_aware)
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)
    
    def test_predict_with_observation(self):
        """Test prediction with observation update."""
        current_state = np.array([0.1, 0.05, 0.001, 0.0005])
        market_features = np.random.randn(20)
        observation = np.array([0.12, 0.06])  # ROI, risk
        
        result = self.kalman_filter.predict_with_regime(current_state, market_features, observation)
        
        self.assertIsInstance(result, RegimeSwitchingResult)
        self.assertEqual(len(result.prediction), 4)
    
    def test_predict_multiple_horizons(self):
        """Test multiple horizon prediction."""
        current_state = np.array([0.1, 0.05, 0.001, 0.0005])
        market_features = np.random.randn(20)
        horizons = [1, 2, 3]
        
        results = self.kalman_filter.predict_multiple_horizons(current_state, market_features, horizons)
        
        self.assertEqual(len(results), 3)
        for horizon in horizons:
            self.assertIn(horizon, results)
            self.assertIsInstance(results[horizon], RegimeSwitchingResult)
    
    def test_regime_transition(self):
        """Test regime transition handling."""
        # Start with sideways market
        self.assertEqual(self.kalman_filter.current_regime, 'sideways_market')
        
        # Create features that should trigger regime change
        current_state = np.array([0.1, 0.05, 0.001, 0.0005])
        market_features = np.random.randn(20)
        
        # Make prediction
        result = self.kalman_filter.predict_with_regime(current_state, market_features)
        
        # Check that regime transition was handled
        self.assertIsInstance(result, RegimeSwitchingResult)
    
    def test_get_regime_statistics(self):
        """Test regime statistics."""
        # Initially no statistics
        stats = self.kalman_filter.get_regime_statistics()
        self.assertIn('error', stats)
        
        # Make some predictions
        current_state = np.array([0.1, 0.05, 0.001, 0.0005])
        market_features = np.random.randn(20)
        
        for _ in range(5):
            self.kalman_filter.predict_with_regime(current_state, market_features)
        
        # Get statistics
        stats = self.kalman_filter.get_regime_statistics()
        
        self.assertIn('total_predictions', stats)
        self.assertIn('regime_distribution', stats)
        self.assertIn('regime_transitions', stats)
        self.assertIn('average_regime_confidence', stats)
        self.assertIn('current_regime', stats)
        self.assertIn('regime_stability', stats)
        
        self.assertEqual(stats['total_predictions'], 5)
    
    def test_reset_history(self):
        """Test history reset."""
        # Make some predictions
        current_state = np.array([0.1, 0.05, 0.001, 0.0005])
        market_features = np.random.randn(20)
        
        for _ in range(3):
            self.kalman_filter.predict_with_regime(current_state, market_features)
        
        # Check history exists
        self.assertGreater(len(self.kalman_filter.prediction_history), 0)
        self.assertGreater(len(self.kalman_filter.regime_history), 0)
        
        # Reset history
        self.kalman_filter.reset_history()
        
        # Check history is cleared
        self.assertEqual(len(self.kalman_filter.prediction_history), 0)
        self.assertEqual(len(self.kalman_filter.regime_history), 0)
        self.assertEqual(self.kalman_filter.current_regime, 'sideways_market')
    
    def test_validate_regime_switching(self):
        """Test regime-switching validation."""
        # Create test data
        test_data = []
        for _ in range(10):
            state = np.random.randn(4)
            features = np.random.randn(20)
            observation = np.random.randn(2)
            test_data.append((state, features, observation))
        
        # Validate
        validation_results = self.kalman_filter.validate_regime_switching(test_data)
        
        self.assertIn('mse', validation_results)
        self.assertIn('mae', validation_results)
        self.assertIn('rmse', validation_results)
        self.assertIn('total_predictions', validation_results)
        
        self.assertGreaterEqual(validation_results['mse'], 0.0)
        self.assertGreaterEqual(validation_results['mae'], 0.0)
        self.assertEqual(validation_results['total_predictions'], 10)


class TestAdaptiveRegimeSwitchingKalmanFilter(unittest.TestCase):
    """Test cases for AdaptiveRegimeSwitchingKalmanFilter class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.regime_detector = MarketRegimeDetectionBNN(input_dim=20, num_regimes=3)
        self.mock_data = self._create_mock_market_data()
        
        self.regime_detector.fit(self.mock_data)
        self.adaptive_kalman = AdaptiveRegimeSwitchingKalmanFilter(self.regime_detector)
    
    def _create_mock_market_data(self) -> Dict[str, pd.DataFrame]:
        """Create mock market data for testing."""
        dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
        
        bull_market_data = self._create_regime_data(dates[:365], trend=0.001, volatility=0.01)
        bear_market_data = self._create_regime_data(dates[365:730], trend=-0.001, volatility=0.015)
        sideways_market_data = self._create_regime_data(dates[730:], trend=0.0001, volatility=0.008)
        
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
        """Test initialization of adaptive regime-switching Kalman filter."""
        self.assertIsNotNone(self.adaptive_kalman.regime_detector)
        self.assertEqual(self.adaptive_kalman.parameter_adaptation_rate, 0.01)
        self.assertEqual(len(self.adaptive_kalman.performance_history), 0)
    
    def test_predict_with_adaptation(self):
        """Test prediction with parameter adaptation."""
        current_state = np.array([0.1, 0.05, 0.001, 0.0005])
        market_features = np.random.randn(20)
        observation = np.array([0.12, 0.06])
        
        # Make prediction with observation
        result = self.adaptive_kalman.predict_with_regime(current_state, market_features, observation)
        
        self.assertIsInstance(result, RegimeSwitchingResult)
        self.assertEqual(len(self.adaptive_kalman.performance_history), 1)
    
    def test_parameter_adaptation(self):
        """Test parameter adaptation over multiple predictions."""
        current_state = np.array([0.1, 0.05, 0.001, 0.0005])
        market_features = np.random.randn(20)
        
        # Make multiple predictions with observations
        for i in range(15):
            observation = np.array([0.12 + i*0.001, 0.06 + i*0.0005])
            result = self.adaptive_kalman.predict_with_regime(current_state, market_features, observation)
        
        # Check that performance history is populated
        self.assertEqual(len(self.adaptive_kalman.performance_history), 15)
        
        # Check that regime performance is tracked
        for regime in self.adaptive_kalman.regime_performance:
            self.assertGreaterEqual(len(self.adaptive_kalman.regime_performance[regime]), 0)
    
    def test_get_adaptive_statistics(self):
        """Test adaptive statistics."""
        # Make some predictions
        current_state = np.array([0.1, 0.05, 0.001, 0.0005])
        market_features = np.random.randn(20)
        
        for i in range(5):
            observation = np.array([0.12 + i*0.001, 0.06 + i*0.0005])
            self.adaptive_kalman.predict_with_regime(current_state, market_features, observation)
        
        # Get adaptive statistics
        stats = self.adaptive_kalman.get_adaptive_statistics()
        
        self.assertIn('parameter_adaptation_rate', stats)
        self.assertIn('performance_history_length', stats)
        self.assertIn('average_performance', stats)
        self.assertIn('regime_performance', stats)
        
        self.assertEqual(stats['performance_history_length'], 5)


class TestConvenienceFunctions(unittest.TestCase):
    """Test cases for convenience functions."""
    
    def test_create_regime_detector(self):
        """Test regime detector creation."""
        # Test basic detector
        detector = create_regime_detector(input_dim=20, num_regimes=3, enhanced=False)
        self.assertIsInstance(detector, MarketRegimeDetector)
        
        # Test enhanced detector
        enhanced_detector = create_regime_detector(input_dim=20, num_regimes=3, enhanced=True)
        self.assertIsInstance(enhanced_detector, MarketRegimeDetectionBNN)
    
    def test_create_regime_switching_kalman(self):
        """Test regime-switching Kalman filter creation."""
        # Create regime detector
        regime_detector = MarketRegimeDetectionBNN(input_dim=20, num_regimes=3)
        
        # Test basic filter
        kalman_filter = create_regime_switching_kalman(regime_detector, adaptive=False)
        self.assertIsInstance(kalman_filter, RegimeSwitchingKalmanFilter)
        
        # Test adaptive filter
        adaptive_kalman = create_regime_switching_kalman(regime_detector, adaptive=True)
        self.assertIsInstance(adaptive_kalman, AdaptiveRegimeSwitchingKalmanFilter)


class TestEPIC1Integration(unittest.TestCase):
    """Integration tests for EPIC 1 components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_data = self._create_mock_market_data()
        
        # Create regime detector
        self.regime_detector = MarketRegimeDetectionBNN(input_dim=20, num_regimes=3)
        self.regime_detector.fit(self.mock_data)
        
        # Create regime-switching Kalman filter
        self.kalman_filter = RegimeSwitchingKalmanFilter(self.regime_detector)
    
    def _create_mock_market_data(self) -> Dict[str, pd.DataFrame]:
        """Create mock market data for testing."""
        dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
        
        bull_market_data = self._create_regime_data(dates[:365], trend=0.001, volatility=0.01)
        bear_market_data = self._create_regime_data(dates[365:730], trend=-0.001, volatility=0.015)
        sideways_market_data = self._create_regime_data(dates[730:], trend=0.0001, volatility=0.008)
        
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
    
    def test_full_epic1_workflow(self):
        """Test complete EPIC 1 workflow."""
        # Step 1: Regime detection
        market_features = np.random.randn(20)
        regime_result = self.regime_detector.detect_regime(market_features)
        
        self.assertIsInstance(regime_result, RegimeDetectionResult)
        self.assertIn(regime_result.predicted_regime, ['bull_market', 'bear_market', 'sideways_market'])
        
        # Step 2: Regime-switching prediction
        current_state = np.array([0.1, 0.05, 0.001, 0.0005])
        observation = np.array([0.12, 0.06])
        
        prediction_result = self.kalman_filter.predict_with_regime(
            current_state, market_features, observation
        )
        
        self.assertIsInstance(prediction_result, RegimeSwitchingResult)
        self.assertTrue(prediction_result.regime_aware)
        
        # Step 3: Multiple horizon prediction
        horizons = [1, 2, 3]
        multi_horizon_results = self.kalman_filter.predict_multiple_horizons(
            current_state, market_features, horizons
        )
        
        self.assertEqual(len(multi_horizon_results), 3)
        for horizon in horizons:
            self.assertIn(horizon, multi_horizon_results)
        
        # Step 4: Statistics
        regime_stats = self.regime_detector.get_enhanced_statistics()
        kalman_stats = self.kalman_filter.get_regime_statistics()
        
        self.assertIn('total_detections', regime_stats)
        self.assertIn('total_predictions', kalman_stats)
    
    def test_epic1_performance_validation(self):
        """Test EPIC 1 performance validation."""
        # Create test data
        test_data = []
        for _ in range(20):
            state = np.random.randn(4)
            features = np.random.randn(20)
            observation = np.random.randn(2)
            test_data.append((state, features, observation))
        
        # Validate regime detection
        regime_validation = self.regime_detector.base_detector.validate_regime_detection(self.mock_data)
        self.assertIn('accuracy', regime_validation)
        
        # Validate regime-switching Kalman filter
        kalman_validation = self.kalman_filter.validate_regime_switching(test_data)
        self.assertIn('mse', kalman_validation)
        self.assertIn('mae', kalman_validation)
        
        # Check that performance is reasonable
        self.assertGreaterEqual(regime_validation['accuracy'], 0.0)
        self.assertLessEqual(regime_validation['accuracy'], 1.0)
        self.assertGreaterEqual(kalman_validation['mse'], 0.0)


if __name__ == '__main__':
    unittest.main()
