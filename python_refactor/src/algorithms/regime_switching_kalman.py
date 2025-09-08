"""
Regime-Switching Kalman Filter

This module implements a Kalman filter with regime-switching capabilities,
integrating with the market regime detection BNN to improve prediction
accuracy and uncertainty quantification.

Based on EPIC 0 analysis: regime detection is more feasible than direct
return prediction, so this approach leverages regime information to enhance
the existing Kalman filter implementation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass
from datetime import datetime

from algorithms.kalman_filter import KalmanParams, kalman_filter, kalman_prediction, kalman_update
from algorithms.regime_detection_bnn import MarketRegimeDetectionBNN, RegimeDetectionResult

logger = logging.getLogger(__name__)


@dataclass
class RegimeSwitchingResult:
    """Data class for regime-switching prediction results."""
    
    prediction: np.ndarray
    covariance: np.ndarray
    regime_info: RegimeDetectionResult
    regime_aware: bool
    confidence: float
    timestamp: float


class RegimeSwitchingKalmanFilter:
    """
    Kalman filter with regime-switching capabilities.
    
    This class integrates market regime detection with Kalman filtering
    to provide regime-aware predictions with improved accuracy.
    """
    
    def __init__(self, regime_detector: MarketRegimeDetectionBNN):
        """
        Initialize regime-switching Kalman filter.
        
        Args:
            regime_detector: Market regime detection BNN
        """
        self.regime_detector = regime_detector
        
        # Initialize Kalman filters for different regimes
        self.kalman_filters = {
            'bull_market': self._create_kalman_params(process_noise=0.005, measurement_noise=0.002),
            'bear_market': self._create_kalman_params(process_noise=0.02, measurement_noise=0.01),
            'sideways_market': self._create_kalman_params(process_noise=0.01, measurement_noise=0.005)
        }
        
        # Current regime state
        self.current_regime = 'sideways_market'
        self.regime_confidence = 0.5
        self.regime_transition_probability = 0.1
        
        # Historical data
        self.prediction_history = []
        self.regime_history = []
        
        logger.info("Initialized RegimeSwitchingKalmanFilter")
    
    def _create_kalman_params(self, process_noise: float = 0.01, 
                            measurement_noise: float = 0.005) -> KalmanParams:
        """
        Create Kalman filter parameters for a specific regime.
        
        Args:
            process_noise: Process noise variance
            measurement_noise: Measurement noise variance
            
        Returns:
            KalmanParams instance
        """
        # State: [ROI, risk, ROI_velocity, risk_velocity]
        x = np.array([0.0, 0.0, 0.0, 0.0])  # Initial state
        
        # State transition matrix (constant velocity model)
        F = np.array([
            [1.0, 0.0, 1.0, 0.0],   # ROI_t = ROI_{t-1} + ROI_velocity_{t-1}
            [0.0, 1.0, 0.0, 1.0],   # risk_t = risk_{t-1} + risk_velocity_{t-1}
            [0.0, 0.0, 1.0, 0.0],   # ROI_velocity_t = ROI_velocity_{t-1}
            [0.0, 0.0, 0.0, 1.0]    # risk_velocity_t = risk_velocity_{t-1}
        ])
        
        # Measurement matrix (observe ROI and risk)
        H = np.array([
            [1.0, 0.0, 0.0, 0.0],   # ROI observation
            [0.0, 1.0, 0.0, 0.0]    # risk observation
        ])
        
        # Process noise covariance
        Q = np.eye(4) * process_noise
        
        # Measurement noise covariance
        R = np.eye(2) * measurement_noise
        
        # Initial state covariance
        P = np.eye(4) * 0.1
        
        return KalmanParams(x=x, F=F, H=H, R=R, P=P)
    
    def predict_with_regime(self, current_state: np.ndarray, 
                          market_features: np.ndarray,
                          observation: Optional[np.ndarray] = None) -> RegimeSwitchingResult:
        """
        Predict with regime-aware Kalman filter.
        
        Args:
            current_state: Current state vector [ROI, risk, ROI_velocity, risk_velocity]
            market_features: Market features for regime detection
            observation: Optional observation for update step
            
        Returns:
            RegimeSwitchingResult with prediction and regime information
        """
        # Detect current regime
        regime_info = self.regime_detector.detect_regime(market_features)
        detected_regime = regime_info.predicted_regime
        
        # Handle regime transitions
        if detected_regime != self.current_regime:
            self._handle_regime_transition(detected_regime, regime_info.confidence)
        
        # Get appropriate Kalman filter
        kalman_params = self.kalman_filters[self.current_regime]
        
        # Update Kalman filter state
        kalman_params.x = current_state.copy()
        
        # Perform prediction
        kalman_prediction(kalman_params)
        
        # Perform update if observation is provided
        if observation is not None:
            kalman_update(kalman_params, observation)
        
        # Calculate prediction confidence
        prediction_confidence = self._calculate_prediction_confidence(
            kalman_params, regime_info
        )
        
        # Create result
        result = RegimeSwitchingResult(
            prediction=kalman_params.x.copy(),
            covariance=kalman_params.P.copy(),
            regime_info=regime_info,
            regime_aware=True,
            confidence=prediction_confidence,
            timestamp=datetime.now().timestamp()
        )
        
        # Store in history
        self.prediction_history.append(result)
        self.regime_history.append(regime_info)
        
        logger.debug(f"Regime-switching prediction: {self.current_regime} "
                    f"(confidence: {prediction_confidence:.3f})")
        
        return result
    
    def _handle_regime_transition(self, new_regime: str, confidence: float) -> None:
        """
        Handle transition to new regime.
        
        Args:
            new_regime: New regime name
            confidence: Confidence in regime detection
        """
        # Only transition if confidence is high enough
        if confidence > 0.6:
            old_regime = self.current_regime
            self.current_regime = new_regime
            self.regime_confidence = confidence
            
            logger.info(f"Regime transition: {old_regime} -> {new_regime} "
                       f"(confidence: {confidence:.3f})")
            
            # Adjust transition probability based on confidence
            self.regime_transition_probability = 1.0 - confidence
        else:
            # Low confidence - maintain current regime but adjust parameters
            self.regime_confidence = confidence
            self._adjust_kalman_parameters(confidence)
    
    def _adjust_kalman_parameters(self, confidence: float) -> None:
        """
        Adjust Kalman filter parameters based on regime confidence.
        
        Args:
            confidence: Confidence in current regime
        """
        # Adjust process noise based on confidence
        # Lower confidence -> higher process noise (more uncertainty)
        noise_multiplier = 1.0 + (1.0 - confidence) * 2.0
        
        for regime, kalman_params in self.kalman_filters.items():
            # Adjust process noise
            kalman_params.Q *= noise_multiplier
            
            # Adjust measurement noise
            kalman_params.R *= noise_multiplier
    
    def _calculate_prediction_confidence(self, kalman_params: KalmanParams, 
                                       regime_info: RegimeDetectionResult) -> float:
        """
        Calculate overall prediction confidence.
        
        Args:
            kalman_params: Kalman filter parameters
            regime_info: Regime detection result
            
        Returns:
            Prediction confidence score
        """
        # Kalman filter confidence (based on covariance)
        kalman_confidence = 1.0 / (1.0 + np.trace(kalman_params.P))
        
        # Regime confidence
        regime_confidence = regime_info.confidence
        
        # Combined confidence
        combined_confidence = 0.6 * kalman_confidence + 0.4 * regime_confidence
        
        return max(0.0, min(1.0, combined_confidence))
    
    def predict_multiple_horizons(self, current_state: np.ndarray,
                                market_features: np.ndarray,
                                horizons: List[int]) -> Dict[int, RegimeSwitchingResult]:
        """
        Predict for multiple horizons with regime awareness.
        
        Args:
            current_state: Current state vector
            market_features: Market features for regime detection
            horizons: List of prediction horizons
            
        Returns:
            Dictionary mapping horizon to prediction result
        """
        results = {}
        current_state_copy = current_state.copy()
        
        for horizon in horizons:
            # Get regime-aware prediction
            result = self.predict_with_regime(current_state_copy, market_features)
            results[horizon] = result
            
            # Update state for next horizon
            current_state_copy = result.prediction.copy()
        
        return results
    
    def get_regime_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about regime switching behavior.
        
        Returns:
            Dictionary with regime statistics
        """
        if not self.regime_history:
            return {'error': 'No regime history available'}
        
        # Regime distribution
        regime_counts = {}
        for regime in ['bull_market', 'bear_market', 'sideways_market']:
            regime_counts[regime] = sum(1 for r in self.regime_history 
                                      if r.predicted_regime == regime)
        
        # Regime transitions
        transitions = []
        for i in range(1, len(self.regime_history)):
            prev_regime = self.regime_history[i-1].predicted_regime
            curr_regime = self.regime_history[i].predicted_regime
            if prev_regime != curr_regime:
                transitions.append(f"{prev_regime} -> {curr_regime}")
        
        # Confidence statistics
        confidences = [r.confidence for r in self.regime_history]
        
        # Prediction confidence statistics
        prediction_confidences = [r.confidence for r in self.prediction_history]
        
        return {
            'total_predictions': len(self.prediction_history),
            'regime_distribution': regime_counts,
            'regime_transitions': transitions,
            'transition_count': len(transitions),
            'average_regime_confidence': np.mean(confidences),
            'average_prediction_confidence': np.mean(prediction_confidences),
            'current_regime': self.current_regime,
            'regime_stability': 1.0 - (len(transitions) / len(self.regime_history)) if self.regime_history else 0.0
        }
    
    def reset_history(self):
        """Reset prediction and regime history."""
        self.prediction_history.clear()
        self.regime_history.clear()
        self.current_regime = 'sideways_market'
        self.regime_confidence = 0.5
        logger.info("Reset regime-switching Kalman filter history")
    
    def validate_regime_switching(self, test_data: List[Tuple[np.ndarray, np.ndarray, np.ndarray]]) -> Dict[str, Any]:
        """
        Validate regime-switching performance on test data.
        
        Args:
            test_data: List of (state, features, observation) tuples
            
        Returns:
            Validation results
        """
        logger.info("Validating regime-switching Kalman filter...")
        
        predictions = []
        observations = []
        regime_predictions = []
        
        for state, features, observation in test_data:
            # Get prediction
            result = self.predict_with_regime(state, features, observation)
            
            predictions.append(result.prediction[:2])  # ROI and risk only
            observations.append(observation)
            regime_predictions.append(result.regime_info.predicted_regime)
        
        if len(predictions) == 0:
            return {'error': 'No valid predictions'}
        
        # Calculate prediction accuracy
        predictions = np.array(predictions)
        observations = np.array(observations)
        
        mse = np.mean((predictions - observations)**2)
        mae = np.mean(np.abs(predictions - observations))
        
        # Calculate regime prediction accuracy (if ground truth available)
        regime_accuracy = None
        if len(regime_predictions) > 0:
            # For now, just count regime distribution
            regime_counts = {}
            for regime in regime_predictions:
                regime_counts[regime] = regime_counts.get(regime, 0) + 1
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': np.sqrt(mse),
            'total_predictions': len(predictions),
            'regime_distribution': regime_counts if regime_accuracy is None else None,
            'regime_accuracy': regime_accuracy
        }


class AdaptiveRegimeSwitchingKalmanFilter(RegimeSwitchingKalmanFilter):
    """
    Enhanced regime-switching Kalman filter with adaptive parameters.
    
    This version automatically adjusts Kalman filter parameters based on
    regime detection performance and prediction accuracy.
    """
    
    def __init__(self, regime_detector: MarketRegimeDetectionBNN):
        """
        Initialize adaptive regime-switching Kalman filter.
        
        Args:
            regime_detector: Market regime detection BNN
        """
        super().__init__(regime_detector)
        
        # Adaptive parameters
        self.parameter_adaptation_rate = 0.01
        self.performance_history = []
        self.regime_performance = {
            'bull_market': [],
            'bear_market': [],
            'sideways_market': []
        }
        
        logger.info("Initialized AdaptiveRegimeSwitchingKalmanFilter")
    
    def predict_with_regime(self, current_state: np.ndarray, 
                          market_features: np.ndarray,
                          observation: Optional[np.ndarray] = None) -> RegimeSwitchingResult:
        """
        Predict with adaptive regime-switching Kalman filter.
        
        Args:
            current_state: Current state vector
            market_features: Market features for regime detection
            observation: Optional observation for update step
            
        Returns:
            RegimeSwitchingResult with adaptive prediction
        """
        # Get base prediction
        result = super().predict_with_regime(current_state, market_features, observation)
        
        # Adapt parameters based on performance
        if observation is not None:
            self._adapt_parameters(result, observation)
        
        return result
    
    def _adapt_parameters(self, prediction_result: RegimeSwitchingResult, 
                         observation: np.ndarray) -> None:
        """
        Adapt Kalman filter parameters based on prediction performance.
        
        Args:
            prediction_result: Prediction result
            observation: Actual observation
        """
        # Calculate prediction error
        prediction_error = np.mean((prediction_result.prediction[:2] - observation)**2)
        
        # Store performance
        self.performance_history.append(prediction_error)
        regime = prediction_result.regime_info.predicted_regime
        self.regime_performance[regime].append(prediction_error)
        
        # Adapt parameters if we have enough history
        if len(self.performance_history) > 10:
            self._update_kalman_parameters(regime, prediction_error)
    
    def _update_kalman_parameters(self, regime: str, prediction_error: float) -> None:
        """
        Update Kalman filter parameters for a specific regime.
        
        Args:
            regime: Regime name
            prediction_error: Recent prediction error
        """
        kalman_params = self.kalman_filters[regime]
        
        # Calculate average performance for this regime
        regime_errors = self.regime_performance[regime]
        if len(regime_errors) > 5:
            avg_error = np.mean(regime_errors[-10:])  # Last 10 predictions
            
            # Adjust process noise based on performance
            if avg_error > np.mean(self.performance_history[-20:]):  # Worse than average
                # Increase process noise (more uncertainty)
                kalman_params.Q *= (1.0 + self.parameter_adaptation_rate)
                kalman_params.R *= (1.0 + self.parameter_adaptation_rate)
            else:
                # Decrease process noise (more confidence)
                kalman_params.Q *= (1.0 - self.parameter_adaptation_rate)
                kalman_params.R *= (1.0 - self.parameter_adaptation_rate)
            
            # Ensure parameters stay within reasonable bounds
            kalman_params.Q = np.clip(kalman_params.Q, 0.001, 0.1)
            kalman_params.R = np.clip(kalman_params.R, 0.001, 0.05)
    
    def get_adaptive_statistics(self) -> Dict[str, Any]:
        """Get statistics including adaptive parameter information."""
        base_stats = self.get_regime_statistics()
        
        # Add adaptive statistics
        base_stats.update({
            'parameter_adaptation_rate': self.parameter_adaptation_rate,
            'performance_history_length': len(self.performance_history),
            'average_performance': np.mean(self.performance_history) if self.performance_history else 0.0,
            'regime_performance': {
                regime: np.mean(errors) if errors else 0.0 
                for regime, errors in self.regime_performance.items()
            }
        })
        
        return base_stats


def create_regime_switching_kalman(regime_detector: MarketRegimeDetectionBNN,
                                 adaptive: bool = False) -> RegimeSwitchingKalmanFilter:
    """
    Convenience function to create regime-switching Kalman filter.
    
    Args:
        regime_detector: Market regime detection BNN
        adaptive: Whether to use adaptive version
        
    Returns:
        Regime-switching Kalman filter instance
    """
    if adaptive:
        return AdaptiveRegimeSwitchingKalmanFilter(regime_detector)
    else:
        return RegimeSwitchingKalmanFilter(regime_detector)


if __name__ == '__main__':
    # Example usage
    print("Regime-Switching Kalman Filter Module")
    print("This module provides regime-aware Kalman filtering.")
    print("Use RegimeSwitchingKalmanFilter class for regime-aware predictions.")
