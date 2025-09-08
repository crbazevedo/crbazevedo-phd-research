"""
Market Regime Detection using Bayesian Neural Network

This module implements a Bayesian Neural Network for detecting market regimes
(bull, bear, sideways) based on the EPIC 0 analysis results. This is the
highest priority BNN implementation due to its high feasibility and clear value.

Based on EPIC 0 findings:
- Regime detection is more predictable than direct return prediction
- BNN can provide uncertainty quantification for regime classification
- Integration with existing Kalman filter can improve performance
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


@dataclass
class RegimeDetectionResult:
    """Data class for regime detection results."""
    
    regime_probabilities: np.ndarray
    regime_uncertainty: np.ndarray
    predicted_regime: str
    confidence: float
    timestamp: float


class MarketRegimeDetector:
    """
    Market regime detector using simple ensemble approach.
    
    This is a simplified implementation that can be enhanced with true BNN
    in the future. For now, it uses ensemble methods to detect market regimes
    with uncertainty quantification.
    """
    
    def __init__(self, input_dim: int = 20, num_regimes: int = 3):
        """
        Initialize market regime detector.
        
        Args:
            input_dim: Number of input features
            num_regimes: Number of market regimes to detect
        """
        self.input_dim = input_dim
        self.num_regimes = num_regimes
        self.regime_labels = ['bull_market', 'bear_market', 'sideways_market']
        
        # Feature scaler
        self.scaler = StandardScaler()
        
        # Ensemble models for uncertainty quantification
        self.models = []
        self.is_fitted = False
        
        # Historical regime data
        self.regime_history = []
        self.regime_transitions = []
        
        logger.info(f"Initialized MarketRegimeDetector with {num_regimes} regimes")
    
    def _create_market_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        Create market features for regime detection.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Array of market features
        """
        features = []
        
        # Price-based features
        if 'Adj Close' in data.columns:
            prices = data['Adj Close'].dropna()
            
            # Returns
            returns = prices.pct_change().dropna()
            
            # Volatility
            volatility = returns.rolling(window=20).std()
            
            # Trend indicators
            sma_20 = prices.rolling(window=20).mean()
            sma_50 = prices.rolling(window=50).mean()
            trend_20 = (prices / sma_20 - 1).fillna(0)
            trend_50 = (prices / sma_50 - 1).fillna(0)
            
            # Momentum indicators
            momentum_5 = returns.rolling(window=5).sum()
            momentum_10 = returns.rolling(window=10).sum()
            momentum_20 = returns.rolling(window=20).sum()
            
            # Volatility features
            vol_5 = returns.rolling(window=5).std()
            vol_10 = returns.rolling(window=10).std()
            vol_20 = returns.rolling(window=20).std()
            
            # Volume features (if available)
            if 'Volume' in data.columns:
                volume = data['Volume'].dropna()
                volume_ma = volume.rolling(window=20).mean()
                volume_ratio = (volume / volume_ma).fillna(1)
            else:
                volume_ratio = pd.Series(1.0, index=prices.index)
            
            # Combine features
            feature_data = pd.DataFrame({
                'returns': returns,
                'volatility': volatility,
                'trend_20': trend_20,
                'trend_50': trend_50,
                'momentum_5': momentum_5,
                'momentum_10': momentum_10,
                'momentum_20': momentum_20,
                'vol_5': vol_5,
                'vol_10': vol_10,
                'vol_20': vol_20,
                'volume_ratio': volume_ratio
            }).dropna()
            
            # Take the most recent values
            if len(feature_data) > 0:
                recent_features = feature_data.iloc[-1].values
                features.extend(recent_features)
            else:
                features.extend([0] * 11)
        else:
            features.extend([0] * 11)
        
        # Ensure we have the right number of features
        while len(features) < self.input_dim:
            features.append(0.0)
        
        return np.array(features[:self.input_dim])
    
    def _label_regimes(self, data: pd.DataFrame) -> np.ndarray:
        """
        Label market regimes based on price movements.
        
        Args:
            data: DataFrame with price data
            
        Returns:
            Array of regime labels
        """
        if 'Adj Close' not in data.columns:
            return np.array(['sideways_market'] * len(data))
        
        prices = data['Adj Close'].dropna()
        if len(prices) < 50:
            return np.array(['sideways_market'] * len(prices))
        
        # Calculate rolling returns
        returns = prices.pct_change().dropna()
        rolling_returns = returns.rolling(window=20).mean()
        rolling_volatility = returns.rolling(window=20).std()
        
        # Define regime thresholds
        high_return_threshold = 0.001  # 0.1% daily return
        low_return_threshold = -0.001  # -0.1% daily return
        high_volatility_threshold = 0.02  # 2% daily volatility
        
        regimes = []
        for i in range(len(rolling_returns)):
            if pd.isna(rolling_returns.iloc[i]):
                regimes.append('sideways_market')
            elif rolling_returns.iloc[i] > high_return_threshold and rolling_volatility.iloc[i] < high_volatility_threshold:
                regimes.append('bull_market')
            elif rolling_returns.iloc[i] < low_return_threshold and rolling_volatility.iloc[i] < high_volatility_threshold:
                regimes.append('bear_market')
            else:
                regimes.append('sideways_market')
        
        return np.array(regimes)
    
    def fit(self, data: Dict[str, pd.DataFrame]) -> None:
        """
        Fit the regime detection model.
        
        Args:
            data: Dictionary of asset data (name -> DataFrame)
        """
        logger.info("Fitting market regime detection model...")
        
        all_features = []
        all_regimes = []
        
        # Process each asset
        for asset_name, df in data.items():
            try:
                # Label regimes first
                regimes = self._label_regimes(df)
                
                # Create features for each time point
                for i in range(50, len(df)):  # Start from 50 to have enough history
                    # Use data up to current point for features
                    historical_data = df.iloc[:i+1]
                    features = self._create_market_features(historical_data)
                    all_features.append(features)
                    
                    # Use corresponding regime (adjust index for regimes)
                    regime_idx = i
                    if regime_idx < len(regimes):
                        all_regimes.append(regimes[regime_idx])
                    else:
                        all_regimes.append('sideways_market')  # Default regime
                
            except Exception as e:
                logger.warning(f"Failed to process {asset_name}: {e}")
                continue
        
        if len(all_features) == 0:
            logger.error("No valid data for training")
            return
        
        # Convert to arrays
        X = np.array(all_features)
        y = np.array(all_regimes)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Create ensemble of simple models for uncertainty quantification
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        
        # Train ensemble models
        self.models = [
            RandomForestClassifier(n_estimators=100, random_state=42),
            LogisticRegression(random_state=42, max_iter=1000),
            SVC(probability=True, random_state=42)
        ]
        
        for model in self.models:
            model.fit(X_scaled, y)
        
        self.is_fitted = True
        logger.info(f"Fitted {len(self.models)} models for regime detection with {len(X)} samples")
    
    def detect_regime(self, market_features: np.ndarray) -> RegimeDetectionResult:
        """
        Detect current market regime with uncertainty.
        
        Args:
            market_features: Market features array
            
        Returns:
            RegimeDetectionResult with regime probabilities and uncertainty
        """
        if not self.is_fitted:
            # Return default regime if not fitted
            return RegimeDetectionResult(
                regime_probabilities=np.array([0.33, 0.33, 0.34]),
                regime_uncertainty=np.array([0.1, 0.1, 0.1]),
                predicted_regime='sideways_market',
                confidence=0.5,
                timestamp=datetime.now().timestamp()
            )
        
        # Ensure features have correct shape
        if len(market_features) != self.input_dim:
            market_features = np.pad(market_features, (0, max(0, self.input_dim - len(market_features))))
            market_features = market_features[:self.input_dim]
        
        # Scale features
        features_scaled = self.scaler.transform(market_features.reshape(1, -1))
        
        # Get predictions from ensemble
        regime_probabilities = []
        for model in self.models:
            try:
                probs = model.predict_proba(features_scaled)[0]
                regime_probabilities.append(probs)
            except Exception as e:
                logger.warning(f"Model prediction failed: {e}")
                # Use uniform probabilities as fallback
                regime_probabilities.append(np.array([0.33, 0.33, 0.34]))
        
        # Calculate ensemble probabilities and uncertainty
        regime_probabilities = np.array(regime_probabilities)
        mean_probabilities = np.mean(regime_probabilities, axis=0)
        uncertainty = np.std(regime_probabilities, axis=0)
        
        # Ensure probabilities sum to 1
        mean_probabilities = mean_probabilities / np.sum(mean_probabilities)
        
        # Get predicted regime
        predicted_regime_idx = np.argmax(mean_probabilities)
        predicted_regime = self.regime_labels[predicted_regime_idx]
        
        # Calculate confidence
        confidence = 1.0 - np.mean(uncertainty)
        confidence = max(0.0, min(1.0, confidence))
        
        # Create result
        result = RegimeDetectionResult(
            regime_probabilities=mean_probabilities,
            regime_uncertainty=uncertainty,
            predicted_regime=predicted_regime,
            confidence=confidence,
            timestamp=datetime.now().timestamp()
        )
        
        # Store in history
        self.regime_history.append(result)
        
        logger.debug(f"Detected regime: {predicted_regime} (confidence: {confidence:.3f})")
        
        return result
    
    def detect_regime_from_data(self, data: pd.DataFrame) -> RegimeDetectionResult:
        """
        Detect regime directly from market data.
        
        Args:
            data: DataFrame with market data
            
        Returns:
            RegimeDetectionResult
        """
        features = self._create_market_features(data)
        return self.detect_regime(features)
    
    def get_regime_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about regime detection.
        
        Returns:
            Dictionary with regime statistics
        """
        if not self.regime_history:
            return {'error': 'No regime history available'}
        
        # Extract regime counts
        regime_counts = {}
        for regime in self.regime_labels:
            regime_counts[regime] = sum(1 for r in self.regime_history 
                                      if r.predicted_regime == regime)
        
        # Calculate average confidence
        avg_confidence = np.mean([r.confidence for r in self.regime_history])
        
        # Calculate regime transition statistics
        transitions = []
        for i in range(1, len(self.regime_history)):
            prev_regime = self.regime_history[i-1].predicted_regime
            curr_regime = self.regime_history[i].predicted_regime
            if prev_regime != curr_regime:
                transitions.append(f"{prev_regime} -> {curr_regime}")
        
        return {
            'total_detections': len(self.regime_history),
            'regime_counts': regime_counts,
            'average_confidence': avg_confidence,
            'regime_transitions': transitions,
            'transition_count': len(transitions),
            'stability_ratio': 1.0 - (len(transitions) / len(self.regime_history)) if self.regime_history else 0.0
        }
    
    def reset_history(self):
        """Reset regime detection history."""
        self.regime_history.clear()
        self.regime_transitions.clear()
        logger.info("Reset regime detection history")
    
    def validate_regime_detection(self, test_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Validate regime detection on test data.
        
        Args:
            test_data: Dictionary of test asset data
            
        Returns:
            Validation results
        """
        if not self.is_fitted:
            return {'error': 'Model not fitted'}
        
        logger.info("Validating regime detection...")
        
        all_predictions = []
        all_true_regimes = []
        
        for asset_name, df in test_data.items():
            try:
                # Get true regimes
                true_regimes = self._label_regimes(df)
                
                # Get predictions
                features = self._create_market_features(df)
                prediction = self.detect_regime(features)
                
                all_predictions.append(prediction.predicted_regime)
                all_true_regimes.extend(true_regimes)
                
            except Exception as e:
                logger.warning(f"Validation failed for {asset_name}: {e}")
                continue
        
        if len(all_predictions) == 0:
            return {'error': 'No valid predictions'}
        
        # Calculate accuracy
        accuracy = accuracy_score(all_true_regimes, all_predictions)
        
        # Generate classification report
        try:
            report = classification_report(all_true_regimes, all_predictions, 
                                        target_names=self.regime_labels, output_dict=True)
        except:
            report = {'error': 'Could not generate classification report'}
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'total_predictions': len(all_predictions),
            'total_true_regimes': len(all_true_regimes)
        }


class MarketRegimeDetectionBNN:
    """
    Enhanced market regime detection with BNN-like uncertainty quantification.
    
    This class provides a more sophisticated approach to regime detection
    with better uncertainty quantification, though it's not a true BNN yet.
    """
    
    def __init__(self, input_dim: int = 20, num_regimes: int = 3):
        """
        Initialize enhanced regime detection BNN.
        
        Args:
            input_dim: Number of input features
            num_regimes: Number of market regimes
        """
        self.input_dim = input_dim
        self.num_regimes = num_regimes
        self.regime_labels = ['bull_market', 'bear_market', 'sideways_market']
        
        # Base detector
        self.base_detector = MarketRegimeDetector(input_dim, num_regimes)
        
        # Enhanced uncertainty quantification
        self.uncertainty_models = []
        self.uncertainty_scaler = StandardScaler()
        
        # Historical uncertainty data
        self.uncertainty_history = []
        
        logger.info(f"Initialized MarketRegimeDetectionBNN with {num_regimes} regimes")
    
    def fit(self, data: Dict[str, pd.DataFrame]) -> None:
        """
        Fit the enhanced regime detection model.
        
        Args:
            data: Dictionary of asset data
        """
        logger.info("Fitting enhanced regime detection BNN...")
        
        # Fit base detector
        self.base_detector.fit(data)
        
        # Fit uncertainty quantification models
        self._fit_uncertainty_models(data)
        
        logger.info("Enhanced regime detection BNN fitted successfully")
    
    def _fit_uncertainty_models(self, data: Dict[str, pd.DataFrame]) -> None:
        """Fit models for uncertainty quantification."""
        from sklearn.ensemble import RandomForestRegressor
        
        uncertainty_features = []
        uncertainty_targets = []
        
        for asset_name, df in data.items():
            try:
                # Create features
                features = self.base_detector._create_market_features(df)
                
                # Get regime predictions
                regime_result = self.base_detector.detect_regime(features)
                
                # Use prediction uncertainty as target
                uncertainty_target = np.mean(regime_result.regime_uncertainty)
                
                uncertainty_features.append(features)
                uncertainty_targets.append(uncertainty_target)
                
            except Exception as e:
                logger.warning(f"Failed to process {asset_name} for uncertainty: {e}")
                continue
        
        if len(uncertainty_features) > 10:  # Need minimum data
            X_unc = np.array(uncertainty_features)
            y_unc = np.array(uncertainty_targets)
            
            # Scale features
            X_unc_scaled = self.uncertainty_scaler.fit_transform(X_unc)
            
            # Train uncertainty models
            self.uncertainty_models = [
                RandomForestRegressor(n_estimators=50, random_state=42),
                RandomForestRegressor(n_estimators=100, random_state=43)
            ]
            
            for model in self.uncertainty_models:
                model.fit(X_unc_scaled, y_unc)
    
    def detect_regime(self, market_features: np.ndarray) -> RegimeDetectionResult:
        """
        Detect regime with enhanced uncertainty quantification.
        
        Args:
            market_features: Market features array
            
        Returns:
            Enhanced RegimeDetectionResult
        """
        # Get base prediction
        base_result = self.base_detector.detect_regime(market_features)
        
        # Enhance uncertainty if models are available
        if self.uncertainty_models and len(market_features) == self.input_dim:
            try:
                # Scale features
                features_scaled = self.uncertainty_scaler.transform(market_features.reshape(1, -1))
                
                # Get uncertainty predictions
                uncertainty_predictions = []
                for model in self.uncertainty_models:
                    pred = model.predict(features_scaled)[0]
                    uncertainty_predictions.append(pred)
                
                # Calculate enhanced uncertainty
                enhanced_uncertainty = np.mean(uncertainty_predictions)
                uncertainty_std = np.std(uncertainty_predictions)
                
                # Adjust regime uncertainty
                base_result.regime_uncertainty = base_result.regime_uncertainty * enhanced_uncertainty
                
                # Adjust confidence
                base_result.confidence = base_result.confidence * (1.0 - uncertainty_std)
                base_result.confidence = max(0.0, min(1.0, base_result.confidence))
                
            except Exception as e:
                logger.warning(f"Enhanced uncertainty calculation failed: {e}")
        
        # Store in history
        self.uncertainty_history.append(base_result)
        
        return base_result
    
    def get_enhanced_statistics(self) -> Dict[str, Any]:
        """Get enhanced statistics including uncertainty analysis."""
        base_stats = self.base_detector.get_regime_statistics()
        
        if not self.uncertainty_history:
            return base_stats
        
        # Add uncertainty statistics
        uncertainties = [r.regime_uncertainty for r in self.uncertainty_history]
        confidences = [r.confidence for r in self.uncertainty_history]
        
        base_stats.update({
            'average_uncertainty': np.mean([np.mean(u) for u in uncertainties]),
            'uncertainty_std': np.std([np.mean(u) for u in uncertainties]),
            'confidence_std': np.std(confidences),
            'uncertainty_trend': self._calculate_uncertainty_trend(),
            'enhanced_detections': len(self.uncertainty_history)
        })
        
        return base_stats
    
    def _calculate_uncertainty_trend(self) -> float:
        """Calculate trend in uncertainty over time."""
        if len(self.uncertainty_history) < 10:
            return 0.0
        
        uncertainties = [np.mean(r.regime_uncertainty) for r in self.uncertainty_history[-20:]]
        x = np.arange(len(uncertainties))
        slope = np.polyfit(x, uncertainties, 1)[0]
        return slope


def create_regime_detector(input_dim: int = 20, num_regimes: int = 3, 
                          enhanced: bool = True) -> MarketRegimeDetector:
    """
    Convenience function to create regime detector.
    
    Args:
        input_dim: Number of input features
        num_regimes: Number of regimes
        enhanced: Whether to use enhanced version
        
    Returns:
        Regime detector instance
    """
    if enhanced:
        return MarketRegimeDetectionBNN(input_dim, num_regimes)
    else:
        return MarketRegimeDetector(input_dim, num_regimes)


if __name__ == '__main__':
    # Example usage
    print("Market Regime Detection BNN Module")
    print("This module provides market regime detection with uncertainty quantification.")
    print("Use MarketRegimeDetectionBNN class for enhanced regime detection.")
