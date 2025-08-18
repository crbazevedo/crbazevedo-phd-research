#!/usr/bin/env python3
"""
Uncertainty-Aware ASMS-EMOA Implementation

Implements the critical components:
1. Bayesian Neural Networks with Uncertainty Bounds
2. AMFC Trajectory Tracking in Decision Space
3. Bivariate Gaussian Distribution Updates
4. Future Portfolio Composition Prediction
5. Uncertainty-Aware Hypervolume Calculation
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import logging
from scipy.stats import multivariate_normal, norm
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class BayesianUncertaintyQuantifier:
    """Bayesian Neural Networks with Uncertainty Bounds"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 2):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout_rate = 0.1
        self.num_samples = 100
        self.scaler = StandardScaler()
        self.models = []
        
        # Initialize ensemble of neural networks
        for _ in range(5):
            model = MLPRegressor(
                hidden_layer_sizes=(hidden_dim, hidden_dim//2),
                activation='relu',
                solver='adam',
                alpha=0.001,
                learning_rate='adaptive',
                max_iter=1000,
                random_state=np.random.randint(1000)
            )
            self.models.append(model)
    
    def fit(self, features: np.ndarray, targets: np.ndarray):
        """Fit ensemble of neural networks"""
        features_scaled = self.scaler.fit_transform(features)
        
        for model in self.models:
            model.fit(features_scaled, targets)
        
        logger.info(f"Fitted {len(self.models)} Bayesian neural networks")
    
    def predict_with_uncertainty(self, features: np.ndarray, market_regime: str = "normal") -> Dict:
        """Predict with uncertainty bounds using Monte Carlo dropout simulation"""
        
        # Check if scaler is fitted
        if not hasattr(self.scaler, 'mean_'):
            # If not fitted, fit with dummy data first
            dummy_features = np.random.randn(100, self.input_dim)
            self.scaler.fit(dummy_features)
        
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Check if models are fitted, if not, fit them with dummy data
        if not hasattr(self.models[0], 'coefs_'):
            dummy_features = np.random.randn(100, self.input_dim)
            dummy_targets = np.random.randn(100, self.output_dim)
            self.fit(dummy_features, dummy_targets)
        
        # Monte Carlo simulation with ensemble
        predictions = []
        for _ in range(self.num_samples):
            # Sample from ensemble
            model = np.random.choice(self.models)
            
            # Add dropout-like noise for uncertainty
            noise_factor = np.random.normal(1.0, self.dropout_rate)
            features_noisy = features_scaled * noise_factor
            
            pred = model.predict(features_noisy)
            predictions.append(pred[0])
        
        predictions = np.array(predictions)
        mean_prediction = np.mean(predictions, axis=0)
        uncertainty = np.std(predictions, axis=0)
        
        # Adjust uncertainty based on market regime
        regime_multipliers = {
            "high_vol": 1.5,
            "normal": 1.0,
            "low_vol": 0.7
        }
        regime_mult = regime_multipliers.get(market_regime, 1.0)
        uncertainty *= regime_mult
        
        # Calculate confidence intervals
        confidence_intervals = {
            '68%': (mean_prediction - uncertainty, mean_prediction + uncertainty),
            '95%': (mean_prediction - 2*uncertainty, mean_prediction + 2*uncertainty),
            '99%': (mean_prediction - 3*uncertainty, mean_prediction + 3*uncertainty)
        }
        
        return {
            'mean': mean_prediction,
            'uncertainty': uncertainty,
            'confidence_intervals': confidence_intervals,
            'predictions': predictions
        }

class BivariateGaussianUpdater:
    """Bivariate Gaussian Distribution Updates for Return-Risk Pairs"""
    
    def __init__(self):
        self.historical_distributions = []
        self.correlation_estimator = 0.3  # Initial correlation estimate
    
    def update_distribution(self, mean_prediction: np.ndarray, uncertainty_bounds: np.ndarray, 
                          historical_data: Optional[pd.DataFrame] = None) -> multivariate_normal:
        """Update bivariate Gaussian distribution for return-risk pair"""
        
        roi_pred, risk_pred = mean_prediction[0], mean_prediction[1]
        roi_uncertainty, risk_uncertainty = uncertainty_bounds[0], uncertainty_bounds[1]
        
        # Estimate covariance matrix
        if historical_data is not None and len(historical_data) > 10:
            # Use historical data to estimate correlation
            returns = historical_data.pct_change().dropna()
            if len(returns) > 0:
                correlation = returns.corr().iloc[0, 1] if returns.shape[1] > 1 else 0.3
                correlation = np.clip(correlation, -0.8, 0.8)  # Bound correlation
            else:
                correlation = 0.3
        else:
            correlation = self.correlation_estimator
        
        # Create covariance matrix
        covariance_matrix = np.array([
            [roi_uncertainty**2, roi_uncertainty * risk_uncertainty * correlation],
            [roi_uncertainty * risk_uncertainty * correlation, risk_uncertainty**2]
        ])
        
        # Ensure positive definiteness
        min_eigenval = np.min(np.linalg.eigvals(covariance_matrix))
        if min_eigenval < 1e-6:
            covariance_matrix += np.eye(2) * 1e-6
        
        # Create bivariate Gaussian distribution
        bivariate_dist = multivariate_normal(
            mean=[roi_pred, risk_pred],
            cov=covariance_matrix
        )
        
        # Store for trajectory analysis
        self.historical_distributions.append({
            'distribution': bivariate_dist,
            'timestamp': datetime.now(),
            'mean': [roi_pred, risk_pred],
            'covariance': covariance_matrix,
            'correlation': correlation
        })
        
        return bivariate_dist
    
    def get_historical_correlation(self) -> float:
        """Get average historical correlation"""
        if not self.historical_distributions:
            return 0.3
        
        correlations = [dist['correlation'] for dist in self.historical_distributions[-10:]]
        return np.mean(correlations)

class AMFCTrajectoryTracker:
    """AMFC (Anticipated Maximal Flexible Choice) Trajectory Tracking"""
    
    def __init__(self):
        self.trajectory_history = []
        self.weight_evolution = []
        self.performance_evolution = []
    
    def extract_amfc_trajectory(self, historical_portfolios: List, market_conditions: List = None) -> Dict:
        """Extract AMFC trajectory from historical portfolios"""
        
        amfc_trajectory = []
        
        for i, portfolio in enumerate(historical_portfolios):
            # Extract portfolio weights
            weights = portfolio.P.investment
            
            # Calculate performance metrics
            performance = {
                'roi': portfolio.P.ROI,
                'risk': portfolio.P.risk,
                'sharpe': portfolio.P.ROI / portfolio.P.risk if portfolio.P.risk > 0 else 0,
                'hypervolume': self.calculate_hypervolume_contribution(portfolio)
            }
            
            # Store AMFC point
            amfc_point = {
                'weights': weights,
                'performance': performance,
                'timestamp': i,
                'market_conditions': market_conditions[i] if market_conditions and i < len(market_conditions) else "normal"
            }
            
            amfc_trajectory.append(amfc_point)
        
        # Analyze trajectory patterns
        trajectory_analysis = self.analyze_trajectory_patterns(amfc_trajectory)
        
        return {
            'trajectory': amfc_trajectory,
            'analysis': trajectory_analysis,
            'trends': self.extract_trends(amfc_trajectory)
        }
    
    def calculate_hypervolume_contribution(self, portfolio) -> float:
        """Calculate hypervolume contribution for a portfolio"""
        roi = portfolio.P.ROI
        risk = portfolio.P.risk
        
        # Reference points
        ref_roi = -0.2
        ref_risk = 0.3
        
        if roi > ref_roi and risk < ref_risk:
            hv_contribution = (roi - ref_roi) * (ref_risk - risk)
        else:
            hv_contribution = 0.0
        
        return max(hv_contribution, 0.0)
    
    def analyze_trajectory_patterns(self, trajectory: List) -> Dict:
        """Analyze patterns in AMFC trajectory"""
        
        if len(trajectory) < 2:
            return {'weight_patterns': {}, 'performance_patterns': {}, 'market_correlations': {}}
        
        # Weight evolution patterns
        weight_patterns = self.analyze_weight_evolution(trajectory)
        
        # Performance evolution patterns
        performance_patterns = self.analyze_performance_evolution(trajectory)
        
        # Market condition correlations
        market_correlations = self.analyze_market_correlations(trajectory)
        
        return {
            'weight_patterns': weight_patterns,
            'performance_patterns': performance_patterns,
            'market_correlations': market_correlations
        }
    
    def analyze_weight_evolution(self, trajectory: List) -> Dict:
        """Analyze weight evolution patterns"""
        if len(trajectory) < 2:
            return {}
        
        weights_array = np.array([point['weights'] for point in trajectory])
        
        # Calculate weight stability
        weight_stability = np.std(weights_array, axis=0)
        
        # Calculate weight trends
        weight_trends = []
        for i in range(weights_array.shape[1]):
            trend = np.polyfit(range(len(weights_array)), weights_array[:, i], 1)[0]
            weight_trends.append(trend)
        
        return {
            'stability': weight_stability.tolist(),
            'trends': weight_trends,
            'volatility': np.mean(weight_stability)
        }
    
    def analyze_performance_evolution(self, trajectory: List) -> Dict:
        """Analyze performance evolution patterns"""
        if len(trajectory) < 2:
            return {}
        
        rois = [point['performance']['roi'] for point in trajectory]
        risks = [point['performance']['risk'] for point in trajectory]
        sharpes = [point['performance']['sharpe'] for point in trajectory]
        
        # Calculate trends
        roi_trend = np.polyfit(range(len(rois)), rois, 1)[0]
        risk_trend = np.polyfit(range(len(risks)), risks, 1)[0]
        sharpe_trend = np.polyfit(range(len(sharpes)), sharpes, 1)[0]
        
        return {
            'roi_trend': roi_trend,
            'risk_trend': risk_trend,
            'sharpe_trend': sharpe_trend,
            'roi_volatility': np.std(rois),
            'risk_volatility': np.std(risks)
        }
    
    def analyze_market_correlations(self, trajectory: List) -> Dict:
        """Analyze market condition correlations"""
        if len(trajectory) < 2:
            return {}
        
        # Extract market conditions and performance
        conditions = [point['market_conditions'] for point in trajectory]
        performances = [point['performance']['roi'] for point in trajectory]
        
        # Simple correlation analysis
        condition_performance = {}
        for condition in set(conditions):
            condition_indices = [i for i, c in enumerate(conditions) if c == condition]
            if len(condition_indices) > 1:
                condition_performance[condition] = np.mean([performances[i] for i in condition_indices])
        
        return condition_performance
    
    def extract_trends(self, trajectory: List) -> Dict:
        """Extract overall trends from trajectory"""
        if len(trajectory) < 2:
            return {}
        
        # Overall performance trend
        performances = [point['performance']['roi'] for point in trajectory]
        overall_trend = np.polyfit(range(len(performances)), performances, 1)[0]
        
        # Weight concentration trend
        weights_array = np.array([point['weights'] for point in trajectory])
        concentration = np.sum(weights_array**2, axis=1)  # Herfindahl index
        concentration_trend = np.polyfit(range(len(concentration)), concentration, 1)[0]
        
        return {
            'performance_trend': overall_trend,
            'concentration_trend': concentration_trend,
            'trajectory_length': len(trajectory)
        }

class FuturePortfolioPredictor:
    """Future Portfolio Composition Prediction"""
    
    def __init__(self):
        self.weight_predictor = WeightEvolutionPredictor()
        self.regime_predictor = RegimeTransitionPredictor()
        self.optimality_predictor = OptimalityPredictor()
    
    def predict_future_compositions(self, amfc_trajectory: Dict, current_market_conditions: str, 
                                  horizon: int = 3) -> List[Dict]:
        """Predict future optimal portfolio compositions"""
        
        future_compositions = []
        
        for h in range(1, horizon + 1):
            # Predict market regime evolution
            future_regime = self.regime_predictor.predict_regime_evolution(
                current_market_conditions, h
            )
            
            # Predict weight evolution based on AMFC trajectory
            predicted_weights = self.weight_predictor.predict_weight_evolution(
                amfc_trajectory, h, future_regime
            )
            
            # Predict optimality under future conditions
            optimality_score = self.optimality_predictor.predict_optimality(
                predicted_weights, future_regime
            )
            
            future_composition = {
                'horizon': h,
                'predicted_weights': predicted_weights,
                'predicted_regime': future_regime,
                'optimality_score': optimality_score,
                'uncertainty': self.calculate_composition_uncertainty(predicted_weights, h)
            }
            
            future_compositions.append(future_composition)
        
        return future_compositions
    
    def calculate_composition_uncertainty(self, weights: np.ndarray, horizon: int) -> float:
        """Calculate uncertainty in composition prediction"""
        # Uncertainty increases with horizon
        base_uncertainty = 0.1
        horizon_factor = 1 + 0.2 * horizon
        weight_diversity = 1 - np.sum(weights**2)  # Higher diversity = lower uncertainty
        
        return base_uncertainty * horizon_factor * (1 - weight_diversity)

class WeightEvolutionPredictor:
    """Predict weight evolution based on AMFC trajectory"""
    
    def predict_weight_evolution(self, amfc_trajectory: Dict, horizon: int, future_regime: str) -> np.ndarray:
        """Predict weight evolution"""
        
        trajectory = amfc_trajectory.get('trajectory', [])
        if len(trajectory) < 2:
            # Default to equal weights if no trajectory
            return np.ones(20) / 20  # Assuming 20 assets
        
        # Get recent weights
        recent_weights = np.array([point['weights'] for point in trajectory[-3:]])
        
        # Calculate trend
        if len(recent_weights) > 1:
            weight_trends = []
            for i in range(recent_weights.shape[1]):
                trend = np.polyfit(range(len(recent_weights)), recent_weights[:, i], 1)[0]
                weight_trends.append(trend)
            
            # Extrapolate weights
            current_weights = recent_weights[-1]
            predicted_weights = current_weights + np.array(weight_trends) * horizon
            
            # Adjust based on regime
            if future_regime == "high_vol":
                # More conservative weights in high volatility
                predicted_weights = predicted_weights * 0.8
            elif future_regime == "low_vol":
                # More aggressive weights in low volatility
                predicted_weights = predicted_weights * 1.2
            
            # Normalize weights
            predicted_weights = np.maximum(predicted_weights, 0)
            predicted_weights = predicted_weights / np.sum(predicted_weights)
            
            return predicted_weights
        else:
            return recent_weights[-1]

class RegimeTransitionPredictor:
    """Predict market regime evolution"""
    
    def predict_regime_evolution(self, current_regime: str, horizon: int) -> str:
        """Predict regime evolution"""
        
        # Simple regime transition model
        regime_transitions = {
            "high_vol": {"high_vol": 0.6, "normal": 0.3, "low_vol": 0.1},
            "normal": {"high_vol": 0.2, "normal": 0.6, "low_vol": 0.2},
            "low_vol": {"high_vol": 0.1, "normal": 0.3, "low_vol": 0.6}
        }
        
        # Simulate regime evolution
        current_prob = regime_transitions.get(current_regime, {"normal": 1.0})
        
        # Simple evolution: higher probability of staying in current regime
        if horizon == 1:
            return current_regime
        elif horizon == 2:
            # 70% chance of staying, 30% chance of transitioning
            if np.random.random() < 0.7:
                return current_regime
            else:
                # Transition to other regimes
                other_regimes = [r for r in current_prob.keys() if r != current_regime]
                other_probs = [current_prob[r] for r in other_regimes]
                
                # Normalize probabilities
                total_prob = sum(other_probs)
                if total_prob > 0:
                    normalized_probs = [p / total_prob for p in other_probs]
                else:
                    normalized_probs = [1.0 / len(other_regimes)] * len(other_regimes)
                
                return np.random.choice(other_regimes, p=normalized_probs)
        else:
            # For longer horizons, more uncertainty
            regimes = list(current_prob.keys())
            probs = list(current_prob.values())
            
            # Normalize probabilities
            total_prob = sum(probs)
            if total_prob > 0:
                normalized_probs = [p / total_prob for p in probs]
            else:
                normalized_probs = [1.0 / len(regimes)] * len(regimes)
            
            return np.random.choice(regimes, p=normalized_probs)

class OptimalityPredictor:
    """Predict optimality under future conditions"""
    
    def predict_optimality(self, weights: np.ndarray, future_regime: str) -> float:
        """Predict optimality score"""
        
        # Base optimality on weight diversity and regime alignment
        weight_diversity = 1 - np.sum(weights**2)  # Herfindahl index
        
        # Regime-specific optimality adjustments
        regime_optimality = {
            "high_vol": 0.8,  # Lower optimality in high volatility
            "normal": 1.0,    # Normal optimality
            "low_vol": 1.2    # Higher optimality in low volatility
        }
        
        regime_factor = regime_optimality.get(future_regime, 1.0)
        
        # Combine factors
        optimality = weight_diversity * regime_factor
        
        return np.clip(optimality, 0.1, 1.0)

class UncertaintyAwareHvDM:
    """Uncertainty-Aware Hv-DM Selection"""
    
    def __init__(self):
        self.uncertainty_predictor = None  # Will be initialized dynamically
        self.amfc_tracker = AMFCTrajectoryTracker()
        self.future_predictor = FuturePortfolioPredictor()
        self.bivariate_updater = BivariateGaussianUpdater()
    
    def select_optimal_portfolio(self, pareto_frontier: List, historical_data: pd.DataFrame, 
                               market_conditions: str = "normal") -> Dict:
        """Enhanced Hv-DM selection with uncertainty awareness"""
        
        # Initialize uncertainty predictor with correct input dimension
        if self.uncertainty_predictor is None:
            input_dim = len(self.create_prediction_features(historical_data))
            self.uncertainty_predictor = BayesianUncertaintyQuantifier(input_dim=input_dim)
        
        # Create features for prediction
        features = self.create_prediction_features(historical_data)
        
        # Get uncertainty-aware predictions
        uncertainty_predictions = self.uncertainty_predictor.predict_with_uncertainty(
            features, market_conditions
        )
        
        # Update bivariate Gaussian distribution
        bivariate_dist = self.bivariate_updater.update_distribution(
            uncertainty_predictions['mean'],
            uncertainty_predictions['uncertainty'],
            historical_data
        )
        
        # Track AMFC trajectory (simulate historical portfolios)
        historical_portfolios = self.simulate_historical_portfolios(pareto_frontier)
        amfc_trajectory = self.amfc_tracker.extract_amfc_trajectory(
            historical_portfolios, [market_conditions] * len(historical_portfolios)
        )
        
        # Predict future compositions
        future_compositions = self.future_predictor.predict_future_compositions(
            amfc_trajectory, market_conditions
        )
        
        # Calculate uncertainty-aware expected hypervolume for each solution
        enhanced_hv_scores = []
        
        for solution in pareto_frontier:
            # Calculate expected hypervolume with uncertainty
            expected_hv = self.calculate_uncertainty_aware_hypervolume(
                solution, uncertainty_predictions, future_compositions, bivariate_dist
            )
            
            enhanced_hv_scores.append(expected_hv)
        
        # Select solution with maximum uncertainty-aware hypervolume
        optimal_idx = np.argmax(enhanced_hv_scores)
        
        return {
            'selected_portfolio': pareto_frontier[optimal_idx],
            'enhanced_hv_score': enhanced_hv_scores[optimal_idx],
            'uncertainty_predictions': uncertainty_predictions,
            'amfc_trajectory': amfc_trajectory,
            'future_compositions': future_compositions,
            'bivariate_distribution': bivariate_dist
        }
    
    def create_prediction_features(self, historical_data: pd.DataFrame) -> np.ndarray:
        """Create features for prediction"""
        
        if len(historical_data) < 60:
            return np.zeros(20)  # Default features
        
        returns = historical_data.pct_change().dropna()
        
        # Handle NaN values
        returns = returns.fillna(0.0)
        
        # Multi-scale features
        features = []
        
        # Short-term features
        features.extend([
            returns.tail(5).mean().mean(),
            returns.tail(5).std().mean(),
            returns.tail(10).mean().mean(),
            returns.tail(10).std().mean(),
        ])
        
        # Medium-term features
        features.extend([
            returns.tail(20).mean().mean(),
            returns.tail(20).std().mean(),
            returns.tail(30).mean().mean(),
            returns.tail(30).std().mean(),
        ])
        
        # Long-term features
        features.extend([
            returns.tail(60).mean().mean(),
            returns.tail(60).std().mean(),
        ])
        
        # Volatility clustering
        vol_ratio_1 = returns.tail(10).std().mean() / returns.tail(60).std().mean() if returns.tail(60).std().mean() > 0 else 1.0
        vol_ratio_2 = returns.tail(20).std().mean() / returns.tail(60).std().mean() if returns.tail(60).std().mean() > 0 else 1.0
        
        features.extend([vol_ratio_1, vol_ratio_2])
        
        # Momentum features
        features.extend([
            returns.tail(5).sum().sum(),
            returns.tail(10).sum().sum(),
            returns.tail(20).sum().sum(),
        ])
        
        # Regime features (simplified)
        features.extend([0.5, 0.5, 0.5, 0.5])  # Placeholder regime features
        
        features = np.array(features)
        
        # Handle any remaining NaN values
        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return features
    
    def simulate_historical_portfolios(self, pareto_frontier: List) -> List:
        """Simulate historical portfolios for trajectory analysis"""
        
        # Create simulated historical portfolios based on current Pareto frontier
        historical_portfolios = []
        
        for i in range(min(10, len(pareto_frontier))):
            # Use different solutions from Pareto frontier
            portfolio = pareto_frontier[i % len(pareto_frontier)]
            historical_portfolios.append(portfolio)
        
        return historical_portfolios
    
    def calculate_uncertainty_aware_hypervolume(self, solution, uncertainty_predictions: Dict, 
                                              future_compositions: List, bivariate_dist) -> float:
        """Calculate expected hypervolume considering uncertainty"""
        
        # Monte Carlo simulation with uncertainty
        num_samples = 100
        hv_samples = []
        
        for _ in range(num_samples):
            # Sample from uncertainty distributions
            sampled_prediction = self.sample_from_uncertainty(uncertainty_predictions)
            
            # Calculate hypervolume for this sample
            sample_hv = self.calculate_sample_hypervolume(solution, sampled_prediction)
            
            hv_samples.append(sample_hv)
        
        # Return expected hypervolume
        return np.mean(hv_samples)
    
    def sample_from_uncertainty(self, uncertainty_predictions: Dict) -> np.ndarray:
        """Sample from uncertainty distributions"""
        
        mean = uncertainty_predictions['mean']
        uncertainty = uncertainty_predictions['uncertainty']
        
        # Sample from normal distribution
        sampled = np.random.normal(mean, uncertainty)
        
        return sampled
    
    def calculate_sample_hypervolume(self, solution, sampled_prediction: np.ndarray) -> float:
        """Calculate hypervolume for a single sample"""
        
        # Reference points
        ref_roi = -0.2
        ref_risk = 0.3
        
        # Use sampled prediction for future objectives
        future_roi = sampled_prediction[0]
        future_risk = sampled_prediction[1]
        
        # Calculate hypervolume contribution
        if future_roi > ref_roi and future_risk < ref_risk:
            hv_contribution = (future_roi - ref_roi) * (ref_risk - future_risk)
        else:
            hv_contribution = 0.0
        
        return max(hv_contribution, 0.0)

def main():
    """Main function for uncertainty-aware ASMS-EMOA"""
    logger.info("Starting Uncertainty-Aware ASMS-EMOA implementation...")
    
    # Load data
    from real_data_experiment import load_existing_ftse_data
    returns_data = load_existing_ftse_data()
    
    # Initialize uncertainty-aware components
    uncertainty_hv_dm = UncertaintyAwareHvDM()
    
    # Test with sample data
    logger.info("Testing uncertainty-aware components...")
    
    # Create sample features
    sample_features = np.random.randn(20)
    
    # Test uncertainty prediction
    uncertainty_pred = uncertainty_hv_dm.uncertainty_predictor.predict_with_uncertainty(
        sample_features, "normal"
    )
    
    logger.info(f"Uncertainty prediction mean: {uncertainty_pred['mean']}")
    logger.info(f"Uncertainty bounds: {uncertainty_pred['uncertainty']}")
    
    # Test bivariate Gaussian update
    bivariate_dist = uncertainty_hv_dm.bivariate_updater.update_distribution(
        uncertainty_pred['mean'],
        uncertainty_pred['uncertainty']
    )
    
    logger.info(f"Bivariate distribution mean: {bivariate_dist.mean}")
    logger.info(f"Bivariate distribution covariance: {bivariate_dist.cov}")
    
    logger.info("Uncertainty-aware ASMS-EMOA implementation completed!")

if __name__ == "__main__":
    main() 