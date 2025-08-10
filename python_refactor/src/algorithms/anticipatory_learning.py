"""
Anticipatory Learning Module

Revised implementation to include:
- Dirichlet MAP filtering for portfolio weights
- Stochastic state observation via Monte Carlo simulation
- Future uncertainty quantification
- Integration with Kalman filter predictions
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from .kalman_filter import KalmanFilter
from .statistics import multivariate_normal_sample, normal_cdf, linear_entropy
from .solution import Solution

class AnticipatoryLearning:
    """Anticipatory learning system with Dirichlet MAP filtering."""
    
    def __init__(self, learning_rate: float = 0.01, prediction_horizon: int = 30,
                 monte_carlo_simulations: int = 1000, state_observation_frequency: int = 10,
                 error_threshold: float = 0.05, learning_type: str = "single_solution"):
        """
        Initialize anticipatory learning system.
        
        Args:
            learning_rate: Learning rate for state updates
            prediction_horizon: Number of time steps to predict ahead
            monte_carlo_simulations: Number of Monte Carlo simulations
            state_observation_frequency: Frequency of state observations
            error_threshold: Threshold for prediction error
            learning_type: Type of learning ("single_solution" or "population")
        """
        self.learning_rate = learning_rate
        self.prediction_horizon = prediction_horizon
        self.monte_carlo_simulations = monte_carlo_simulations
        self.state_observation_frequency = state_observation_frequency
        self.error_threshold = error_threshold
        self.learning_type = learning_type
        
        # Kalman filter for state tracking
        self.kalman_filter = KalmanFilter()
        
        # Learning history
        self.learning_history = []
        self.prediction_errors = []
        self.state_qualities = []
        
    def learn_single_solution(self, solution: Solution, current_time: int):
        """
        Apply anticipatory learning to a single solution.
        
        Args:
            solution: Solution to apply learning to
            current_time: Current time step
        """
        # Observe state via Monte Carlo simulation
        self._observe_state(solution, current_time)
        
        # Compute prediction error and uncertainty
        prediction_error = self._compute_prediction_error(solution, current_time)
        state_quality = self._compute_state_quality(solution)
        
        # Compute non-dominance probability
        nd_probability = self._compute_non_dominance_probability(solution)
        
        # Compute learning confidence
        alpha = 1.0 - linear_entropy(nd_probability)
        
        # Update solution state based on anticipatory knowledge
        self._update_solution_state(solution, alpha)
        
        # Store learning metrics
        solution.anticipation = True
        solution.alpha = alpha
        solution.prediction_error = prediction_error
        
        # Log learning event
        self.learning_history.append({
            'timestamp': datetime.now().isoformat(),
            'solution_id': id(solution),
            'current_time': current_time,
            'alpha': alpha,
            'prediction_error': prediction_error,
            'state_quality': state_quality,
            'nd_probability': nd_probability
        })
        
        self.prediction_errors.append(prediction_error)
        self.state_qualities.append(state_quality)
    
    def learn_population(self, population: List[Solution], current_time: int):
        """
        Apply anticipatory learning to entire population.
        
        Args:
            population: Population of solutions
            current_time: Current time step
        """
        for solution in population:
            if not hasattr(solution, 'anticipation') or not solution.anticipation:
                self.learn_single_solution(solution, current_time)
    
    def _observe_state(self, solution: Solution, current_time: int):
        """
        Observe portfolio state via Monte Carlo simulation.
        
        Args:
            solution: Solution to observe
            current_time: Current time step
        """
        portfolio = solution.P
        kalman_state = portfolio.kalman_state
        
        # Initialize state if not already done
        if not hasattr(portfolio, 'state_initialized') or not portfolio.state_initialized:
            self._initialize_state(portfolio, current_time)
            portfolio.state_initialized = True
        
        # Run Monte Carlo simulations for state observation
        roi_samples = []
        risk_samples = []
        
        for _ in range(self.monte_carlo_simulations):
            # Sample from current portfolio state
            current_state = kalman_state.x
            
            # Generate future scenarios
            future_state = self._simulate_future_state(current_state, kalman_state.P)
            
            # Extract ROI and risk from future state
            future_roi = future_state[0]
            future_risk = future_state[1]
            
            roi_samples.append(future_roi)
            risk_samples.append(future_risk)
        
        # Compute statistics from samples
        mean_roi = np.mean(roi_samples)
        mean_risk = np.mean(risk_samples)
        var_roi = np.var(roi_samples)
        var_risk = np.var(risk_samples)
        cov_roi_risk = np.cov(roi_samples, risk_samples)[0, 1] if len(roi_samples) > 1 else 0.0
        
        # Update measurement noise covariance
        kalman_state.R = np.array([
            [var_roi / self.monte_carlo_simulations, cov_roi_risk / self.monte_carlo_simulations],
            [cov_roi_risk / self.monte_carlo_simulations, var_risk / self.monte_carlo_simulations]
        ])
        
        # Update Kalman filter with observed state
        measurement = np.array([mean_roi, mean_risk])
        self.kalman_filter.update(kalman_state, measurement)
        
        # Store predictions
        portfolio.ROI_prediction = kalman_state.x_next[0] if hasattr(kalman_state, 'x_next') else mean_roi
        portfolio.risk_prediction = kalman_state.x_next[1] if hasattr(kalman_state, 'x_next') else mean_risk
        portfolio.error_covar_prediction = kalman_state.P_next if hasattr(kalman_state, 'P_next') else kalman_state.P
    
    def _initialize_state(self, portfolio, current_time: int):
        """Initialize Kalman filter state."""
        kalman_state = portfolio.kalman_state
        
        # Initial state: [ROI, risk, ROI_velocity, risk_velocity]
        initial_state = np.array([portfolio.ROI, portfolio.risk, 0.0, 0.0])
        kalman_state.x = initial_state
        
        # Initial covariance matrix
        kalman_state.P = np.array([
            [0.1, 0.0, 0.0, 0.0],
            [0.0, 0.1, 0.0, 0.0],
            [0.0, 0.0, 1000.0, 0.0],
            [0.0, 0.0, 0.0, 1000.0]
        ])
        
        # State transition matrix (constant velocity model)
        kalman_state.F = np.array([
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
        
        # Measurement matrix
        kalman_state.H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ])
        
        # Initial measurement noise
        kalman_state.R = np.array([
            [0.01, 0.0],
            [0.0, 0.01]
        ])
    
    def _simulate_future_state(self, current_state: np.ndarray, covariance: np.ndarray) -> np.ndarray:
        """
        Simulate future state using Monte Carlo sampling.
        
        Args:
            current_state: Current state vector
            covariance: State covariance matrix
            
        Returns:
            Simulated future state
        """
        # Sample from multivariate normal distribution
        future_state = multivariate_normal_sample(current_state, covariance)
        
        # Apply state transition
        F = np.array([
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
        
        return F @ future_state
    
    def _compute_prediction_error(self, solution: Solution, current_time: int) -> float:
        """
        Compute prediction error for solution.
        
        Args:
            solution: Solution to evaluate
            current_time: Current time step
            
        Returns:
            Prediction error
        """
        portfolio = solution.P
        
        # Compare predicted vs current values
        roi_error = abs(portfolio.ROI_prediction - portfolio.ROI)
        risk_error = abs(portfolio.risk_prediction - portfolio.risk)
        
        # Normalize errors
        roi_error_norm = roi_error / (abs(portfolio.ROI) + 1e-8)
        risk_error_norm = risk_error / (abs(portfolio.risk) + 1e-8)
        
        # Combined error
        prediction_error = (roi_error_norm + risk_error_norm) / 2.0
        
        return prediction_error
    
    def _compute_state_quality(self, solution: Solution) -> float:
        """
        Compute quality of state observation.
        
        Args:
            solution: Solution to evaluate
            
        Returns:
            State quality measure
        """
        kalman_state = solution.P.kalman_state
        
        # State quality based on covariance matrix determinant
        # Lower determinant = higher quality (less uncertainty)
        covariance_det = np.linalg.det(kalman_state.P[:2, :2])  # ROI and risk only
        
        # Normalize to [0, 1] range
        state_quality = 1.0 / (1.0 + covariance_det)
        
        return state_quality
    
    def _compute_non_dominance_probability(self, solution: Solution) -> float:
        """
        Compute probability that solution will be non-dominated in future.
        
        Args:
            solution: Solution to evaluate
            
        Returns:
            Non-dominance probability
        """
        portfolio = solution.P
        kalman_state = portfolio.kalman_state
        
        # Current state
        current_roi = portfolio.ROI
        current_risk = portfolio.risk
        
        # Predicted state
        predicted_roi = portfolio.ROI_prediction
        predicted_risk = portfolio.risk_prediction
        
        # Compute deltas
        delta1 = np.array([current_roi - predicted_roi, current_risk - predicted_risk])
        delta2 = np.array([predicted_roi - current_roi, predicted_risk - current_risk])
        
        # Point of interest: u = [0, 0]^T
        u = np.array([0.0, 0.0])
        
        # Combine prediction and observation covariances
        covar = kalman_state.P_next[:2, :2] + kalman_state.P[:2, :2]
        
        # Compute Cholesky decomposition
        try:
            L = np.linalg.cholesky(covar)
            L_inv = np.linalg.inv(L)
            
            # Transform to standard normal
            z1 = L_inv @ (u - delta1)
            z2 = L_inv @ (u - delta2)
            
            # Compute probabilities
            prob1 = normal_cdf(z1, np.eye(2))
            prob2 = normal_cdf(z2, np.eye(2))
            
            # Non-dominance probability
            nd_probability = prob1 + prob2
            
        except np.linalg.LinAlgError:
            # Fallback if matrix is not positive definite
            nd_probability = 0.5
        
        return nd_probability
    
    def _update_solution_state(self, solution: Solution, alpha: float):
        """
        Update solution state based on anticipatory knowledge.
        
        Args:
            solution: Solution to update
            alpha: Learning confidence parameter
        """
        portfolio = solution.P
        kalman_state = portfolio.kalman_state
        
        # Update state vector
        if hasattr(kalman_state, 'x_next'):
            kalman_state.x = kalman_state.x + alpha * (kalman_state.x_next - kalman_state.x)
        
        # Update covariance matrix
        if hasattr(kalman_state, 'P_next'):
            kalman_state.P = kalman_state.P + alpha * (kalman_state.P_next - kalman_state.P)
        
        # Update portfolio metrics
        portfolio.ROI = kalman_state.x[0]
        portfolio.risk = kalman_state.x[1]
        
        # Update robust/non-robust metrics
        if hasattr(portfolio, 'robustness') and portfolio.robustness:
            portfolio.robust_ROI = portfolio.ROI
            portfolio.robust_risk = portfolio.risk
        else:
            portfolio.non_robust_ROI = portfolio.ROI
            portfolio.non_robust_risk = portfolio.risk
    
    def apply_dirichlet_map_filtering(self, solution: Solution):
        """
        Apply Dirichlet MAP filtering to portfolio weights.
        
        Args:
            solution: Solution with portfolio weights
        """
        portfolio = solution.P
        weights = portfolio.investment
        
        # Dirichlet MAP estimation
        # Prior: uniform distribution (alpha = 1 for all assets)
        alpha_prior = np.ones_like(weights)
        
        # Likelihood: current weights as observations
        # For MAP estimation, we treat current weights as "pseudo-counts"
        alpha_likelihood = weights * 10  # Scale to make them more like counts
        
        # Posterior: sum of prior and likelihood
        alpha_posterior = alpha_prior + alpha_likelihood
        
        # MAP estimate: mode of Dirichlet distribution
        # Mode = (alpha_i - 1) / (sum(alpha) - K) where K is number of assets
        K = len(weights)
        sum_alpha = np.sum(alpha_posterior)
        
        if sum_alpha > K:
            map_weights = (alpha_posterior - 1) / (sum_alpha - K)
        else:
            # Fallback to uniform distribution
            map_weights = np.ones_like(weights) / K
        
        # Ensure weights sum to 1 and are non-negative
        map_weights = np.maximum(map_weights, 0.0)
        map_weights = map_weights / np.sum(map_weights)
        
        # Update portfolio weights
        portfolio.investment = map_weights
        
        # Update cardinality
        portfolio.cardinality = np.sum(map_weights > 0.01)  # Count assets with >1% allocation
    
    def get_learning_metrics(self) -> Dict[str, Any]:
        """Get learning performance metrics."""
        if not self.learning_history:
            return {}
        
        # Compute metrics from history
        alphas = [entry['alpha'] for entry in self.learning_history]
        prediction_errors = [entry['prediction_error'] for entry in self.learning_history]
        state_qualities = [entry['state_quality'] for entry in self.learning_history]
        
        metrics = {
            'total_learning_events': len(self.learning_history),
            'mean_alpha': np.mean(alphas),
            'std_alpha': np.std(alphas),
            'mean_prediction_error': np.mean(prediction_errors),
            'std_prediction_error': np.std(prediction_errors),
            'mean_state_quality': np.mean(state_qualities),
            'std_state_quality': np.std(state_qualities),
            'learning_trend': self._compute_learning_trend()
        }
        
        return metrics
    
    def _compute_learning_trend(self) -> float:
        """Compute learning trend based on prediction error."""
        if len(self.prediction_errors) < 10:
            return 0.0
        
        # Use last 10 errors to compute trend
        recent_errors = self.prediction_errors[-10:]
        
        # Compute linear trend (negative slope means improvement)
        x = np.arange(len(recent_errors))
        slope = np.polyfit(x, recent_errors, 1)[0]
        
        return -slope  # Return positive for improvement
    
    def reset(self):
        """Reset learning system."""
        self.learning_history = []
        self.prediction_errors = []
        self.state_qualities = [] 