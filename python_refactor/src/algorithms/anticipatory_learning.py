"""
Anticipatory Learning Module

Enhanced implementation to include:
- Anticipative distribution concept
- Adaptive learning rate with Kalman error and entropy
- 1-step ahead horizon for predictive decisions
- Predicted portfolio rebalancing for maximal expected hypervolume
- Stochastic Pareto frontier storage for visualization
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

class AnticipativeDistribution:
    """Represents the anticipative distribution for portfolio state prediction."""
    
    def __init__(self, current_state: np.ndarray, predicted_state: np.ndarray, 
                 current_covariance: np.ndarray, predicted_covariance: np.ndarray):
        """
        Initialize anticipative distribution.
        
        Args:
            current_state: Current portfolio state [ROI, risk, ROI_velocity, risk_velocity]
            predicted_state: Predicted portfolio state
            current_covariance: Current state covariance matrix
            predicted_covariance: Predicted state covariance matrix
        """
        self.current_state = current_state
        self.predicted_state = predicted_state
        self.current_covariance = current_covariance
        self.predicted_covariance = predicted_covariance
        
        # Combined covariance for anticipative distribution
        self.anticipative_covariance = current_covariance + predicted_covariance
        
        # Anticipative mean (weighted combination)
        self.anticipative_mean = (current_state + predicted_state) / 2.0
    
    def sample_anticipative_state(self, num_samples: int = 1000) -> np.ndarray:
        """Sample from anticipative distribution."""
        return multivariate_normal_sample(self.anticipative_mean, self.anticipative_covariance, num_samples)
    
    def compute_anticipative_confidence(self) -> float:
        """Compute confidence in anticipative distribution."""
        # Confidence based on determinant of anticipative covariance
        det = np.linalg.det(self.anticipative_covariance[:2, :2])  # ROI and risk only
        return 1.0 / (1.0 + det)

class AnticipatoryLearning:
    """Enhanced anticipatory learning system with adaptive learning rate and 1-step ahead horizon."""
    
    def __init__(self, learning_rate: float = 0.01, prediction_horizon: int = 1,  # Changed to 1-step ahead
                 monte_carlo_simulations: int = 1000, state_observation_frequency: int = 10,
                 error_threshold: float = 0.05, learning_type: str = "single_solution",
                 adaptive_learning: bool = True):
        """
        Initialize enhanced anticipatory learning system.
        
        Args:
            learning_rate: Base learning rate for state updates
            prediction_horizon: Number of time steps to predict ahead (default: 1)
            monte_carlo_simulations: Number of Monte Carlo simulations
            state_observation_frequency: Frequency of state observations
            error_threshold: Threshold for prediction error
            learning_type: Type of learning ("single_solution" or "population")
            adaptive_learning: Whether to use adaptive learning rate
        """
        self.base_learning_rate = learning_rate
        self.prediction_horizon = prediction_horizon
        self.monte_carlo_simulations = monte_carlo_simulations
        self.state_observation_frequency = state_observation_frequency
        self.error_threshold = error_threshold
        self.learning_type = learning_type
        self.adaptive_learning = adaptive_learning
        
        # Kalman filter for state tracking
        self.kalman_filter = KalmanFilter()
        
        # Learning history
        self.learning_history = []
        self.prediction_errors = []
        self.state_qualities = []
        self.adaptive_learning_rates = []
        
        # Anticipative distributions storage
        self.anticipative_distributions = []
        
        # Stochastic Pareto frontiers storage for visualization
        self.stochastic_pareto_frontiers = []
        self.anticipative_pareto_frontiers = []
        
    def learn_single_solution(self, solution: Solution, current_time: int):
        """
        Apply enhanced anticipatory learning to a single solution.
        
        Args:
            solution: Solution to apply learning to
            current_time: Current time step
        """
        # Observe state via Monte Carlo simulation with 1-step ahead horizon
        self._observe_state_1step_ahead(solution, current_time)
        
        # Create anticipative distribution
        anticipative_dist = self._create_anticipative_distribution(solution)
        self.anticipative_distributions.append(anticipative_dist)
        
        # Compute prediction error and uncertainty
        prediction_error = self._compute_prediction_error(solution, current_time)
        state_quality = self._compute_state_quality(solution)
        
        # Compute non-dominance probability
        nd_probability = self._compute_non_dominance_probability(solution)
        
        # Compute adaptive learning rate
        if self.adaptive_learning:
            alpha = self._compute_adaptive_learning_rate(solution, prediction_error, nd_probability)
        else:
            alpha = 1.0 - linear_entropy(nd_probability)
        
        # Update solution state based on anticipatory knowledge
        self._update_solution_state_anticipative(solution, alpha, anticipative_dist)
        
        # Apply predicted portfolio rebalancing for maximal expected hypervolume
        self._apply_predicted_rebalancing(solution, anticipative_dist)
        
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
            'nd_probability': nd_probability,
            'anticipative_confidence': anticipative_dist.compute_anticipative_confidence()
        })
        
        self.prediction_errors.append(prediction_error)
        self.state_qualities.append(state_quality)
        self.adaptive_learning_rates.append(alpha)
    
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
        
        # Store stochastic Pareto frontier for visualization
        self._store_stochastic_pareto_frontier(population, current_time)
    
    def _observe_state_1step_ahead(self, solution: Solution, current_time: int):
        """
        Observe portfolio state with 1-step ahead horizon.
        
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
        
        # Run Monte Carlo simulations for 1-step ahead prediction
        roi_samples = []
        risk_samples = []
        
        for _ in range(self.monte_carlo_simulations):
            # Sample from current portfolio state
            current_state = kalman_state.x
            
            # Generate 1-step ahead scenarios
            future_state = self._simulate_1step_ahead_state(current_state, kalman_state.P)
            
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
        
        # Store predictions for 1-step ahead
        portfolio.ROI_prediction = kalman_state.x_next[0] if hasattr(kalman_state, 'x_next') else mean_roi
        portfolio.risk_prediction = kalman_state.x_next[1] if hasattr(kalman_state, 'x_next') else mean_risk
        portfolio.error_covar_prediction = kalman_state.P_next if hasattr(kalman_state, 'P_next') else kalman_state.P
    
    def _simulate_1step_ahead_state(self, current_state: np.ndarray, covariance: np.ndarray) -> np.ndarray:
        """
        Simulate 1-step ahead state using Monte Carlo sampling.
        
        Args:
            current_state: Current state vector
            covariance: State covariance matrix
            
        Returns:
            Simulated 1-step ahead state
        """
        # Sample from multivariate normal distribution
        future_state = multivariate_normal_sample(current_state, covariance)
        
        # Apply 1-step state transition
        F = np.array([
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
        
        return F @ future_state
    
    def _create_anticipative_distribution(self, solution: Solution) -> AnticipativeDistribution:
        """
        Create anticipative distribution for solution.
        
        Args:
            solution: Solution to create distribution for
            
        Returns:
            Anticipative distribution
        """
        portfolio = solution.P
        kalman_state = portfolio.kalman_state
        
        current_state = kalman_state.x
        predicted_state = kalman_state.x_next if hasattr(kalman_state, 'x_next') else current_state
        current_covariance = kalman_state.P
        predicted_covariance = kalman_state.P_next if hasattr(kalman_state, 'P_next') else current_covariance
        
        return AnticipativeDistribution(
            current_state, predicted_state, current_covariance, predicted_covariance
        )
    
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
    
    def _compute_prediction_error(self, solution: Solution, current_time: int) -> float:
        """Compute prediction error for solution."""
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
        """Compute quality of state observation."""
        kalman_state = solution.P.kalman_state
        
        # State quality based on covariance matrix determinant
        covariance_det = np.linalg.det(kalman_state.P[:2, :2])
        
        # Normalize to [0, 1] range
        state_quality = 1.0 / (1.0 + covariance_det)
        
        return state_quality
    
    def _compute_non_dominance_probability(self, solution: Solution) -> float:
        """Compute probability that solution will be non-dominated in future."""
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
    
    def _compute_adaptive_learning_rate(self, solution: Solution, prediction_error: float, 
                                      nd_probability: float) -> float:
        """
        Compute adaptive learning rate based on Kalman error and entropy.
        
        Args:
            solution: Solution to compute learning rate for
            prediction_error: Current prediction error
            nd_probability: Non-dominance probability
            
        Returns:
            Adaptive learning rate
        """
        portfolio = solution.P
        kalman_state = portfolio.kalman_state
        
        # Kalman error: determinant of prediction covariance
        kalman_error = np.linalg.det(kalman_state.P_next[:2, :2]) if hasattr(kalman_state, 'P_next') else np.linalg.det(kalman_state.P[:2, :2])
        
        # Entropy over probability of dominance
        dominance_entropy = linear_entropy(nd_probability)
        
        # Adaptive learning rate: combines base rate, Kalman error, and entropy
        error_factor = 1.0 / (1.0 + kalman_error)
        entropy_factor = 1.0 - dominance_entropy
        
        adaptive_alpha = self.base_learning_rate * error_factor * entropy_factor
        
        # Ensure alpha is in [0, 1] range
        adaptive_alpha = np.clip(adaptive_alpha, 0.0, 1.0)
        
        return adaptive_alpha
    
    def _update_solution_state_anticipative(self, solution: Solution, alpha: float, 
                                          anticipative_dist: AnticipativeDistribution):
        """
        Update solution state using anticipative distribution.
        
        Args:
            solution: Solution to update
            alpha: Learning rate
            anticipative_dist: Anticipative distribution
        """
        portfolio = solution.P
        kalman_state = portfolio.kalman_state
        
        # Update state vector using anticipative distribution
        anticipative_state = anticipative_dist.anticipative_mean
        kalman_state.x = kalman_state.x + alpha * (anticipative_state - kalman_state.x)
        
        # Update covariance matrix
        anticipative_covariance = anticipative_dist.anticipative_covariance
        kalman_state.P = kalman_state.P + alpha * (anticipative_covariance - kalman_state.P)
        
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
    
    def _apply_predicted_rebalancing(self, solution: Solution, anticipative_dist: AnticipativeDistribution):
        """
        Apply predicted portfolio rebalancing for maximal expected hypervolume.
        
        Args:
            solution: Solution to rebalance
            anticipative_dist: Anticipative distribution
        """
        portfolio = solution.P
        
        # Sample from anticipative distribution to get predicted states
        predicted_states = anticipative_dist.sample_anticipative_state(100)
        
        # Extract predicted ROI and risk values
        predicted_rois = predicted_states[:, 0]
        predicted_risks = predicted_states[:, 1]
        
        # Compute expected hypervolume contribution for current weights
        current_expected_hv = self._compute_expected_hypervolume_contribution(
            predicted_rois, predicted_risks, portfolio.investment
        )
        
        # Optimize weights for maximal expected hypervolume
        optimal_weights = self._optimize_weights_for_expected_hypervolume(
            predicted_rois, predicted_risks, portfolio.investment
        )
        
        # Apply rebalancing if improvement is significant
        if optimal_weights is not None:
            optimal_expected_hv = self._compute_expected_hypervolume_contribution(
                predicted_rois, predicted_risks, optimal_weights
            )
            
            if optimal_expected_hv > current_expected_hv * 1.05:  # 5% improvement threshold
                portfolio.investment = optimal_weights
                portfolio.cardinality = np.sum(optimal_weights > 0.01)
    
    def _compute_expected_hypervolume_contribution(self, predicted_rois: np.ndarray, 
                                                 predicted_risks: np.ndarray, 
                                                 weights: np.ndarray) -> float:
        """Compute expected hypervolume contribution for given weights."""
        # This is a simplified version - in practice, you'd need to compute
        # hypervolume contribution considering the full Pareto front
        expected_roi = np.mean(predicted_rois)
        expected_risk = np.mean(predicted_risks)
        
        # Simple hypervolume approximation
        return expected_roi * (1.0 - expected_risk)
    
    def _optimize_weights_for_expected_hypervolume(self, predicted_rois: np.ndarray, 
                                                 predicted_risks: np.ndarray, 
                                                 current_weights: np.ndarray) -> Optional[np.ndarray]:
        """Optimize weights for maximal expected hypervolume."""
        # This is a placeholder - in practice, you'd implement a proper optimization
        # algorithm to find weights that maximize expected hypervolume
        
        # For now, return None (no rebalancing)
        return None
    
    def _store_stochastic_pareto_frontier(self, population: List[Solution], current_time: int):
        """
        Store stochastic Pareto frontier for visualization.
        
        Args:
            population: Current population
            current_time: Current time step
        """
        # Extract Pareto front
        pareto_front = [s for s in population if s.pareto_rank == 0]
        
        # Create stochastic Pareto frontier representation
        stochastic_frontier = []
        for solution in pareto_front:
            if hasattr(solution, 'P') and hasattr(solution.P, 'kalman_state'):
                kalman_state = solution.P.kalman_state
                stochastic_frontier.append({
                    'roi': solution.P.ROI,
                    'risk': solution.P.risk,
                    'roi_prediction': solution.P.ROI_prediction,
                    'risk_prediction': solution.P.risk_prediction,
                    'roi_variance': kalman_state.P[0, 0],
                    'risk_variance': kalman_state.P[1, 1],
                    'covariance': kalman_state.P[0, 1],
                    'alpha': getattr(solution, 'alpha', 0.0),
                    'prediction_error': getattr(solution, 'prediction_error', 0.0),
                    'weights': solution.P.investment.tolist()
                })
        
        # Store with timestamp
        self.stochastic_pareto_frontiers.append({
            'timestamp': datetime.now().isoformat(),
            'current_time': current_time,
            'frontier': stochastic_frontier
        })
        
        # Create anticipative Pareto frontier
        anticipative_frontier = []
        for solution in pareto_front:
            if hasattr(solution, 'P') and hasattr(solution.P, 'kalman_state'):
                kalman_state = solution.P.kalman_state
                anticipative_frontier.append({
                    'roi': kalman_state.x[0],
                    'risk': kalman_state.x[1],
                    'roi_variance': kalman_state.P[0, 0],
                    'risk_variance': kalman_state.P[1, 1],
                    'covariance': kalman_state.P[0, 1],
                    'weights': solution.P.investment.tolist()
                })
        
        self.anticipative_pareto_frontiers.append({
            'timestamp': datetime.now().isoformat(),
            'current_time': current_time,
            'frontier': anticipative_frontier
        })
    
    def apply_dirichlet_map_filtering(self, solution: Solution):
        """Apply Dirichlet MAP filtering to portfolio weights."""
        portfolio = solution.P
        weights = portfolio.investment
        
        # Dirichlet MAP estimation
        alpha_prior = np.ones_like(weights)
        alpha_likelihood = weights * 10
        alpha_posterior = alpha_prior + alpha_likelihood
        
        # MAP estimate
        K = len(weights)
        sum_alpha = np.sum(alpha_posterior)
        
        if sum_alpha > K:
            map_weights = (alpha_posterior - 1) / (sum_alpha - K)
        else:
            map_weights = np.ones_like(weights) / K
        
        # Ensure weights sum to 1 and are non-negative
        map_weights = np.maximum(map_weights, 0.0)
        map_weights = map_weights / np.sum(map_weights)
        
        # Update portfolio weights
        portfolio.investment = map_weights
        portfolio.cardinality = np.sum(map_weights > 0.01)
    
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
            'learning_trend': self._compute_learning_trend(),
            'adaptive_learning_enabled': self.adaptive_learning,
            'prediction_horizon': self.prediction_horizon
        }
        
        return metrics
    
    def get_stochastic_pareto_frontiers(self) -> List[Dict[str, Any]]:
        """Get stored stochastic Pareto frontiers for visualization."""
        return self.stochastic_pareto_frontiers
    
    def get_anticipative_pareto_frontiers(self) -> List[Dict[str, Any]]:
        """Get stored anticipative Pareto frontiers for visualization."""
        return self.anticipative_pareto_frontiers
    
    def _compute_learning_trend(self) -> float:
        """Compute learning trend based on prediction error."""
        if len(self.prediction_errors) < 10:
            return 0.0
        
        recent_errors = self.prediction_errors[-10:]
        x = np.arange(len(recent_errors))
        slope = np.polyfit(x, recent_errors, 1)[0]
        
        return -slope
    
    def reset(self):
        """Reset learning system."""
        self.learning_history = []
        self.prediction_errors = []
        self.state_qualities = []
        self.adaptive_learning_rates = []
        self.anticipative_distributions = []
        self.stochastic_pareto_frontiers = []
        self.anticipative_pareto_frontiers = [] 