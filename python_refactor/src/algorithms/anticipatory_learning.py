"""
Anticipatory Learning implementation for portfolio optimization.

This module implements the anticipatory learning system that uses
Kalman filters and stochastic dominance to predict future portfolio
performance and update portfolio estimates accordingly.
"""

import numpy as np
from typing import List, Optional
from .kalman_filter import KalmanParams, kalman_filter, kalman_prediction
from .statistics import (
    multi_norm, normal_cdf, linear_entropy,
    compute_stochastic_parameters, compute_non_dominance_probability,
    compute_confidence_alpha
)
from ..portfolio.portfolio import Portfolio
from .solution import Solution


def observe_state(portfolio: Portfolio, num_samples: int, current_time: int) -> None:
    """
    Observe portfolio state via Monte Carlo simulation.
    
    Args:
        portfolio: Portfolio object
        num_samples: Number of Monte Carlo samples
        current_time: Current time step
    """
    if portfolio.kalman_state is None:
        return
    
    # Extract next-state returns data
    if Portfolio.complete_returns_data is None:
        return
    
    window_size = Portfolio.window_size
    index = (current_time + 1) * (window_size - 1)
    
    # Get returns data for next time step
    if index + window_size > Portfolio.complete_returns_data.shape[0]:
        return
    
    returns_data = Portfolio.complete_returns_data[index:index + window_size, :]
    
    # Compute observed ROI and risk
    if Portfolio.robustness:
        mean = Portfolio.estimate_assets_median_ROI(returns_data)
        portfolio.error_covar = Portfolio.estimate_robust_covariance(mean, returns_data)
    else:
        mean = Portfolio.estimate_assets_mean_ROI(returns_data)
        portfolio.error_covar = Portfolio.estimate_covariance(mean, returns_data)
    
    portfolio.ROI_observed = Portfolio.compute_ROI(portfolio, mean)
    portfolio.risk_observed = Portfolio.compute_risk(portfolio, portfolio.error_covar)
    
    # Update Kalman filter with observed measurements
    measurement = np.array([portfolio.ROI_observed, portfolio.risk_observed])
    kalman_filter(portfolio.kalman_state, measurement)
    
    # Update portfolio state from Kalman filter
    portfolio.ROI = portfolio.kalman_state.x[0]
    portfolio.risk = portfolio.kalman_state.x[2]
    portfolio.error_covar = portfolio.kalman_state.P.copy()


def prediction_error(portfolio: Portfolio, current_time: int) -> float:
    """
    Compute prediction error for portfolio.
    
    Args:
        portfolio: Portfolio object
        current_time: Current time step
        
    Returns:
        Mean square prediction error
    """
    # Extract next-state returns data
    if Portfolio.complete_returns_data is None:
        return 0.0
    
    window_size = Portfolio.window_size
    index = (current_time + 1) * (window_size - 1)
    
    if index + window_size > Portfolio.complete_returns_data.shape[0]:
        return 0.0
    
    returns_data = Portfolio.complete_returns_data[index:index + window_size, :]
    
    # Compute observed values
    if Portfolio.robustness:
        mean = Portfolio.estimate_assets_median_ROI(returns_data)
        portfolio.error_covar = Portfolio.estimate_robust_covariance(mean, returns_data)
    else:
        mean = Portfolio.estimate_assets_mean_ROI(returns_data)
        portfolio.error_covar = Portfolio.estimate_covariance(mean, returns_data)
    
    roi_observed = Portfolio.compute_ROI(portfolio, mean)
    risk_observed = Portfolio.compute_risk(portfolio, portfolio.error_covar)
    
    # Compute prediction errors
    error_roi = (portfolio.ROI - roi_observed) ** 2
    error_risk = (portfolio.risk - risk_observed) ** 2
    
    return error_roi + error_risk


def anticipatory_learning_single(solution: Solution, current_time: int) -> None:
    """
    Apply anticipatory learning to a single solution.
    
    Args:
        solution: Solution object
        current_time: Current time step
    """
    portfolio = solution.P
    
    # Check if Kalman filter is available
    if portfolio.kalman_state is None:
        return
    
    # Observe state via Monte Carlo simulation
    observe_state(portfolio, 10, current_time)
    
    # Extract 2x2 covariance matrix for ROI and risk (indices 0 and 2)
    # From the 4x4 Kalman covariance matrix, we need the ROI and risk components
    kalman_covar = portfolio.error_covar_prediction + portfolio.error_covar
    covar = np.array([
        [kalman_covar[0, 0], kalman_covar[0, 2]],  # ROI-ROI, ROI-risk
        [kalman_covar[2, 0], kalman_covar[2, 2]]   # risk-ROI, risk-risk
    ])
    
    # Compute delta vectors for dominance scenarios
    delta1 = np.array([
        portfolio.ROI - portfolio.ROI_prediction,
        portfolio.risk - portfolio.risk_prediction
    ])
    
    delta2 = np.array([
        portfolio.ROI_prediction - portfolio.ROI,
        portfolio.risk_prediction - portfolio.risk
    ])
    
    # Compute non-dominance probability
    nd_probability = compute_non_dominance_probability(delta1, delta2, covar)
    
    # Compute confidence parameter alpha
    alpha = compute_confidence_alpha(nd_probability)
    
    # Update Kalman state based on anticipatory knowledge
    portfolio.kalman_state.x = (portfolio.kalman_state.x + 
                               alpha * (portfolio.kalman_state.x_next - portfolio.kalman_state.x))
    portfolio.kalman_state.P = (portfolio.kalman_state.P + 
                               alpha * (portfolio.kalman_state.P_next - portfolio.kalman_state.P))
    
    # Update portfolio ROI and risk from Kalman state
    portfolio.ROI = portfolio.kalman_state.x[0]
    portfolio.risk = portfolio.kalman_state.x[2]
    
    # Update robust/non-robust values
    if Portfolio.robustness:
        portfolio.robust_ROI = portfolio.ROI
        portfolio.robust_risk = portfolio.risk
    else:
        portfolio.non_robust_ROI = portfolio.ROI
        portfolio.non_robust_risk = portfolio.risk
    
    # Compute prediction error
    solution.prediction_error = prediction_error(portfolio, current_time)
    
    # Mark solution as having anticipatory learning applied
    solution.anticipation = True
    solution.alpha = alpha


def anticipatory_learning_population(population: List[Solution], current_time: int) -> None:
    """
    Apply anticipatory learning to entire population.
    
    Args:
        population: List of solutions
        current_time: Current time step
    """
    for solution in population:
        if not solution.anticipation:
            anticipatory_learning_single(solution, current_time)


def kalman_filter_prediction(population: List[Solution]) -> None:
    """
    Apply Kalman filter prediction to entire population.
    
    Args:
        population: List of solutions
    """
    for solution in population:
        if solution.P.kalman_state is not None:
            kalman_prediction(solution.P.kalman_state)
            
            # Update prediction values
            solution.P.ROI_prediction = solution.P.kalman_state.x_next[0]
            solution.P.risk_prediction = solution.P.kalman_state.x_next[2]
            solution.P.error_covar_prediction = solution.P.kalman_state.P_next.copy()


def initialize_portfolio_kalman(portfolio: Portfolio, initial_roi: float = 0.0, 
                              initial_risk: float = 0.0) -> None:
    """
    Initialize Kalman filter for a portfolio.
    
    Args:
        portfolio: Portfolio object
        initial_roi: Initial ROI value
        initial_risk: Initial risk value
    """
    from .kalman_filter import create_kalman_params
    
    # Create Kalman parameters
    kalman_params = create_kalman_params(initial_roi, initial_risk)
    
    # Initialize portfolio Kalman state
    portfolio.kalman_state = kalman_params
    portfolio.error_covar = kalman_params.P.copy()
    portfolio.error_covar_prediction = kalman_params.P_next.copy()
    
    # Set initial predictions
    portfolio.ROI_prediction = initial_roi
    portfolio.risk_prediction = initial_risk


def update_kalman_measurement_noise(portfolio: Portfolio, roi_variance: float, 
                                  risk_variance: float, covariance: float = 0.0) -> None:
    """
    Update Kalman filter measurement noise for a portfolio.
    
    Args:
        portfolio: Portfolio object
        roi_variance: ROI measurement variance
        risk_variance: Risk measurement variance
        covariance: ROI-risk measurement covariance
    """
    if portfolio.kalman_state is not None:
        from .kalman_filter import update_measurement_noise
        update_measurement_noise(portfolio.kalman_state, roi_variance, risk_variance, covariance)


def compute_stochastic_delta_s_contribution(solution: Solution, 
                                          reference_point: tuple[float, float]) -> float:
    """
    Compute stochastic Delta-S contribution for SMS-EMOA.
    
    Args:
        solution: Solution object
        reference_point: Reference point (rx, ry)
        
    Returns:
        Stochastic Delta-S contribution
    """
    if solution.P.kalman_state is None:
        return 0.0
    
    # Get stochastic parameters
    params = compute_stochastic_parameters(
        solution.P.ROI, 
        solution.P.risk, 
        solution.P.kalman_state.P
    )
    
    # Compute stochastic contribution based on uncertainty
    rx, ry = reference_point
    
    # Consider uncertainty in ROI and risk
    roi_uncertainty = np.sqrt(params['var_roi'])
    risk_uncertainty = np.sqrt(params['var_risk'])
    
    # Stochastic Delta-S contribution
    stochastic_contribution = (solution.P.ROI - rx) * (solution.P.risk - ry)
    uncertainty_factor = 1.0 / (1.0 + roi_uncertainty + risk_uncertainty)
    
    return stochastic_contribution * uncertainty_factor


def apply_anticipatory_learning_to_algorithm(population: List[Solution], 
                                           current_time: int,
                                           algorithm_type: str = 'both') -> None:
    """
    Apply anticipatory learning to optimization algorithm.
    
    Args:
        population: List of solutions
        current_time: Current time step
        algorithm_type: Type of algorithm ('nsga2', 'sms_emoa', 'both')
    """
    if current_time < 0:
        return  # No anticipatory learning for initial generation
    
    # Apply Kalman filter prediction
    kalman_filter_prediction(population)
    
    # Apply anticipatory learning
    anticipatory_learning_population(population, current_time)
    
    # Recompute efficiency metrics after learning
    for solution in population:
        if Portfolio.mean_ROI is not None and Portfolio.covariance is not None:
            if Portfolio.robustness:
                Portfolio.compute_robust_efficiency(solution.P)
            else:
                Portfolio.compute_efficiency(solution.P) 