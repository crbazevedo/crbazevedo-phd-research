"""
Kalman Filter implementation for portfolio optimization.

This module implements the Kalman filter for state estimation of portfolio
ROI and risk, including prediction and update steps.
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass


@dataclass
class KalmanParams:
    """
    Kalman filter parameters equivalent to C++ Kalman_params struct.
    
    Attributes:
        F: State transition matrix (static)
        H: Measurement matrix (static) 
        R: Measurement noise covariance matrix (static)
        x: Current state vector
        x_next: Next state vector
        u: Control input vector
        P: Current error covariance matrix
        P_next: Next error covariance matrix
    """
    # Static matrices (shared across all instances)
    F: Optional[np.ndarray] = None  # State transition matrix
    H: Optional[np.ndarray] = None  # Measurement matrix
    R: Optional[np.ndarray] = None  # Measurement noise covariance
    
    # Instance variables
    x: Optional[np.ndarray] = None      # Current state vector
    x_next: Optional[np.ndarray] = None # Next state vector
    u: Optional[np.ndarray] = None      # Control input vector
    P: Optional[np.ndarray] = None      # Current error covariance matrix
    P_next: Optional[np.ndarray] = None # Next error covariance matrix
    
    def __post_init__(self):
        """Initialize with default values if not provided."""
        if self.x is None:
            self.x = np.zeros(4)  # [ROI, ROI_velocity, risk, risk_velocity]
        if self.x_next is None:
            self.x_next = np.zeros(4)
        if self.u is None:
            self.u = np.zeros(4)
        if self.P is None:
            self.P = np.eye(4) * 0.1  # Initial covariance
        if self.P_next is None:
            self.P_next = np.eye(4) * 0.1


def kalman_prediction(params: KalmanParams) -> None:
    """
    Perform Kalman filter prediction step.
    
    Args:
        params: Kalman filter parameters
    """
    # State prediction: x_next = F * x + u
    params.x_next = params.F @ params.x + params.u
    
    # Covariance prediction: P_next = F * P * F^T
    params.P_next = params.F @ params.P @ params.F.T


def kalman_update(params: KalmanParams, measurement: np.ndarray) -> None:
    """
    Perform Kalman filter update step.
    
    Args:
        params: Kalman filter parameters
        measurement: Measurement vector [ROI, risk]
    """
    # Create measurement vector Z
    Z = np.array([measurement[0], measurement[1]])  # [ROI, risk]
    
    # Innovation: y = Z - H * x_next
    y = Z - params.H @ params.x_next
    
    # Innovation covariance: S = H * P_next * H^T + R
    S = params.H @ params.P_next @ params.H.T + params.R
    
    # Kalman gain: K = P_next * H^T * S^(-1)
    K = params.P_next @ params.H.T @ np.linalg.inv(S)
    
    # State update: x = x_next + K * y
    params.x = params.x_next + K @ y
    
    # Covariance update: P = (I - K * H) * P_next
    I = np.eye(params.F.shape[0])
    params.P = (I - K @ params.H) @ params.P_next


def kalman_filter(params: KalmanParams, measurement: np.ndarray) -> None:
    """
    Perform complete Kalman filter step (prediction + update).
    
    Args:
        params: Kalman filter parameters
        measurement: Measurement vector [ROI, risk]
    """
    kalman_prediction(params)
    kalman_update(params, measurement)


def initialize_kalman_matrices() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Initialize Kalman filter matrices based on C++ implementation.
    
    Returns:
        Tuple of (F, H, R) matrices
    """
    # State transition matrix F (4x4)
    # [ROI, ROI_velocity, risk, risk_velocity]
    F = np.array([
        [1.0, 1.0, 0.0, 0.0],  # ROI_next = ROI + ROI_velocity
        [0.0, 1.0, 0.0, 0.0],  # ROI_velocity_next = ROI_velocity
        [0.0, 0.0, 1.0, 1.0],  # risk_next = risk + risk_velocity
        [0.0, 0.0, 0.0, 1.0]   # risk_velocity_next = risk_velocity
    ])
    
    # Measurement matrix H (2x4)
    # We only observe ROI and risk, not their velocities
    H = np.array([
        [1.0, 0.0, 0.0, 0.0],  # Observe ROI
        [0.0, 0.0, 1.0, 0.0]   # Observe risk
    ])
    
    # Measurement noise covariance R (2x2)
    # Initial values, will be updated based on data
    R = np.array([
        [0.01, 0.005],  # ROI measurement noise
        [0.005, 0.01]   # Risk measurement noise
    ])
    
    return F, H, R


def create_kalman_params(initial_roi: float = 0.0, initial_risk: float = 0.0) -> KalmanParams:
    """
    Create and initialize Kalman filter parameters.
    
    Args:
        initial_roi: Initial ROI value
        initial_risk: Initial risk value
        
    Returns:
        Initialized KalmanParams object
    """
    # Initialize matrices
    F, H, R = initialize_kalman_matrices()
    
    # Create instance with matrices
    params = KalmanParams(F=F, H=H, R=R)
    
    # Initialize state vector [ROI, ROI_velocity, risk, risk_velocity]
    params.x = np.array([initial_roi, 0.0, initial_risk, 0.0])
    params.x_next = params.x.copy()
    
    # Initialize control input (zero for now)
    params.u = np.zeros(4)
    
    # Initialize covariance matrices
    params.P = np.eye(4) * 0.1
    params.P_next = params.P.copy()
    
    return params


def update_measurement_noise(params: KalmanParams, roi_variance: float, risk_variance: float, 
                           covariance: float = 0.0) -> None:
    """
    Update measurement noise covariance matrix.
    
    Args:
        params: Kalman filter parameters
        roi_variance: ROI measurement variance
        risk_variance: Risk measurement variance
        covariance: ROI-risk measurement covariance
    """
    params.R = np.array([
        [roi_variance, covariance],
        [covariance, risk_variance]
    ])


def get_portfolio_state(params: KalmanParams) -> tuple[float, float]:
    """
    Extract current portfolio ROI and risk from Kalman state.
    
    Args:
        params: Kalman filter parameters
        
    Returns:
        Tuple of (ROI, risk)
    """
    return params.x[0], params.x[2]  # ROI, risk


def get_portfolio_prediction(params: KalmanParams) -> tuple[float, float]:
    """
    Extract predicted portfolio ROI and risk from Kalman state.
    
    Args:
        params: Kalman filter parameters
        
    Returns:
        Tuple of (predicted_ROI, predicted_risk)
    """
    return params.x_next[0], params.x_next[2]  # ROI_prediction, risk_prediction


def get_error_covariance(params: KalmanParams) -> np.ndarray:
    """
    Get current error covariance matrix.
    
    Args:
        params: Kalman filter parameters
        
    Returns:
        Error covariance matrix
    """
    return params.P.copy()


def get_prediction_error_covariance(params: KalmanParams) -> np.ndarray:
    """
    Get prediction error covariance matrix.
    
    Args:
        params: Kalman filter parameters
        
    Returns:
        Prediction error covariance matrix
    """
    return params.P_next.copy() 