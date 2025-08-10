"""
Metrics Collector Module

Provides comprehensive metrics collection for portfolio optimization experiments
including portfolio performance, optimization quality, and learning effectiveness.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class MetricsCollector:
    """Comprehensive metrics collector for portfolio optimization experiments."""
    
    def __init__(self, experiment_id: str, metrics_dir: str = "experiments/metrics"):
        """
        Initialize the metrics collector.
        
        Args:
            experiment_id: Unique identifier for the experiment
            metrics_dir: Directory to store metrics
        """
        self.experiment_id = experiment_id
        self.metrics_dir = Path(metrics_dir) / experiment_id
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metrics storage
        self.portfolio_metrics = {}
        self.optimization_metrics = {}
        self.learning_metrics = {}
        self.computational_metrics = {}
        
        # Time series data
        self.portfolio_returns = []
        self.portfolio_values = []
        self.hypervolume_history = []
        self.pareto_front_history = []
        self.learning_progress = []
        
        # Performance tracking
        self.start_time = datetime.now()
        self.metrics_history = []
    
    def collect_portfolio_metrics(self, portfolio_weights: Dict[str, float], 
                                asset_returns: pd.DataFrame, 
                                portfolio_value: float = 1.0) -> Dict[str, float]:
        """
        Collect comprehensive portfolio performance metrics.
        
        Args:
            portfolio_weights: Portfolio weight allocation
            asset_returns: Asset return data
            portfolio_value: Current portfolio value
            
        Returns:
            Dictionary of portfolio metrics
        """
        # Calculate portfolio returns
        portfolio_return = self._calculate_portfolio_return(portfolio_weights, asset_returns)
        
        # Store time series data
        self.portfolio_returns.append(portfolio_return)
        self.portfolio_values.append(portfolio_value)
        
        # Calculate cumulative metrics
        cumulative_return = self._calculate_cumulative_return()
        volatility = self._calculate_volatility()
        sharpe_ratio = self._calculate_sharpe_ratio()
        max_drawdown = self._calculate_max_drawdown()
        calmar_ratio = self._calculate_calmar_ratio()
        
        # Calculate additional metrics
        var_95 = self._calculate_value_at_risk(0.05)
        cvar_95 = self._calculate_conditional_var(0.05)
        sortino_ratio = self._calculate_sortino_ratio()
        information_ratio = self._calculate_information_ratio()
        
        # Portfolio composition metrics
        concentration = self._calculate_concentration(portfolio_weights)
        diversification = self._calculate_diversification(portfolio_weights)
        
        metrics = {
            'portfolio_return': portfolio_return,
            'cumulative_return': cumulative_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'sortino_ratio': sortino_ratio,
            'information_ratio': information_ratio,
            'concentration': concentration,
            'diversification': diversification,
            'portfolio_value': portfolio_value,
            'num_assets': len(portfolio_weights)
        }
        
        self.portfolio_metrics.update(metrics)
        return metrics
    
    def collect_optimization_metrics(self, population: List, generation: int, 
                                   pareto_front: List, hypervolume: float) -> Dict[str, Any]:
        """
        Collect optimization quality metrics.
        
        Args:
            population: Current population of solutions
            generation: Current generation number
            pareto_front: Current Pareto front
            hypervolume: Current hypervolume
            
        Returns:
            Dictionary of optimization metrics
        """
        # Store historical data
        self.hypervolume_history.append(hypervolume)
        self.pareto_front_history.append(pareto_front)
        
        # Calculate convergence metrics
        convergence_metric = self._calculate_convergence_metric()
        diversity_metric = self._calculate_diversity_metric(population)
        spread_metric = self._calculate_spread_metric(pareto_front)
        
        # Calculate solution quality metrics
        solution_quality = self._calculate_solution_quality(population)
        pareto_efficiency = self._calculate_pareto_efficiency(pareto_front)
        
        metrics = {
            'generation': generation,
            'hypervolume': hypervolume,
            'convergence_metric': convergence_metric,
            'diversity_metric': diversity_metric,
            'spread_metric': spread_metric,
            'solution_quality': solution_quality,
            'pareto_efficiency': pareto_efficiency,
            'population_size': len(population),
            'pareto_front_size': len(pareto_front)
        }
        
        self.optimization_metrics.update(metrics)
        return metrics
    
    def collect_learning_metrics(self, prediction_error: float, state_quality: float,
                               learning_progress: float, event_type: str) -> Dict[str, Any]:
        """
        Collect anticipatory learning metrics.
        
        Args:
            prediction_error: Current prediction error
            state_quality: Quality of state observation
            learning_progress: Learning progress indicator
            event_type: Type of learning event
            
        Returns:
            Dictionary of learning metrics
        """
        # Store learning progress
        self.learning_progress.append({
            'prediction_error': prediction_error,
            'state_quality': state_quality,
            'learning_progress': learning_progress,
            'event_type': event_type,
            'timestamp': datetime.now().isoformat()
        })
        
        # Calculate learning effectiveness metrics
        avg_prediction_error = np.mean([p['prediction_error'] for p in self.learning_progress])
        avg_state_quality = np.mean([p['state_quality'] for p in self.learning_progress])
        learning_trend = self._calculate_learning_trend()
        
        metrics = {
            'prediction_error': prediction_error,
            'state_quality': state_quality,
            'learning_progress': learning_progress,
            'avg_prediction_error': avg_prediction_error,
            'avg_state_quality': avg_state_quality,
            'learning_trend': learning_trend,
            'event_type': event_type,
            'total_learning_events': len(self.learning_progress)
        }
        
        self.learning_metrics.update(metrics)
        return metrics
    
    def collect_computational_metrics(self, execution_time: float, memory_usage: float,
                                    function_evaluations: int) -> Dict[str, Any]:
        """
        Collect computational performance metrics.
        
        Args:
            execution_time: Algorithm execution time
            memory_usage: Peak memory usage
            function_evaluations: Number of function evaluations
            
        Returns:
            Dictionary of computational metrics
        """
        total_time = (datetime.now() - self.start_time).total_seconds()
        
        metrics = {
            'execution_time': execution_time,
            'total_time': total_time,
            'memory_usage_mb': memory_usage,
            'function_evaluations': function_evaluations,
            'evaluations_per_second': function_evaluations / execution_time if execution_time > 0 else 0
        }
        
        self.computational_metrics.update(metrics)
        return metrics
    
    def _calculate_portfolio_return(self, weights: Dict[str, float], 
                                  asset_returns: pd.DataFrame) -> float:
        """Calculate portfolio return."""
        if asset_returns.empty:
            return 0.0
        
        # Get latest returns
        latest_returns = asset_returns.iloc[-1] if len(asset_returns) > 0 else pd.Series(0, index=weights.keys())
        
        # Calculate weighted return
        portfolio_return = sum(weights[asset] * latest_returns.get(asset, 0) 
                             for asset in weights.keys())
        
        return portfolio_return
    
    def _calculate_cumulative_return(self) -> float:
        """Calculate cumulative return from portfolio returns."""
        if not self.portfolio_returns:
            return 0.0
        
        cumulative = 1.0
        for ret in self.portfolio_returns:
            cumulative *= (1 + ret)
        
        return cumulative - 1.0
    
    def _calculate_volatility(self, window: int = 30) -> float:
        """Calculate portfolio volatility."""
        if len(self.portfolio_returns) < window:
            return np.std(self.portfolio_returns) if self.portfolio_returns else 0.0
        
        recent_returns = self.portfolio_returns[-window:]
        return np.std(recent_returns)
    
    def _calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        if not self.portfolio_returns:
            return 0.0
        
        returns = np.array(self.portfolio_returns)
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        
        if np.std(excess_returns) == 0:
            return 0.0
        
        return np.mean(excess_returns) / np.std(excess_returns)
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown."""
        if not self.portfolio_values:
            return 0.0
        
        values = np.array(self.portfolio_values)
        peak = np.maximum.accumulate(values)
        drawdown = (values - peak) / peak
        
        return np.min(drawdown)
    
    def _calculate_calmar_ratio(self) -> float:
        """Calculate Calmar ratio."""
        cumulative_return = self._calculate_cumulative_return()
        max_drawdown = abs(self._calculate_max_drawdown())
        
        if max_drawdown == 0:
            return 0.0
        
        return cumulative_return / max_drawdown
    
    def _calculate_value_at_risk(self, confidence_level: float) -> float:
        """Calculate Value at Risk."""
        if not self.portfolio_returns:
            return 0.0
        
        return np.percentile(self.portfolio_returns, confidence_level * 100)
    
    def _calculate_conditional_var(self, confidence_level: float) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)."""
        if not self.portfolio_returns:
            return 0.0
        
        var = self._calculate_value_at_risk(confidence_level)
        returns = np.array(self.portfolio_returns)
        tail_returns = returns[returns <= var]
        
        return np.mean(tail_returns) if len(tail_returns) > 0 else var
    
    def _calculate_sortino_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio."""
        if not self.portfolio_returns:
            return 0.0
        
        returns = np.array(self.portfolio_returns)
        excess_returns = returns - risk_free_rate / 252
        
        # Calculate downside deviation
        downside_returns = excess_returns[excess_returns < 0]
        downside_deviation = np.std(downside_returns) if len(downside_returns) > 0 else 0.0
        
        if downside_deviation == 0:
            return 0.0
        
        return np.mean(excess_returns) / downside_deviation
    
    def _calculate_information_ratio(self, benchmark_return: float = 0.0) -> float:
        """Calculate Information ratio."""
        if not self.portfolio_returns:
            return 0.0
        
        returns = np.array(self.portfolio_returns)
        active_returns = returns - benchmark_return
        tracking_error = np.std(active_returns)
        
        if tracking_error == 0:
            return 0.0
        
        return np.mean(active_returns) / tracking_error
    
    def _calculate_concentration(self, weights: Dict[str, float]) -> float:
        """Calculate portfolio concentration (Herfindahl index)."""
        weight_values = list(weights.values())
        return sum(w**2 for w in weight_values)
    
    def _calculate_diversification(self, weights: Dict[str, float]) -> float:
        """Calculate portfolio diversification."""
        concentration = self._calculate_concentration(weights)
        return 1.0 - concentration
    
    def _calculate_convergence_metric(self) -> float:
        """Calculate convergence metric based on hypervolume stability."""
        if len(self.hypervolume_history) < 10:
            return 0.0
        
        recent_hypervolume = self.hypervolume_history[-10:]
        stability = 1.0 - np.std(recent_hypervolume) / np.mean(recent_hypervolume)
        
        return max(0.0, stability)
    
    def _calculate_diversity_metric(self, population: List) -> float:
        """Calculate population diversity metric."""
        if not population:
            return 0.0
        
        # Calculate average pairwise distance
        distances = []
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                if hasattr(population[i], 'P') and hasattr(population[j], 'P'):
                    dist = np.linalg.norm(population[i].P.investment - population[j].P.investment)
                    distances.append(dist)
        
        return np.mean(distances) if distances else 0.0
    
    def _calculate_spread_metric(self, pareto_front: List) -> float:
        """Calculate Pareto front spread metric."""
        if len(pareto_front) < 2:
            return 0.0
        
        # Calculate spread based on objective values
        objectives = []
        for solution in pareto_front:
            if hasattr(solution, 'objectives'):
                objectives.append(solution.objectives)
        
        if not objectives:
            return 0.0
        
        objectives = np.array(objectives)
        spread = np.std(objectives, axis=0)
        
        return np.mean(spread)
    
    def _calculate_solution_quality(self, population: List) -> float:
        """Calculate average solution quality."""
        if not population:
            return 0.0
        
        qualities = []
        for solution in population:
            if hasattr(solution, 'fitness'):
                qualities.append(solution.fitness)
        
        return np.mean(qualities) if qualities else 0.0
    
    def _calculate_pareto_efficiency(self, pareto_front: List) -> float:
        """Calculate Pareto front efficiency."""
        if not pareto_front:
            return 0.0
        
        # Calculate average objective values
        objectives = []
        for solution in pareto_front:
            if hasattr(solution, 'objectives'):
                objectives.append(solution.objectives)
        
        if not objectives:
            return 0.0
        
        objectives = np.array(objectives)
        efficiency = np.mean(objectives)
        
        return efficiency
    
    def _calculate_learning_trend(self) -> float:
        """Calculate learning trend based on prediction error."""
        if len(self.learning_progress) < 10:
            return 0.0
        
        recent_errors = [p['prediction_error'] for p in self.learning_progress[-10:]]
        
        # Calculate trend (negative slope means improvement)
        x = np.arange(len(recent_errors))
        slope = np.polyfit(x, recent_errors, 1)[0]
        
        return -slope  # Return positive for improvement
    
    def save_metrics(self):
        """Save all collected metrics to files."""
        # Save portfolio metrics
        portfolio_file = self.metrics_dir / "portfolio_metrics.json"
        with open(portfolio_file, 'w') as f:
            json.dump(self.portfolio_metrics, f, indent=2)
        
        # Save optimization metrics
        optimization_file = self.metrics_dir / "optimization_metrics.json"
        with open(optimization_file, 'w') as f:
            json.dump(self.optimization_metrics, f, indent=2)
        
        # Save learning metrics
        learning_file = self.metrics_dir / "learning_metrics.json"
        with open(learning_file, 'w') as f:
            json.dump(self.learning_metrics, f, indent=2)
        
        # Save computational metrics
        computational_file = self.metrics_dir / "computational_metrics.json"
        with open(computational_file, 'w') as f:
            json.dump(self.computational_metrics, f, indent=2)
        
        # Save time series data
        time_series_file = self.metrics_dir / "time_series_data.json"
        time_series_data = {
            'portfolio_returns': self.portfolio_returns,
            'portfolio_values': self.portfolio_values,
            'hypervolume_history': self.hypervolume_history,
            'learning_progress': self.learning_progress
        }
        with open(time_series_file, 'w') as f:
            json.dump(time_series_data, f, indent=2, default=str)
    
    def get_summary_metrics(self) -> Dict[str, Any]:
        """Get summary of all collected metrics."""
        return {
            'portfolio': self.portfolio_metrics,
            'optimization': self.optimization_metrics,
            'learning': self.learning_metrics,
            'computational': self.computational_metrics
        }
    
    def reset(self):
        """Reset all metrics."""
        self.portfolio_metrics = {}
        self.optimization_metrics = {}
        self.learning_metrics = {}
        self.computational_metrics = {}
        self.portfolio_returns = []
        self.portfolio_values = []
        self.hypervolume_history = []
        self.pareto_front_history = []
        self.learning_progress = []
        self.start_time = datetime.now() 