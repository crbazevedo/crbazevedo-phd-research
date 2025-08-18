#!/usr/bin/env python3
"""
Uncertainty-Aware ASMS-EMOA Experiment

Integrates uncertainty quantification and decision space tracking with ASMS-EMOA
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import logging
from scipy.stats import multivariate_normal, norm
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class UncertaintyAwareASMSEMOAExperiment:
    """Uncertainty-Aware ASMS-EMOA Experiment"""
    
    def __init__(self, returns_data: pd.DataFrame):
        self.returns_data = returns_data
        from uncertainty_aware_asmsoa import (
            UncertaintyAwareHvDM, 
            BayesianUncertaintyQuantifier,
            AMFCTrajectoryTracker,
            BivariateGaussianUpdater
        )
        
        self.uncertainty_hv_dm = UncertaintyAwareHvDM()
        self.uncertainty_predictor = BayesianUncertaintyQuantifier(input_dim=20)
        self.amfc_tracker = AMFCTrajectoryTracker()
        self.bivariate_updater = BivariateGaussianUpdater()
        
    def run_uncertainty_aware_experiment(self, num_runs: int = 5):
        """Run uncertainty-aware ASMS-EMOA experiment"""
        
        # Experiment parameters
        historical_days = 120
        stride_days = 30
        k_values = [0, 1, 2, 3]
        h_values = [1, 2]
        
        # Calculate periods
        total_days = len(self.returns_data)
        n_periods = max(1, (total_days - historical_days) // stride_days)
        
        logger.info(f"Running Uncertainty-Aware experiment with {n_periods} periods")
        logger.info("Features: Uncertainty quantification, AMFC tracking, Bivariate Gaussian updates")
        
        all_results = {}
        
        for run in range(num_runs):
            logger.info(f"Starting Uncertainty-Aware run {run + 1}/{num_runs}")
            
            run_results = {}
            
            # Test Uncertainty-Aware ASMS-EMOA
            for k in k_values:
                for h in h_values:
                    key = f'Uncertainty_ASMS_EMOA_K{k}_h{h}_Hv-DM'
                    run_results[key] = self._run_uncertainty_aware_asmsoa_experiment(
                        k, h, historical_days, stride_days, n_periods
                    )
            
            # Test traditional benchmarks for comparison
            run_results['Uncertainty_Equal_Weighted'] = self._run_uncertainty_traditional_benchmark(
                historical_days, stride_days, n_periods
            )
            
            all_results[f'run_{run}'] = run_results
        
        return all_results
    
    def _run_uncertainty_aware_asmsoa_experiment(self, k, h, historical_days, stride_days, n_periods):
        """Run uncertainty-aware ASMS-EMOA experiment"""
        
        wealth_history = [100000.0]
        roi_history = []
        anticipative_rates = []
        expected_hv_values = []
        prediction_accuracy = []
        uncertainty_metrics = []
        amfc_trajectory_lengths = []
        
        current_wealth = 100000.0
        
        for period in range(n_periods):
            # Data windows
            start_idx = period * stride_days
            end_idx = start_idx + historical_days
            future_start = end_idx
            future_end = min(end_idx + 60, len(self.returns_data))
            
            historical_data = self.returns_data.iloc[start_idx:end_idx]
            future_data = self.returns_data.iloc[future_start:future_end]
            
            # Detect market regime
            market_regime = self._detect_market_regime(historical_data)
            
            # Run SMS-EMOA
            from real_data_experiment import SMSEMOA, Portfolio
            
            # Set Portfolio static variables
            Portfolio.mean_ROI = historical_data.mean().values
            Portfolio.covariance = historical_data.cov().values
            Portfolio.median_ROI = historical_data.median().values
            Portfolio.robust_covariance = historical_data.cov().values
            
            # Enhanced algorithm parameters
            algorithm_params = {
                'population_size': 120,
                'generations': 45,
                'reference_point_1': -0.2,
                'reference_point_2': 0.3
            }
            
            sms_emoa = SMSEMOA(**algorithm_params)
            data_dict = {
                'returns': historical_data.values,
                'num_assets': len(historical_data.columns),
                'anticipation_horizon': k
            }
            
            pareto_frontier = sms_emoa.run(data_dict)
            
            if not pareto_frontier:
                continue
            
            # Uncertainty-aware portfolio selection
            selection_result = self.uncertainty_hv_dm.select_optimal_portfolio(
                pareto_frontier, historical_data, market_regime
            )
            
            selected_portfolio = selection_result['selected_portfolio']
            enhanced_hv_score = selection_result['enhanced_hv_score']
            uncertainty_predictions = selection_result['uncertainty_predictions']
            amfc_trajectory = selection_result['amfc_trajectory']
            
            # Calculate performance
            portfolio_weights = selected_portfolio.P.investment
            
            if len(future_data) > 0:
                period_returns = future_data.values @ portfolio_weights
                period_returns = np.clip(period_returns, -0.20, 0.20)
                period_roi = np.mean(period_returns)
            else:
                period_roi = 0.0
            
            # Calculate uncertainty metrics
            uncertainty_metric = self._calculate_uncertainty_metric(
                uncertainty_predictions, period_roi, selected_portfolio.P.risk
            )
            
            # Enhanced anticipative rate calculation
            anticipative_rate = self._calculate_enhanced_anticipative_rate(
                k, h, uncertainty_predictions, amfc_trajectory
            )
            
            # Enhanced expected hypervolume
            expected_hv = enhanced_hv_score
            
            # Update wealth and history
            wealth_change = current_wealth * period_roi
            new_wealth = current_wealth + wealth_change
            
            wealth_history.append(new_wealth)
            roi_history.append(period_roi)
            anticipative_rates.append(anticipative_rate)
            expected_hv_values.append(expected_hv)
            prediction_accuracy.append(uncertainty_metric['prediction_accuracy'])
            uncertainty_metrics.append(uncertainty_metric)
            amfc_trajectory_lengths.append(len(amfc_trajectory.get('trajectory', [])))
            
            current_wealth = new_wealth
        
        return {
            'wealth_history': wealth_history,
            'roi_history': roi_history,
            'anticipative_rates': anticipative_rates,
            'expected_hv_values': expected_hv_values,
            'prediction_accuracy': prediction_accuracy,
            'uncertainty_metrics': uncertainty_metrics,
            'amfc_trajectory_lengths': amfc_trajectory_lengths,
            'final_wealth': wealth_history[-1] if wealth_history else 100000.0,
            'total_roi': (wealth_history[-1] - 100000.0) / 100000.0 if wealth_history else 0.0,
            'avg_roi_per_period': np.mean(roi_history) if roi_history else 0.0,
            'avg_anticipative_rate': np.mean(anticipative_rates) if anticipative_rates else 0.5,
            'avg_expected_hv': np.mean(expected_hv_values) if expected_hv_values else 0.0,
            'avg_prediction_accuracy': np.mean(prediction_accuracy) if prediction_accuracy else 0.5,
            'avg_uncertainty_coverage': np.mean([m['uncertainty_coverage'] for m in uncertainty_metrics]) if uncertainty_metrics else 0.5
        }
    
    def _detect_market_regime(self, historical_data: pd.DataFrame) -> str:
        """Detect market regime"""
        
        if len(historical_data) < 30:
            return "normal"
        
        returns = historical_data.pct_change().dropna()
        
        # Volatility regime
        recent_vol = returns.tail(20).std().mean()
        long_vol = returns.std().mean()
        vol_ratio = recent_vol / long_vol if long_vol > 0 else 1.0
        
        # Trend regime
        recent_mean = returns.tail(20).mean().mean()
        
        if vol_ratio > 1.5:
            vol_regime = "high_vol"
        elif vol_ratio < 0.7:
            vol_regime = "low_vol"
        else:
            vol_regime = "normal"
        
        return vol_regime
    
    def _calculate_uncertainty_metric(self, uncertainty_predictions: Dict, actual_roi: float, actual_risk: float) -> Dict:
        """Calculate uncertainty metrics"""
        
        predicted_mean = uncertainty_predictions['mean']
        predicted_uncertainty = uncertainty_predictions['uncertainty']
        
        # Check if actual values fall within uncertainty bounds
        roi_within_bounds = (
            predicted_mean[0] - 2*predicted_uncertainty[0] <= actual_roi <= 
            predicted_mean[0] + 2*predicted_uncertainty[0]
        )
        
        risk_within_bounds = (
            predicted_mean[1] - 2*predicted_uncertainty[1] <= actual_risk <= 
            predicted_mean[1] + 2*predicted_uncertainty[1]
        )
        
        # Calculate prediction accuracy
        roi_error = abs(predicted_mean[0] - actual_roi)
        risk_error = abs(predicted_mean[1] - actual_risk)
        
        prediction_accuracy = 1.0 / (1.0 + (roi_error + risk_error))
        
        # Uncertainty coverage (95% confidence interval)
        uncertainty_coverage = 1.0 if (roi_within_bounds and risk_within_bounds) else 0.0
        
        return {
            'prediction_accuracy': prediction_accuracy,
            'uncertainty_coverage': uncertainty_coverage,
            'roi_within_bounds': roi_within_bounds,
            'risk_within_bounds': risk_within_bounds,
            'roi_error': roi_error,
            'risk_error': risk_error
        }
    
    def _calculate_enhanced_anticipative_rate(self, k, h, uncertainty_predictions: Dict, amfc_trajectory: Dict) -> float:
        """Calculate enhanced anticipative rate with uncertainty awareness"""
        
        H = 2
        
        # Component 1: Temporal incomparability with uncertainty
        prediction_uncertainty = np.mean(uncertainty_predictions['uncertainty'])
        tip = max(0.1, min(0.9, 0.5 + prediction_uncertainty))
        temporal_component = self._calculate_anticipation_rate_from_tip(tip, h, H)
        
        # Component 2: AMFC trajectory stability
        trajectory = amfc_trajectory.get('trajectory', [])
        if len(trajectory) > 1:
            # Calculate trajectory stability
            weights_array = np.array([point['weights'] for point in trajectory])
            trajectory_stability = 1.0 - np.mean(np.std(weights_array, axis=0))
        else:
            trajectory_stability = 0.5
        
        # Combined anticipation rate
        anticipative_rate = 0.5 * (temporal_component + trajectory_stability)
        
        return max(0.1, min(0.9, anticipative_rate))
    
    def _calculate_anticipation_rate_from_tip(self, tip, horizon, H=2):
        """Calculate anticipation rate using Equation 6.6"""
        
        if tip <= 0 or tip >= 1:
            entropy = 0.0
        else:
            entropy = -tip * np.log(tip) - (1 - tip) * np.log(1 - tip)
        
        anticipation_rate = (1.0 / (H - 1)) * (1.0 - entropy)
        return max(0.1, min(0.9, anticipation_rate))
    
    def _run_uncertainty_traditional_benchmark(self, historical_days, stride_days, n_periods):
        """Run uncertainty-aware traditional benchmark"""
        
        wealth_history = [100000.0]
        roi_history = []
        
        current_wealth = 100000.0
        
        for period in range(n_periods):
            start_idx = period * stride_days
            end_idx = start_idx + historical_days
            future_start = end_idx
            future_end = min(end_idx + 60, len(self.returns_data))
            
            historical_data = self.returns_data.iloc[start_idx:end_idx]
            future_data = self.returns_data.iloc[future_start:future_end]
            
            # Equal weighted portfolio
            n_assets = len(historical_data.columns)
            weights = np.ones(n_assets) / n_assets
            
            if len(future_data) > 0:
                period_returns = future_data.values @ weights
                period_returns = np.clip(period_returns, -0.20, 0.20)
                period_roi = np.mean(period_returns)
            else:
                period_roi = 0.0
            
            wealth_change = current_wealth * period_roi
            new_wealth = current_wealth + wealth_change
            
            wealth_history.append(new_wealth)
            roi_history.append(period_roi)
            current_wealth = new_wealth
        
        return {
            'wealth_history': wealth_history,
            'roi_history': roi_history,
            'final_wealth': wealth_history[-1] if wealth_history else 100000.0,
            'total_roi': (wealth_history[-1] - 100000.0) / 100000.0 if wealth_history else 0.0,
            'avg_roi_per_period': np.mean(roi_history) if roi_history else 0.0
        }

def generate_uncertainty_aware_report(all_results):
    """Generate uncertainty-aware experiment report"""
    
    # Create strategy list
    strategy_list = []
    
    # Uncertainty-aware ASMS-EMOA strategies
    for k in [0, 1, 2, 3]:
        for h in [1, 2]:
            strategy_list.append(f'Uncertainty_ASMS_EMOA_K{k}_h{h}_Hv-DM')
    
    # Traditional benchmarks
    strategy_list.extend(['Uncertainty_Equal_Weighted'])
    
    # Aggregate results
    aggregated_results = {}
    
    for strategy in strategy_list:
        if strategy in all_results['run_0']:
            total_rois = []
            avg_rois = []
            final_wealths = []
            prediction_accuracies = []
            uncertainty_coverages = []
            
            for run_key in all_results.keys():
                if strategy in all_results[run_key]:
                    total_rois.append(all_results[run_key][strategy]['total_roi'])
                    avg_rois.append(all_results[run_key][strategy]['avg_roi_per_period'])
                    final_wealths.append(all_results[run_key][strategy]['final_wealth'])
                    if 'avg_prediction_accuracy' in all_results[run_key][strategy]:
                        prediction_accuracies.append(all_results[run_key][strategy]['avg_prediction_accuracy'])
                    if 'avg_uncertainty_coverage' in all_results[run_key][strategy]:
                        uncertainty_coverages.append(all_results[run_key][strategy]['avg_uncertainty_coverage'])
            
            aggregated_results[strategy] = {
                'mean_total_roi': np.mean(total_rois),
                'std_total_roi': np.std(total_rois),
                'mean_avg_roi': np.mean(avg_rois),
                'std_avg_roi': np.std(avg_rois),
                'mean_final_wealth': np.mean(final_wealths),
                'std_final_wealth': np.std(final_wealths),
                'mean_prediction_accuracy': np.mean(prediction_accuracies) if prediction_accuracies else 0.5,
                'mean_uncertainty_coverage': np.mean(uncertainty_coverages) if uncertainty_coverages else 0.5
            }
    
    # Generate report
    report = []
    report.append("# Uncertainty-Aware ASMS-EMOA Experiment Report")
    report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    report.append("## Uncertainty-Aware Features Implemented")
    report.append("")
    report.append("1. **Bayesian Neural Networks with Uncertainty Bounds**")
    report.append("2. **AMFC Trajectory Tracking in Decision Space**")
    report.append("3. **Bivariate Gaussian Distribution Updates**")
    report.append("4. **Future Portfolio Composition Prediction**")
    report.append("5. **Uncertainty-Aware Hypervolume Calculation**")
    report.append("")
    
    report.append("## Performance Summary (Mean ± Std across runs)")
    report.append("")
    report.append("| Strategy | Total ROI (%) | Avg ROI/Period (%) | Final Wealth (R$) | Prediction Accuracy | Uncertainty Coverage |")
    report.append("|----------|---------------|-------------------|-------------------|-------------------|-------------------|")
    
    for strategy in strategy_list:
        if strategy in aggregated_results:
            data = aggregated_results[strategy]
            report.append(f"| {strategy} | {data['mean_total_roi']*100:.2f} ± {data['std_total_roi']*100:.2f} | "
                         f"{data['mean_avg_roi']*100:.4f} ± {data['std_avg_roi']*100:.4f} | "
                         f"R$ {data['mean_final_wealth']:,.0f} ± {data['std_final_wealth']:,.0f} | "
                         f"{data['mean_prediction_accuracy']:.3f} | "
                         f"{data['mean_uncertainty_coverage']:.3f} |")
    
    # Best performing strategy
    report.append("\n## Best Performing Strategy")
    if aggregated_results:
        best_strategy = max(aggregated_results.keys(), key=lambda x: aggregated_results[x]['mean_total_roi'])
        best_data = aggregated_results[best_strategy]
        report.append(f"\n**Best Overall Strategy**: {best_strategy}")
        report.append(f"- Total ROI: {best_data['mean_total_roi']*100:.2f}% ± {best_data['std_total_roi']*100:.2f}%")
        report.append(f"- Average ROI per Period: {best_data['mean_avg_roi']*100:.4f}% ± {best_data['std_avg_roi']*100:.4f}%")
        report.append(f"- Final Wealth: R$ {best_data['mean_final_wealth']:,.0f} ± {best_data['std_final_wealth']:,.0f}")
        report.append(f"- Prediction Accuracy: {best_data['mean_prediction_accuracy']:.3f}")
        report.append(f"- Uncertainty Coverage: {best_data['mean_uncertainty_coverage']:.3f}")
    
    # Save report
    with open('uncertainty_aware_experiment_report.md', 'w') as f:
        f.write('\n'.join(report))
    
    logger.info("Uncertainty-aware report generated: uncertainty_aware_experiment_report.md")

def main():
    """Main function for uncertainty-aware experiment"""
    logger.info("Starting Uncertainty-Aware ASMS-EMOA experiment...")
    
    # Load data
    from real_data_experiment import load_existing_ftse_data
    returns_data = load_existing_ftse_data()
    
    # Run uncertainty-aware experiment
    experiment = UncertaintyAwareASMSEMOAExperiment(returns_data)
    all_results = experiment.run_uncertainty_aware_experiment(num_runs=3)
    
    # Generate uncertainty-aware report
    generate_uncertainty_aware_report(all_results)
    
    logger.info("Uncertainty-aware experiment completed!")

if __name__ == "__main__":
    main() 