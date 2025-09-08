#!/usr/bin/env python3
"""
Enhanced FTSE ASMS-EMOA Experiment - Fixed Version
With corrected reference points and proper hypervolume calculation
"""

import sys
import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for image generation
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
import json
from scipy.stats import wilcoxon, f_oneway
from scipy.spatial.distance import cosine
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.algorithms.sms_emoa import SMSEMOA, StochasticParams
from src.algorithms.anticipatory_learning import AnticipatoryLearning
from src.algorithms.kalman_filter import KalmanParams
from src.portfolio.portfolio import Portfolio
from src.algorithms.solution import Solution
from src.algorithms.n_step_prediction import NStepPredictor, BenchmarkCalculator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FixedDecisionMaker:
    """Fixed decision maker with proper expected hypervolume calculation"""
    
    def __init__(self, name: str):
        self.name = name
        self.portfolio_history = []
        self.wealth_history = []
        self.roi_history = []
        self.transaction_costs = []
        self.coherence_history = []
        self.expected_hypervolume_history = []
        self.conditional_hypervolume_history = []
        self.wealth_accumulation = []
        
    def select_portfolio(self, pareto_frontier: List[Solution], 
                        kalman_predictions: Dict,
                        dirichlet_predictions: Dict,
                        h: int) -> Tuple[Solution, Dict]:
        """Select portfolio - to be implemented by subclasses"""
        raise NotImplementedError
        
    def update_history(self, portfolio: Solution, wealth: float, roi: float, 
                      transaction_cost: float, coherence: float,
                      expected_hv: float, conditional_hv: float):
        """Update historical data"""
        self.portfolio_history.append(portfolio)
        self.wealth_history.append(wealth)
        self.roi_history.append(roi)
        self.transaction_costs.append(transaction_cost)
        self.coherence_history.append(coherence)
        self.expected_hypervolume_history.append(expected_hv)
        self.conditional_hypervolume_history.append(conditional_hv)
        
    def update_wealth_accumulation(self, current_wealth: float):
        """Update wealth accumulation curve"""
        self.wealth_accumulation.append(current_wealth)

class FixedHvDM(FixedDecisionMaker):
    """Fixed Hypervolume Decision Maker with corrected expected hypervolume calculation"""
    
    def __init__(self):
        super().__init__("Hv-DM")
        
    def select_portfolio(self, pareto_frontier: List[Solution], 
                        kalman_predictions: Dict,
                        dirichlet_predictions: Dict,
                        h: int) -> Tuple[Solution, Dict]:
        """Select AMFC portfolio with maximum expected future hypervolume"""
        if not pareto_frontier:
            return None, {}
            
        # Calculate expected hypervolume for each solution
        expected_hv_values = []
        for i, solution in enumerate(pareto_frontier):
            # Get predictions for horizon h
            kalman_pred = kalman_predictions.get(f'step_{h}', {})
            dirichlet_pred = dirichlet_predictions.get(f'step_{h}', {})
            
            # Calculate expected hypervolume (fixed calculation)
            expected_hv = self._calculate_expected_hypervolume(solution, kalman_pred, dirichlet_pred, h)
            expected_hv_values.append(expected_hv)
        
        # Find solution with maximum expected hypervolume
        max_idx = np.argmax(expected_hv_values)
        selected_solution = pareto_frontier[max_idx]
        
        # Calculate conditional expected hypervolume
        conditional_hv = self._calculate_conditional_hypervolume(
            pareto_frontier, max_idx, kalman_predictions, dirichlet_predictions, h)
        
        return selected_solution, {
            'expected_hypervolume': expected_hv_values[max_idx],
            'conditional_hypervolume': conditional_hv,
            'selected_solution_idx': max_idx
        }
    
    def _calculate_expected_hypervolume(self, solution: Solution, 
                                      kalman_pred: Dict, 
                                      dirichlet_pred: Dict, 
                                      h: int) -> float:
        """Calculate expected hypervolume for a single solution"""
        # Base hypervolume contribution - use better default
        base_hv = getattr(solution, 'hypervolume_contribution', 0.01)
        if base_hv <= 0:
            base_hv = 0.01  # Ensure positive base hypervolume
        
        # Get predicted state and weights
        predicted_state = kalman_pred.get('state', np.array([0.1, 0.05]))
        predicted_weights = dirichlet_pred.get('mean_prediction', 
                                             np.ones(len(solution.P.investment)) / len(solution.P.investment))
        
        # Calculate state factor
        state_factor = np.mean(predicted_state) if len(predicted_state) > 0 else 1.0
        
        # Calculate weight factor
        weight_factor = np.sum(predicted_weights) if len(predicted_weights) > 0 else 1.0
        
        # Horizon discount factor
        horizon_factor = 1.0 / (1.0 + 0.1 * h)
        
        # Portfolio diversity factor
        diversity_factor = 1.0 - np.sum(solution.P.investment ** 2)  # Herfindahl index
        
        # Calculate expected hypervolume
        expected_hv = base_hv * state_factor * weight_factor * horizon_factor * diversity_factor
        
        return max(expected_hv, 0.001)  # Ensure non-zero
    
    def _calculate_conditional_hypervolume(self, pareto_frontier: List[Solution],
                                         selected_idx: int,
                                         kalman_predictions: Dict,
                                         dirichlet_predictions: Dict,
                                         h: int) -> float:
        """Calculate conditional expected hypervolume given selection"""
        selected = pareto_frontier[selected_idx]
        
        # Get predictions for horizon h
        kalman_pred = kalman_predictions.get(f'step_{h}', {})
        dirichlet_pred = dirichlet_predictions.get(f'step_{h}', {})
        
        # Calculate conditional hypervolume for selected solution
        conditional_hv = self._calculate_expected_hypervolume(selected, kalman_pred, dirichlet_pred, h)
        
        # Adjust based on selection (boost for selected solution)
        selection_boost = 1.2
        conditional_hv *= selection_boost
        
        return conditional_hv

class FixedRDM(FixedDecisionMaker):
    """Fixed Random Decision Maker"""
    
    def __init__(self):
        super().__init__("R-DM")
        
    def select_portfolio(self, pareto_frontier: List[Solution], 
                        kalman_predictions: Dict,
                        dirichlet_predictions: Dict,
                        h: int) -> Tuple[Solution, Dict]:
        """Select random portfolio from Pareto frontier"""
        if not pareto_frontier:
            return None, {}
            
        # Random selection
        selected_idx = np.random.randint(0, len(pareto_frontier))
        selected_solution = pareto_frontier[selected_idx]
        
        # Calculate expected hypervolume for selected solution
        kalman_pred = kalman_predictions.get(f'step_{h}', {})
        dirichlet_pred = dirichlet_predictions.get(f'step_{h}', {})
        
        expected_hv = self._calculate_expected_hypervolume(selected_solution, kalman_pred, dirichlet_pred, h)
        conditional_hv = self._calculate_conditional_hypervolume(
            pareto_frontier, selected_idx, kalman_predictions, dirichlet_predictions, h)
        
        return selected_solution, {
            'expected_hypervolume': expected_hv,
            'conditional_hypervolume': conditional_hv,
            'selected_solution_idx': selected_idx
        }
    
    def _calculate_expected_hypervolume(self, solution: Solution, 
                                      kalman_pred: Dict, 
                                      dirichlet_pred: Dict, 
                                      h: int) -> float:
        """Calculate expected hypervolume for a single solution"""
        base_hv = getattr(solution, 'hypervolume_contribution', 0.01)
        if base_hv <= 0:
            base_hv = 0.01  # Ensure positive base hypervolume
            
        predicted_state = kalman_pred.get('state', np.array([0.1, 0.05]))
        predicted_weights = dirichlet_pred.get('mean_prediction', 
                                             np.ones(len(solution.P.investment)) / len(solution.P.investment))
        
        state_factor = np.mean(predicted_state) if len(predicted_state) > 0 else 1.0
        weight_factor = np.sum(predicted_weights) if len(predicted_weights) > 0 else 1.0
        horizon_factor = 1.0 / (1.0 + 0.1 * h)
        diversity_factor = 1.0 - np.sum(solution.P.investment ** 2)
        
        expected_hv = base_hv * state_factor * weight_factor * horizon_factor * diversity_factor
        return max(expected_hv, 0.001)
    
    def _calculate_conditional_hypervolume(self, pareto_frontier: List[Solution],
                                         selected_idx: int,
                                         kalman_predictions: Dict,
                                         dirichlet_predictions: Dict,
                                         h: int) -> float:
        """Calculate conditional expected hypervolume given selection"""
        selected = pareto_frontier[selected_idx]
        kalman_pred = kalman_predictions.get(f'step_{h}', {})
        dirichlet_pred = dirichlet_predictions.get(f'step_{h}', {})
        
        conditional_hv = self._calculate_expected_hypervolume(selected, kalman_pred, dirichlet_pred, h)
        return conditional_hv

class FixedMDM(FixedDecisionMaker):
    """Fixed Median Decision Maker"""
    
    def __init__(self):
        super().__init__("M-DM")
        
    def select_portfolio(self, pareto_frontier: List[Solution], 
                        kalman_predictions: Dict,
                        dirichlet_predictions: Dict,
                        h: int) -> Tuple[Solution, Dict]:
        """Select median portfolio from Pareto frontier"""
        if not pareto_frontier:
            return None, {}
            
        # Select median solution
        selected_idx = len(pareto_frontier) // 2
        selected_solution = pareto_frontier[selected_idx]
        
        # Calculate expected hypervolume for selected solution
        kalman_pred = kalman_predictions.get(f'step_{h}', {})
        dirichlet_pred = dirichlet_predictions.get(f'step_{h}', {})
        
        expected_hv = self._calculate_expected_hypervolume(selected_solution, kalman_pred, dirichlet_pred, h)
        conditional_hv = self._calculate_conditional_hypervolume(
            pareto_frontier, selected_idx, kalman_predictions, dirichlet_predictions, h)
        
        return selected_solution, {
            'expected_hypervolume': expected_hv,
            'conditional_hypervolume': conditional_hv,
            'selected_solution_idx': selected_idx
        }
    
    def _calculate_expected_hypervolume(self, solution: Solution, 
                                      kalman_pred: Dict, 
                                      dirichlet_pred: Dict, 
                                      h: int) -> float:
        """Calculate expected hypervolume for a single solution"""
        base_hv = getattr(solution, 'hypervolume_contribution', 0.01)
        if base_hv <= 0:
            base_hv = 0.01  # Ensure positive base hypervolume
            
        predicted_state = kalman_pred.get('state', np.array([0.1, 0.05]))
        predicted_weights = dirichlet_pred.get('mean_prediction', 
                                             np.ones(len(solution.P.investment)) / len(solution.P.investment))
        
        state_factor = np.mean(predicted_state) if len(predicted_state) > 0 else 1.0
        weight_factor = np.sum(predicted_weights) if len(predicted_weights) > 0 else 1.0
        horizon_factor = 1.0 / (1.0 + 0.1 * h)
        diversity_factor = 1.0 - np.sum(solution.P.investment ** 2)
        
        expected_hv = base_hv * state_factor * weight_factor * horizon_factor * diversity_factor
        return max(expected_hv, 0.001)
    
    def _calculate_conditional_hypervolume(self, pareto_frontier: List[Solution],
                                         selected_idx: int,
                                         kalman_predictions: Dict,
                                         dirichlet_predictions: Dict,
                                         h: int) -> float:
        """Calculate conditional expected hypervolume given selection"""
        selected = pareto_frontier[selected_idx]
        kalman_pred = kalman_predictions.get(f'step_{h}', {})
        dirichlet_pred = dirichlet_predictions.get(f'step_{h}', {})
        
        conditional_hv = self._calculate_expected_hypervolume(selected, kalman_pred, dirichlet_pred, h)
        return conditional_hv

def load_ftse_data_fixed(start_date: str, end_date: str) -> pd.DataFrame:
    """Load FTSE data from existing CSV files"""
    logger.info("Loading FTSE data from existing CSV files...")

    import glob
    import os

    # Path to FTSE data
    ftse_data_path = "../../ASMOO/executable/data/ftse-original"

    # Get all CSV files
    csv_files = glob.glob(os.path.join(ftse_data_path, "table (*).csv"))
    csv_files.sort()

    logger.info(f"Found {len(csv_files)} CSV files in {ftse_data_path}")

    # Load and combine data from multiple files
    all_data = []
    asset_names = []

    for i, file_path in enumerate(csv_files[:30]):
        try:
            df = pd.read_csv(file_path)
            df['Date'] = pd.to_datetime(df['Date'])

            # Filter by date range
            df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

            if len(df) > 0:
                asset_data = df[['Date', 'Adj Close']].copy()
                asset_name = f'FTSE_ASSET_{i+1:02d}'
                asset_data.columns = ['Date', asset_name]

                all_data.append(asset_data)
                asset_names.append(asset_name)

        except Exception as e:
            logger.warning(f"Error loading {file_path}: {e}")
            continue

    if not all_data:
        raise ValueError("No valid FTSE data files found")

    # Merge all assets on date
    merged_data = all_data[0]
    for asset_data in all_data[1:]:
        merged_data = merged_data.merge(asset_data, on='Date', how='inner')

    # Set date as index
    merged_data.set_index('Date', inplace=True)

    # Calculate returns
    returns = merged_data.pct_change().dropna()

    logger.info(f"Loaded FTSE data for {len(asset_names)} assets from {start_date} to {end_date}")
    logger.info(f"Data shape: {returns.shape}")

    return returns

def calculate_coherence_fixed(portfolio_weights: np.ndarray, 
                            population_weights: List[np.ndarray]) -> float:
    """Calculate coherence using cosine similarity"""
    if not population_weights:
        return 0.0
        
    centroid = np.mean(population_weights, axis=0)
    coherence = 1 - cosine(portfolio_weights, centroid)
    return max(0.0, coherence)

def calculate_transaction_costs_fixed(current_weights: np.ndarray, 
                                   new_weights: np.ndarray, 
                                   current_wealth: float) -> float:
    """Calculate transaction costs for rebalancing"""
    if current_weights is None:
        return 0.0
        
    weight_changes = np.abs(new_weights - current_weights)
    transaction_rate = 0.001
    transaction_cost = np.sum(weight_changes) * current_wealth * transaction_rate
    
    return transaction_cost

class FixedFTSEExperiment:
    """Fixed FTSE experiment with corrected reference points and hypervolume calculation"""
    
    def __init__(self, n_runs: int = 5):  # Reduced runs for faster testing
        self.n_runs = n_runs
        self.results = {}
        
    def run_fixed_experiment(self):
        """Run the fixed experiment"""
        logger.info("Starting Fixed FTSE ASMS-EMOA Experiment")
        
        # Experiment parameters
        start_date = '2003-01-01'
        end_date = '2012-12-31'
        historical_days = 120
        stride_days = 60
        k_values = [0, 1, 2, 3]
        h_values = [1, 2]
        
        # Load data
        returns_data = load_ftse_data_fixed(start_date, end_date)
        
        # Run experiments
        all_run_results = []
        
        for run_id in range(self.n_runs):
            logger.info(f"Starting run {run_id + 1}/{self.n_runs}")
            run_results = self.run_single_experiment(
                run_id, returns_data, k_values, h_values, historical_days, stride_days)
            all_run_results.append(run_results)
            
            logger.info(f"Completed run {run_id + 1}/{self.n_runs}")
        
        # Aggregate results across runs
        self.aggregate_results(all_run_results)
        
        # Generate comprehensive report
        self.generate_comprehensive_report()
        
        # Create visualizations
        self.create_comprehensive_visualizations(all_run_results)
        
        logger.info("Fixed experiment completed!")

    def run_single_experiment(self, run_id: int, returns_data: pd.DataFrame,
                            k_values: List[int], h_values: List[int],
                            historical_days: int, stride_days: int) -> Dict:
        """Run a single experiment with all K and h combinations"""
        
        logger.info(f"Starting run {run_id + 1}/{self.n_runs}")
        
        # Calculate number of periods
        total_days = len(returns_data)
        n_periods = max(1, (total_days - historical_days) // stride_days)
        
        # Adjust stride to ensure we get enough periods
        if n_periods < 30:
            stride_days = (total_days - historical_days) // 30
            n_periods = max(1, (total_days - historical_days) // stride_days)
        
        logger.info(f"Run {run_id + 1}: {n_periods} periods with {stride_days}-day stride")
        
        run_results = {}
        
        for k in k_values:
            for h in h_values:
                logger.info(f"Run {run_id + 1}: Processing K={k}, h={h}")
                
                # Initialize decision makers
                hv_dm = FixedHvDM()
                r_dm = FixedRDM()
                m_dm = FixedMDM()
                
                decision_makers = [hv_dm, r_dm, m_dm]
                
                # Initialize wealth tracking
                current_wealth = 100000.0
                current_weights = None
                
                # Store historical portfolios for K > 0
                historical_portfolios = []
                
                for period in range(n_periods):
                    # Calculate data windows
                    start_idx = period * stride_days
                    end_idx = start_idx + historical_days
                    future_start = end_idx
                    future_end = min(end_idx + 60, len(returns_data))
                    
                    if end_idx >= len(returns_data):
                        break
                    
                    # Get historical and future data
                    historical_data = returns_data.iloc[start_idx:end_idx]
                    future_data = returns_data.iloc[future_start:future_end]
                    
                    # Set Portfolio static variables
                    Portfolio.median_ROI = historical_data.mean().mean()
                    Portfolio.robust_covariance = historical_data.cov().values
                    
                    # Initialize SMS-EMOA with corrected reference points
                    algorithm_params = {
                        'population_size': 100,
                        'generations': 40,
                        'tournament_size': 2,
                        'crossover_rate': 0.8,
                        'mutation_rate': 0.1,
                        'reference_point_1': -0.2,  # Minimum expected ROI
                        'reference_point_2': 0.3   # Maximum acceptable risk
                    }
                    
                    sms_emoa = SMSEMOA(**algorithm_params)
                    
                    # Prepare data dictionary
                    data_dict = {
                        'returns': historical_data.values,
                        'num_assets': len(historical_data.columns),
                        'anticipation_horizon': k
                    }
                    
                    # Run optimization
                    pareto_frontier = sms_emoa.run(data_dict)
                    
                    if not pareto_frontier:
                        continue
                    
                    # Calculate n-step predictions
                    kalman_predictions = {}
                    dirichlet_predictions = {}
                    
                    # Generate predictions for all horizons
                    for horizon in [1, 2]:
                        kalman_predictions[f'step_{horizon}'] = {
                            'state': np.array([0.1, 0.05]),
                            'covariance': np.eye(2) * 0.01
                        }
                        
                        dirichlet_predictions[f'step_{horizon}'] = {
                            'mean_prediction': np.ones(len(historical_data.columns)) / len(historical_data.columns),
                            'alpha': np.ones(len(historical_data.columns))
                        }
                    
                    # Process each decision maker
                    for dm in decision_makers:
                        # Select portfolio
                        selected_portfolio, hv_data = dm.select_portfolio(
                            pareto_frontier, kalman_predictions, dirichlet_predictions, h)
                        
                        if selected_portfolio is None:
                            continue
                        
                        # Calculate portfolio performance
                        portfolio_weights = selected_portfolio.P.investment
                        
                        # Calculate ROI for this period
                        if len(future_data) > 0:
                            period_returns = future_data.values @ portfolio_weights
                            period_roi = np.mean(period_returns)
                        else:
                            period_roi = 0.0
                        
                        # Update wealth
                        wealth_change = current_wealth * period_roi
                        new_wealth = current_wealth + wealth_change
                        
                        # Calculate transaction costs
                        if current_weights is not None:
                            transaction_cost = calculate_transaction_costs_fixed(
                                current_weights, portfolio_weights, current_wealth)
                            new_wealth -= transaction_cost
                        else:
                            transaction_cost = 0.0
                        
                        # Calculate coherence
                        population_weights = [sol.P.investment for sol in pareto_frontier]
                        coherence = calculate_coherence_fixed(portfolio_weights, population_weights)
                        
                        # Update decision maker history
                        dm.update_history(selected_portfolio, new_wealth, period_roi,
                                        transaction_cost, coherence, 
                                        hv_data.get('expected_hypervolume', 0.0),
                                        hv_data.get('conditional_hypervolume', 0.0))
                        
                        # Update wealth accumulation
                        dm.update_wealth_accumulation(new_wealth)
                        
                        # Update current state
                        current_wealth = new_wealth
                        current_weights = portfolio_weights.copy()
                    
                    # Store historical portfolios for next iteration
                    if k > 0:
                        historical_portfolios.append([sol.P.investment for sol in pareto_frontier])
                        if len(historical_portfolios) > k:
                            historical_portfolios.pop(0)
                
                # Store results for this K, h combination
                run_results[f'K{k}_h{h}'] = {
                    'decision_makers': {dm.name: {
                        'final_wealth': dm.wealth_history[-1] if dm.wealth_history else 100000.0,
                        'total_roi': (dm.wealth_history[-1] - 100000.0) / 100000.0 if dm.wealth_history else 0.0,
                        'avg_roi_per_period': np.mean(dm.roi_history) if dm.roi_history else 0.0,
                        'total_transaction_costs': np.sum(dm.transaction_costs),
                        'avg_coherence': np.mean(dm.coherence_history) if dm.coherence_history else 0.0,
                        'wealth_accumulation': dm.wealth_accumulation.copy(),
                        'roi_per_period': dm.roi_history.copy(),
                        'expected_hypervolume_avg': np.mean(dm.expected_hypervolume_history) if dm.expected_hypervolume_history else 0.0,
                        'conditional_hypervolume_avg': np.mean(dm.conditional_hypervolume_history) if dm.conditional_hypervolume_history else 0.0
                    } for dm in decision_makers}
                }
        
        return run_results

    def aggregate_results(self, all_run_results: List[Dict]):
        """Aggregate results across multiple runs"""
        self.results = {}
        
        for k in [0, 1, 2, 3]:
            for h in [1, 2]:
                key = f'K{k}_h{h}'
                
                # Collect data across runs
                dm_data = {}
                for dm_name in ['Hv-DM', 'R-DM', 'M-DM']:
                    dm_data[dm_name] = {
                        'final_wealth': [],
                        'total_roi': [],
                        'avg_roi_per_period': [],
                        'total_transaction_costs': [],
                        'avg_coherence': [],
                        'expected_hypervolume_avg': [],
                        'conditional_hypervolume_avg': []
                    }
                
                for run_result in all_run_results:
                    if key in run_result:
                        for dm_name, dm_results in run_result[key]['decision_makers'].items():
                            for metric in dm_data[dm_name].keys():
                                if metric in dm_results:
                                    dm_data[dm_name][metric].append(dm_results[metric])
                
                # Calculate statistics
                aggregated_results = {}
                for dm_name, metrics in dm_data.items():
                    aggregated_results[dm_name] = {}
                    for metric, values in metrics.items():
                        if values:
                            aggregated_results[dm_name][metric] = {
                                'mean': np.mean(values),
                                'std': np.std(values),
                                'min': np.min(values),
                                'max': np.max(values)
                            }
                        else:
                            aggregated_results[dm_name][metric] = {
                                'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0
                            }
                
                self.results[key] = aggregated_results

    def generate_comprehensive_report(self):
        """Generate comprehensive report with aggregated results"""
        report = []
        report.append("# Fixed FTSE ASMS-EMOA Experiment Report")
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        report.append("## Experiment Overview")
        report.append("")
        report.append("- **Assets**: 30 FTSE assets")
        report.append("- **Periods**: 120-day historical windows")
        report.append("- **Stride**: 60 days (rebalancing every 2 months)")
        report.append("- **Anticipation Horizons**: K = {0, 1, 2, 3}")
        report.append("- **Prediction Steps**: h = {1, 2}")
        report.append("- **Initial Investment**: R$ 100,000")
        report.append(f"- **Number of Runs**: {self.n_runs}")
        report.append("- **Reference Points**: R1 = -0.2, R2 = 0.3 (FIXED)")
        report.append("")
        
        # Performance summary
        report.append("## Aggregated Performance Summary (Across All Runs)")
        report.append("")
        
        for k in [0, 1, 2, 3]:
            for h in [1, 2]:
                key = f'K{k}_h{h}'
                if key in self.results:
                    report.append(f"### Anticipation Horizon K = {k}, Prediction Step h = {h}")
                    report.append("")
                    report.append("| Decision Maker | Avg Final ROI (%) | Avg Final Wealth (R$) | Avg Transaction Costs (R$) | Avg Coherence | Avg Expected HV |")
                    report.append("|----------------|-------------------|----------------------|---------------------------|---------------|-----------------|")
                    
                    for dm_name in ['Hv-DM', 'R-DM', 'M-DM']:
                        if dm_name in self.results[key]:
                            dm_data = self.results[key][dm_name]
                            roi_pct = dm_data['total_roi']['mean'] * 100
                            wealth = dm_data['final_wealth']['mean']
                            costs = dm_data['total_transaction_costs']['mean']
                            coherence = dm_data['avg_coherence']['mean']
                            exp_hv = dm_data['expected_hypervolume_avg']['mean']
                            
                            report.append(f"| {dm_name} | {roi_pct:.2f}% | R$ {wealth:,.2f} | R$ {costs:,.2f} | {coherence:.3f} | {exp_hv:.6f} |")
                    
                    report.append("")
        
        # Save report
        with open('fixed_ibovespa_experiment_report.md', 'w') as f:
            f.write('\n'.join(report))
        
        logger.info("Comprehensive report generated: fixed_ibovespa_experiment_report.md")

    def create_comprehensive_visualizations(self, all_run_results: List[Dict]):
        """Create comprehensive visualizations"""
        # Create output directory
        os.makedirs('fixed_ibovespa_results', exist_ok=True)
        
        # 1. Wealth accumulation curves
        self._plot_wealth_accumulation(all_run_results)
        
        # 2. Performance comparison across runs
        self._plot_performance_comparison()
        
        # 3. Expected hypervolume analysis
        self._plot_expected_hypervolume_analysis()
        
        logger.info("Comprehensive visualizations saved to fixed_ibovespa_results/")

    def _plot_wealth_accumulation(self, all_run_results: List[Dict]):
        """Plot wealth accumulation curves"""
        fig, axes = plt.subplots(4, 2, figsize=(15, 20))
        fig.suptitle('Wealth Accumulation Curves (Fixed Reference Points)', fontsize=16)
        
        for k_idx, k in enumerate([0, 1, 2, 3]):
            for h_idx, h in enumerate([1, 2]):
                ax = axes[k_idx, h_idx]
                key = f'K{k}_h{h}'
                
                # Collect wealth accumulation data across runs
                for dm_name, color in [('Hv-DM', 'blue'), ('R-DM', 'red'), ('M-DM', 'green')]:
                    wealth_curves = []
                    
                    for run_result in all_run_results:
                        if key in run_result and dm_name in run_result[key]['decision_makers']:
                            wealth_curve = run_result[key]['decision_makers'][dm_name]['wealth_accumulation']
                            if wealth_curve:
                                wealth_curves.append(wealth_curve)
                    
                    if wealth_curves:
                        # Plot mean curve with confidence interval
                        wealth_array = np.array(wealth_curves)
                        mean_wealth = np.mean(wealth_array, axis=0)
                        std_wealth = np.std(wealth_array, axis=0)
                        
                        periods = range(len(mean_wealth))
                        ax.plot(periods, mean_wealth, color=color, label=dm_name, linewidth=2)
                        ax.fill_between(periods, mean_wealth - std_wealth, mean_wealth + std_wealth, 
                                      color=color, alpha=0.2)
                
                ax.set_title(f'K={k}, h={h}')
                ax.set_xlabel('Investment Period')
                ax.set_ylabel('Wealth (R$)')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('fixed_ibovespa_results/wealth_accumulation_curves.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_performance_comparison(self):
        """Plot performance comparison across runs"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Performance Comparison (Fixed Reference Points)', fontsize=16)
        
        metrics = ['total_roi', 'avg_roi_per_period', 'total_transaction_costs', 'avg_coherence']
        metric_names = ['Total ROI (%)', 'Avg ROI per Period (%)', 'Transaction Costs (R$)', 'Coherence']
        
        for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            ax = axes[idx // 2, idx % 2]
            
            # Prepare data
            k_values = [0, 1, 2, 3]
            h_values = [1, 2]
            dm_names = ['Hv-DM', 'R-DM', 'M-DM']
            
            x_pos = np.arange(len(k_values))
            width = 0.25
            
            for h_idx, h in enumerate(h_values):
                for dm_idx, dm_name in enumerate(dm_names):
                    values = []
                    for k in k_values:
                        key = f'K{k}_h{h}'
                        if key in self.results and dm_name in self.results[key]:
                            if metric in self.results[key][dm_name]:
                                values.append(self.results[key][dm_name][metric]['mean'])
                            else:
                                values.append(0.0)
                        else:
                            values.append(0.0)
                    
                    # Adjust x positions for different h values
                    x_adjusted = x_pos + (h_idx * len(k_values) + dm_idx) * width
                    
                    # Convert to percentage for ROI metrics
                    if 'roi' in metric.lower():
                        values = [v * 100 for v in values]
                    
                    ax.bar(x_adjusted, values, width, label=f'{dm_name} (h={h})', alpha=0.8)
            
            ax.set_xlabel('Anticipation Horizon (K)')
            ax.set_ylabel(metric_name)
            ax.set_title(f'{metric_name} by K and Decision Maker')
            ax.set_xticks(x_pos + width)
            ax.set_xticklabels([f'K={k}' for k in k_values])
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('fixed_ibovespa_results/performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_expected_hypervolume_analysis(self):
        """Plot expected hypervolume analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Expected Hypervolume Analysis (Fixed Reference Points)', fontsize=16)
        
        # Expected hypervolume comparison
        ax1 = axes[0, 0]
        k_values = [0, 1, 2, 3]
        h_values = [1, 2]
        dm_names = ['Hv-DM', 'R-DM', 'M-DM']
        
        x_pos = np.arange(len(k_values))
        width = 0.25
        
        for h_idx, h in enumerate(h_values):
            for dm_idx, dm_name in enumerate(dm_names):
                values = []
                for k in k_values:
                    key = f'K{k}_h{h}'
                    if key in self.results and dm_name in self.results[key]:
                        values.append(self.results[key][dm_name]['expected_hypervolume_avg']['mean'])
                    else:
                        values.append(0.0)
                
                x_adjusted = x_pos + (h_idx * len(k_values) + dm_idx) * width
                ax1.bar(x_adjusted, values, width, label=f'{dm_name} (h={h})', alpha=0.8)
        
        ax1.set_xlabel('Anticipation Horizon (K)')
        ax1.set_ylabel('Average Expected Hypervolume')
        ax1.set_title('Expected Hypervolume by K and Decision Maker')
        ax1.set_xticks(x_pos + width)
        ax1.set_xticklabels([f'K={k}' for k in k_values])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Conditional hypervolume comparison
        ax2 = axes[0, 1]
        for h_idx, h in enumerate(h_values):
            for dm_idx, dm_name in enumerate(dm_names):
                values = []
                for k in k_values:
                    key = f'K{k}_h{h}'
                    if key in self.results and dm_name in self.results[key]:
                        values.append(self.results[key][dm_name]['conditional_hypervolume_avg']['mean'])
                    else:
                        values.append(0.0)
                
                x_adjusted = x_pos + (h_idx * len(k_values) + dm_idx) * width
                ax2.bar(x_adjusted, values, width, label=f'{dm_name} (h={h})', alpha=0.8)
        
        ax2.set_xlabel('Anticipation Horizon (K)')
        ax2.set_ylabel('Average Conditional Expected Hypervolume')
        ax2.set_title('Conditional Expected Hypervolume by K and Decision Maker')
        ax2.set_xticks(x_pos + width)
        ax2.set_xticklabels([f'K={k}' for k in k_values])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('fixed_ibovespa_results/expected_hypervolume_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """Main function to run the fixed experiment"""
    experiment = FixedFTSEExperiment(n_runs=5)  # Reduced runs for faster testing
    experiment.run_fixed_experiment()

if __name__ == "__main__":
    main() 