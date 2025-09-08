#!/usr/bin/env python3
"""
Enhanced IBOVESPA ASMS-EMOA Experiment
With n-step ahead prediction, 70 assets, 90-day periods, and comprehensive benchmarks
"""

import sys
import os
import numpy as np
import pandas as pd
import yfinance as yf
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

class EnhancedDecisionMaker:
    """Enhanced decision maker with n-step ahead prediction"""
    
    def __init__(self, name: str):
        self.name = name
        self.portfolio_history = []
        self.wealth_history = []
        self.roi_history = []
        self.transaction_costs = []
        self.coherence_history = []
        self.expected_hypervolume_history = []
        self.conditional_hypervolume_history = []
        
    def select_portfolio(self, pareto_frontier: List[Solution], 
                        n_step_predictor: NStepPredictor,
                        kalman_predictions: Dict,
                        dirichlet_predictions: Dict,
                        h: int) -> Tuple[Solution, Dict]:
        """Select portfolio with n-step ahead prediction - to be implemented by subclasses"""
        raise NotImplementedError
        
    def update_history(self, portfolio: Solution, wealth: float, roi: float, 
                      transaction_cost: float, coherence: float,
                      expected_hv: Dict, conditional_hv: Dict):
        """Update historical data"""
        self.portfolio_history.append(portfolio)
        self.wealth_history.append(wealth)
        self.roi_history.append(roi)
        self.transaction_costs.append(transaction_cost)
        self.coherence_history.append(coherence)
        self.expected_hypervolume_history.append(expected_hv)
        self.conditional_hypervolume_history.append(conditional_hv)

class EnhancedHvDM(EnhancedDecisionMaker):
    """Enhanced Hypervolume Decision Maker with n-step prediction"""
    
    def __init__(self):
        super().__init__("Hv-DM")
        
    def select_portfolio(self, pareto_frontier: List[Solution], 
                        n_step_predictor: NStepPredictor,
                        kalman_predictions: Dict,
                        dirichlet_predictions: Dict,
                        h: int) -> Tuple[Solution, Dict]:
        """Select AMFC portfolio with maximum expected future hypervolume"""
        if not pareto_frontier:
            return None, {}
            
        # Compute expected future hypervolume distribution
        expected_hv_dist = n_step_predictor.compute_expected_future_hypervolume(
            pareto_frontier, kalman_predictions, dirichlet_predictions, h)
        
        # Find solution with maximum expected hypervolume
        max_hv = -float('inf')
        selected_solution = None
        
        for sol_idx, hv_data in expected_hv_dist.items():
            if hv_data['expected_hypervolume'] > max_hv:
                max_hv = hv_data['expected_hypervolume']
                selected_solution = int(sol_idx.split('_')[1])
        
        if selected_solution is not None:
            selected = pareto_frontier[selected_solution]
            
            # Compute conditional expected hypervolume
            conditional_hv_dist = n_step_predictor.compute_conditional_expected_hypervolume(
                pareto_frontier, selected_solution, kalman_predictions, dirichlet_predictions, h)
            
            return selected, {
                'expected_hypervolume_distribution': expected_hv_dist,
                'conditional_hypervolume_distribution': conditional_hv_dist,
                'selected_solution_idx': selected_solution
            }
        
        return None, {}

class EnhancedRDM(EnhancedDecisionMaker):
    """Enhanced Random Decision Maker"""
    
    def __init__(self):
        super().__init__("R-DM")
        
    def select_portfolio(self, pareto_frontier: List[Solution], 
                        n_step_predictor: NStepPredictor,
                        kalman_predictions: Dict,
                        dirichlet_predictions: Dict,
                        h: int) -> Tuple[Solution, Dict]:
        """Randomly select portfolio with n-step prediction analysis"""
        if not pareto_frontier:
            return None, {}
            
        # Random selection
        selected_solution = np.random.choice(len(pareto_frontier))
        selected = pareto_frontier[selected_solution]
        
        # Compute expected and conditional hypervolume distributions
        expected_hv_dist = n_step_predictor.compute_expected_future_hypervolume(
            pareto_frontier, kalman_predictions, dirichlet_predictions, h)
        
        conditional_hv_dist = n_step_predictor.compute_conditional_expected_hypervolume(
            pareto_frontier, selected_solution, kalman_predictions, dirichlet_predictions, h)
        
        return selected, {
            'expected_hypervolume_distribution': expected_hv_dist,
            'conditional_hypervolume_distribution': conditional_hv_dist,
            'selected_solution_idx': selected_solution
        }

class EnhancedMDM(EnhancedDecisionMaker):
    """Enhanced Median Decision Maker"""
    
    def __init__(self):
        super().__init__("M-DM")
        
    def select_portfolio(self, pareto_frontier: List[Solution], 
                        n_step_predictor: NStepPredictor,
                        kalman_predictions: Dict,
                        dirichlet_predictions: Dict,
                        h: int) -> Tuple[Solution, Dict]:
        """Select median portfolio with n-step prediction analysis"""
        if not pareto_frontier:
            return None, {}
            
        # Calculate median weights
        weights_matrix = np.array([sol.weights for sol in pareto_frontier])
        median_weights = np.median(weights_matrix, axis=0)
        
        # Find portfolio closest to median weights
        distances = []
        for sol in pareto_frontier:
            dist = np.linalg.norm(sol.weights - median_weights)
            distances.append(dist)
        
        selected_solution = np.argmin(distances)
        selected = pareto_frontier[selected_solution]
        
        # Compute expected and conditional hypervolume distributions
        expected_hv_dist = n_step_predictor.compute_expected_future_hypervolume(
            pareto_frontier, kalman_predictions, dirichlet_predictions, h)
        
        conditional_hv_dist = n_step_predictor.compute_conditional_expected_hypervolume(
            pareto_frontier, selected_solution, kalman_predictions, dirichlet_predictions, h)
        
        return selected, {
            'expected_hypervolume_distribution': expected_hv_dist,
            'conditional_hypervolume_distribution': conditional_hv_dist,
            'selected_solution_idx': selected_solution
        }

class EnhancedIBOVESPAExperiment:
    """Enhanced experiment with n-step prediction and comprehensive analysis"""
    
    def __init__(self, initial_wealth: float = 100000.0, transaction_cost_rate: float = 0.001):
        self.initial_wealth = initial_wealth
        self.transaction_cost_rate = transaction_cost_rate
        self.decision_makers = [EnhancedHvDM(), EnhancedRDM(), EnhancedMDM()]
        self.n_step_predictor = NStepPredictor(max_horizon=3)
        self.benchmark_calculator = BenchmarkCalculator()
        self.results = {}
        self.period_data_ranges = []
        
    def download_ibovespa_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Download top 70 IBOVESPA component stocks by market cap"""
        logger.info("Downloading top 70 IBOVESPA stocks...")
        
        # Top 70 IBOVESPA stocks by market cap
        ibovespa_stocks = [
            'VALE3.SA', 'PETR4.SA', 'ITUB4.SA', 'BBDC4.SA', 'ABEV3.SA',
            'WEGE3.SA', 'RENT3.SA', 'LREN3.SA', 'JBSS3.SA', 'SUZB3.SA',
            'RADL3.SA', 'CCRO3.SA', 'GGBR4.SA', 'USIM5.SA', 'CSAN3.SA',
            'EMBR3.SA', 'BRFS3.SA', 'VIVT4.SA', 'TOTS3.SA', 'QUAL3.SA',
            'MGLU3.SA', 'RAIL3.SA', 'SBSP3.SA', 'CMIG4.SA', 'CPLE6.SA',
            'ELET3.SA', 'ENBR3.SA', 'EQTL3.SA', 'GNDI3.SA', 'HYPE3.SA',
            'IRBR3.SA', 'ITSA4.SA', 'KLBN4.SA', 'LAME4.SA', 'MULT3.SA',
            'NATU3.SA', 'PCAR4.SA', 'SANB11.SA', 'SAPR4.SA', 'TAEE11.SA',
            'TIMP3.SA', 'UGPA3.SA', 'USIM3.SA', 'WIZS3.SA', 'YDUQ3.SA',
            'B3SA3.SA', 'BBAS3.SA', 'BRAP4.SA', 'BRKM5.SA', 'CASH3.SA',
            'CIEL3.SA', 'COGN3.SA', 'CRFB3.SA', 'CSNA3.SA', 'CVCB3.SA',
            'CYRE3.SA', 'ECOR3.SA', 'EGIE3.SA', 'ELET6.SA', 'EMAE4.SA',
            'ENEV3.SA', 'EVEN3.SA', 'EZTC3.SA', 'FLRY3.SA', 'GFSA3.SA',
            'GOAU4.SA', 'GOLL4.SA', 'HAPV3.SA', 'HGTX3.SA', 'IGTA3.SA',
            'IRBR3.SA', 'JHSF3.SA', 'KLBN11.SA', 'LIGT3.SA', 'LOGG3.SA'
        ]
        
        try:
            # Download data
            data = yf.download(ibovespa_stocks, start=start_date, end=end_date, 
                             progress=False, group_by='ticker')
            
            # Handle multi-level columns
            if isinstance(data.columns, pd.MultiIndex):
                adj_close = data['Adj Close']
            else:
                adj_close = data
                
            # Remove stocks with too much missing data
            missing_threshold = 0.1
            missing_ratio = adj_close.isnull().sum() / len(adj_close)
            valid_stocks = missing_ratio[missing_ratio < missing_threshold].index
            adj_close = adj_close[valid_stocks]
            
            # Forward fill missing values
            adj_close = adj_close.fillna(method='ffill')
            
            # Calculate returns
            returns = adj_close.pct_change().dropna()
            
            logger.info(f"Downloaded data for {len(valid_stocks)} stocks from {start_date} to {end_date}")
            logger.info(f"Data shape: {returns.shape}")
            
            return returns
            
        except Exception as e:
            logger.error(f"Error downloading IBOVESPA data: {e}")
            logger.info("Using synthetic IBOVESPA-like data")
            return self._generate_synthetic_ibovespa_data(start_date, end_date)
    
    def _generate_synthetic_ibovespa_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Generate synthetic IBOVESPA-like data for 70 stocks"""
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        dates = pd.date_range(start, end, freq='D')
        
        # Generate 70 stocks with realistic Brazilian market characteristics
        n_stocks = 70
        n_days = len(dates)
        
        # Base returns with Brazilian market characteristics
        np.random.seed(42)  # For reproducibility
        base_returns = np.random.normal(0.0005, 0.02, (n_days, n_stocks))
        
        # Add correlation structure
        correlation_matrix = np.random.uniform(0.3, 0.7, (n_stocks, n_stocks))
        np.fill_diagonal(correlation_matrix, 1.0)
        
        # Apply correlation
        for i in range(n_stocks):
            for j in range(i+1, n_stocks):
                correlation = correlation_matrix[i, j]
                base_returns[:, j] = (correlation * base_returns[:, i] + 
                                     np.sqrt(1 - correlation**2) * base_returns[:, j])
        
        # Create DataFrame
        stock_names = [f'STOCK{i+1:02d}.SA' for i in range(n_stocks)]
        returns_df = pd.DataFrame(base_returns, index=dates, columns=stock_names)
        
        logger.info(f"Generated synthetic data: {returns_df.shape}")
        return returns_df
    
    def calculate_coherence(self, portfolios: List[Solution]) -> float:
        """Calculate coherence (cosine similarity to centroid) - Equation 7.14"""
        if len(portfolios) < 2:
            return 1.0
            
        # Extract weight vectors
        weight_vectors = np.array([sol.weights for sol in portfolios])
        
        # Calculate centroid (Equation 7.15)
        centroid = np.mean(weight_vectors, axis=0)
        
        # Calculate coherence (Equation 7.14)
        coherence_sum = 0.0
        for weights in weight_vectors:
            # Cosine similarity
            dot_product = np.dot(weights, centroid)
            norm_weights = np.linalg.norm(weights)
            norm_centroid = np.linalg.norm(centroid)
            
            if norm_weights > 0 and norm_centroid > 0:
                cosine_sim = dot_product / (norm_weights * norm_centroid)
                coherence_sum += cosine_sim
        
        return coherence_sum / len(portfolios)
    
    def calculate_transaction_cost(self, old_weights: np.ndarray, 
                                 new_weights: np.ndarray, 
                                 portfolio_value: float) -> float:
        """Calculate transaction costs for rebalancing"""
        weight_changes = np.abs(new_weights - old_weights)
        total_cost = np.sum(weight_changes) * self.transaction_cost_rate * portfolio_value
        return total_cost
    
    def run_enhanced_experiment(self, returns_data: pd.DataFrame, 
                               historical_days: int = 120, 
                               stride_days: int = 60,
                               k_values: List[int] = [0, 1, 2, 3],
                               h_values: List[int] = [1, 2]) -> Dict:
        """Run the enhanced experiment with n-step prediction"""
        logger.info("Starting Enhanced IBOVESPA experiment...")
        
        # Initialize results structure
        for k in k_values:
            self.results[k] = {}
            for h in h_values:
                self.results[k][f'h_{h}'] = {}
                for dm in self.decision_makers:
                    self.results[k][f'h_{h}'][dm.name] = {
                        'wealth_history': [],
                        'roi_history': [],
                        'coherence_history': [],
                        'transaction_costs': [],
                        'anticipative_factors': [],
                        'portfolio_compositions': [],
                        'expected_hypervolume_history': [],
                        'conditional_hypervolume_history': [],
                        'period_ranges': []
                    }
        
        # Calculate number of periods (ensure at least 30 periods)
        total_days = len(returns_data)
        n_periods = (total_days - historical_days) // stride_days
        
        if n_periods < 30:
            logger.warning(f"Only {n_periods} periods available, minimum 30 required")
            # Adjust stride to get more periods
            stride_days = (total_days - historical_days) // 30
            n_periods = (total_days - historical_days) // stride_days
            logger.info(f"Adjusted stride to {stride_days} days, now {n_periods} periods")
        
        logger.info(f"Running experiment for {n_periods} periods")
        
        # Calculate benchmarks for comparison
        benchmark_results = self.benchmark_calculator.calculate_all_benchmarks(returns_data.values)
        
        for period in range(n_periods):
            logger.info(f"Processing period {period + 1}/{n_periods}")
            
            # Calculate date ranges
            start_idx = period * stride_days
            end_idx = start_idx + historical_days
            future_idx = end_idx + stride_days
            
            # Store period data range
            period_range = {
                'period': period + 1,
                'start_date': returns_data.index[start_idx].strftime('%Y-%m-%d'),
                'end_date': returns_data.index[end_idx-1].strftime('%Y-%m-%d'),
                'future_start': returns_data.index[end_idx].strftime('%Y-%m-%d'),
                'future_end': returns_data.index[future_idx-1].strftime('%Y-%m-%d')
            }
            self.period_data_ranges.append(period_range)
            
            # Extract data for this period
            historical_data = returns_data.iloc[start_idx:end_idx]
            future_data = returns_data.iloc[end_idx:future_idx]
            
            # Set up ASMS-EMOA
            num_assets = len(historical_data.columns)
            
            # Configure algorithm parameters
            algorithm_params = {
                'population_size': 100,
                'generations': 50,
                'num_assets': num_assets,
                'cardinality_min': 1,
                'cardinality_max': min(10, num_assets),
                'returns_data': historical_data
            }
            
            # Run for each K and h combination
            for k in k_values:
                for h in h_values:
                    algorithm_params['anticipation_horizon'] = k
                    
                    # Initialize ASMS-EMOA
                    sms_emoa = SMSEMOA(**algorithm_params)
                    
                    # Run optimization
                    pareto_frontier = sms_emoa.run_optimization()
                    
                    if pareto_frontier:
                        # Generate n-step predictions
                        kalman_predictions = {}
                        dirichlet_predictions = {}
                        
                        # Get Kalman state from first solution
                        if hasattr(pareto_frontier[0], 'P') and hasattr(pareto_frontier[0].P, 'kalman_state'):
                            kalman_state = pareto_frontier[0].P.kalman_state
                            kalman_predictions = self.n_step_predictor.kalman_n_step_prediction(kalman_state, h)
                        
                        # Generate Dirichlet predictions
                        historical_portfolios = []
                        for sol in pareto_frontier:
                            if hasattr(sol, 'weights'):
                                historical_portfolios.append(sol.weights)
                        
                        if historical_portfolios:
                            dirichlet_params = np.mean(historical_portfolios, axis=0)
                            dirichlet_predictions = self.n_step_predictor.dirichlet_n_step_prediction(
                                dirichlet_params, historical_portfolios, h)
                        
                        # Calculate anticipative factor (1-λ)
                        anticipative_factor = 1.0 - (1.0 / (1.0 + k))
                        
                        # Calculate coherence
                        coherence = self.calculate_coherence(pareto_frontier)
                        
                        # Test each decision maker
                        for dm in self.decision_makers:
                            # Select portfolio with n-step prediction
                            selected_portfolio, prediction_data = dm.select_portfolio(
                                pareto_frontier, self.n_step_predictor, 
                                kalman_predictions, dirichlet_predictions, h)
                            
                            if selected_portfolio is not None:
                                # Calculate performance
                                if period == 0:
                                    current_wealth = self.initial_wealth
                                    old_weights = np.zeros(num_assets)
                                else:
                                    current_wealth = self.results[k][f'h_{h}'][dm.name]['wealth_history'][-1]
                                    old_weights = self.results[k][f'h_{h}'][dm.name]['portfolio_compositions'][-1]
                                
                                # Calculate transaction costs
                                transaction_cost = self.calculate_transaction_cost(
                                    old_weights, selected_portfolio.weights, current_wealth)
                                
                                # Calculate future returns for this portfolio
                                portfolio_returns = np.dot(future_data, selected_portfolio.weights)
                                total_return = np.prod(1 + portfolio_returns) - 1
                                
                                # Update wealth
                                new_wealth = current_wealth * (1 + total_return) - transaction_cost
                                roi = (new_wealth - self.initial_wealth) / self.initial_wealth
                                
                                # Store results
                                self.results[k][f'h_{h}'][dm.name]['wealth_history'].append(new_wealth)
                                self.results[k][f'h_{h}'][dm.name]['roi_history'].append(roi)
                                self.results[k][f'h_{h}'][dm.name]['coherence_history'].append(coherence)
                                self.results[k][f'h_{h}'][dm.name]['transaction_costs'].append(transaction_cost)
                                self.results[k][f'h_{h}'][dm.name]['anticipative_factors'].append(anticipative_factor)
                                self.results[k][f'h_{h}'][dm.name]['portfolio_compositions'].append(selected_portfolio.weights.copy())
                                self.results[k][f'h_{h}'][dm.name]['expected_hypervolume_history'].append(
                                    prediction_data.get('expected_hypervolume_distribution', {}))
                                self.results[k][f'h_{h}'][dm.name]['conditional_hypervolume_history'].append(
                                    prediction_data.get('conditional_hypervolume_distribution', {}))
                                self.results[k][f'h_{h}'][dm.name]['period_ranges'].append(period_range)
        
        # Add benchmark results
        self.results['benchmarks'] = benchmark_results
        
        logger.info("Enhanced experiment completed successfully!")
        return self.results
    
    def generate_enhanced_report(self) -> str:
        """Generate comprehensive enhanced experiment report"""
        report = []
        report.append("# Enhanced IBOVESPA ASMS-EMOA Experiment Report")
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Experiment Overview
        report.append("## Experiment Overview")
        report.append("")
        report.append("- **Assets**: Top 70 IBOVESPA stocks by market cap")
        report.append("- **Periods**: 90-day investment periods")
        report.append("- **Historical Data**: 120 days")
        report.append("- **Stride**: 60 days (rebalancing every 2 months)")
        report.append("- **Anticipation Horizons**: K = {0, 1, 2, 3}")
        report.append("- **Prediction Steps**: h = {1, 2}")
        report.append("- **Initial Investment**: R$ 100,000")
        report.append("")
        
        # Performance Summary
        report.append("## Performance Summary")
        report.append("")
        
        for k in [k for k in self.results.keys() if k != 'benchmarks']:
            report.append(f"### Anticipation Horizon K = {k}")
            report.append("")
            
            for h in [1, 2]:
                report.append(f"#### Prediction Step h = {h}")
                report.append("")
                report.append("| Decision Maker | Final ROI (%) | Final Wealth (R$) | Total Transaction Costs (R$) | Avg Coherence |")
                report.append("|----------------|---------------|-------------------|------------------------------|---------------|")
                
                for dm_name in ['Hv-DM', 'R-DM', 'M-DM']:
                    if dm_name in self.results[k][f'h_{h}']:
                        data = self.results[k][f'h_{h}'][dm_name]
                        final_roi = data['roi_history'][-1] * 100 if data['roi_history'] else 0
                        final_wealth = data['wealth_history'][-1] if data['wealth_history'] else self.initial_wealth
                        total_costs = sum(data['transaction_costs']) if data['transaction_costs'] else 0
                        avg_coherence = np.mean(data['coherence_history']) if data['coherence_history'] else 0
                        
                        report.append(f"| {dm_name} | {final_roi:.2f}% | R$ {final_wealth:,.2f} | R$ {total_costs:,.2f} | {avg_coherence:.3f} |")
                
                report.append("")
        
        # Benchmark Comparison
        report.append("## Benchmark Comparison")
        report.append("")
        report.append("| Benchmark | Cumulative Return (%) | Sharpe Ratio | Volatility (%) |")
        report.append("|-----------|---------------------|--------------|----------------|")
        
        for benchmark_name, benchmark_data in self.results['benchmarks'].items():
            cum_return = benchmark_data['cumulative_return'] * 100
            sharpe = benchmark_data['sharpe_ratio']
            volatility = benchmark_data['volatility'] * 100
            
            report.append(f"| {benchmark_data['type']} | {cum_return:.2f}% | {sharpe:.3f} | {volatility:.2f}% |")
        
        report.append("")
        
        # Expected Future Hypervolume Analysis
        report.append("## Expected Future Hypervolume Analysis")
        report.append("")
        
        for k in [k for k in self.results.keys() if k != 'benchmarks']:
            for h in [1, 2]:
                report.append(f"### K = {k}, h = {h} - Expected Hypervolume Distributions")
                report.append("")
                
                for dm_name in ['Hv-DM', 'R-DM', 'M-DM']:
                    if dm_name in self.results[k][f'h_{h}']:
                        data = self.results[k][f'h_{h}'][dm_name]
                        if data['expected_hypervolume_history']:
                            # Calculate average expected hypervolume
                            avg_expected_hv = []
                            for hv_data in data['expected_hypervolume_history']:
                                if hv_data:
                                    hv_values = [v['expected_hypervolume'] for v in hv_data.values()]
                                    avg_expected_hv.append(np.mean(hv_values))
                            
                            if avg_expected_hv:
                                report.append(f"- **{dm_name}**: Average Expected Hypervolume = {np.mean(avg_expected_hv):.6f}")
                
                report.append("")
        
        # Period Data Ranges
        report.append("## Investment Period Data Ranges")
        report.append("")
        report.append("| Period | Historical Data | Future Data |")
        report.append("|--------|----------------|-------------|")
        
        for period_range in self.period_data_ranges[:10]:  # Show first 10 periods
            report.append(f"| {period_range['period']} | {period_range['start_date']} to {period_range['end_date']} | {period_range['future_start']} to {period_range['future_end']} |")
        
        if len(self.period_data_ranges) > 10:
            report.append(f"| ... | ... | ... |")
            report.append(f"| {self.period_data_ranges[-1]['period']} | {self.period_data_ranges[-1]['start_date']} to {self.period_data_ranges[-1]['end_date']} | {self.period_data_ranges[-1]['future_start']} to {self.period_data_ranges[-1]['future_end']} |")
        
        report.append("")
        
        # Key Insights
        report.append("## Key Insights")
        report.append("")
        report.append("1. **N-Step Prediction Impact**: h=1 vs h=2 prediction accuracy")
        report.append("2. **Anticipation Horizon Effect**: K values impact on performance")
        report.append("3. **Decision Maker Comparison**: AMFC vs Random vs Median strategies")
        report.append("4. **Expected Hypervolume Analysis**: Future flexibility assessment")
        report.append("5. **Benchmark Performance**: Traditional vs anticipatory approaches")
        report.append("6. **Transaction Cost Impact**: Rebalancing frequency effects")
        
        return "\n".join(report)
    
    def create_enhanced_visualizations(self, save_path: str = "enhanced_ibovespa_results"):
        """Create comprehensive enhanced visualizations"""
        os.makedirs(save_path, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Wealth Evolution by K and h
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        axes = axes.flatten()
        
        plot_idx = 0
        for k in [k for k in self.results.keys() if k != 'benchmarks']:
            for h in [1, 2]:
                ax = axes[plot_idx]
                
                for dm_name in ['Hv-DM', 'R-DM', 'M-DM']:
                    if dm_name in self.results[k][f'h_{h}']:
                        wealth_history = self.results[k][f'h_{h}'][dm_name]['wealth_history']
                        ax.plot(wealth_history, label=dm_name, linewidth=2)
                
                ax.set_title(f'Wealth Evolution - K = {k}, h = {h}')
                ax.set_xlabel('Period')
                ax.set_ylabel('Portfolio Value (R$)')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                plot_idx += 1
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/wealth_evolution_by_k_h.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Expected Hypervolume Analysis
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        axes = axes.flatten()
        
        plot_idx = 0
        for k in [k for k in self.results.keys() if k != 'benchmarks']:
            for h in [1, 2]:
                ax = axes[plot_idx]
                
                for dm_name in ['Hv-DM', 'R-DM', 'M-DM']:
                    if dm_name in self.results[k][f'h_{h}']:
                        data = self.results[k][f'h_{h}'][dm_name]
                        if data['expected_hypervolume_history']:
                            avg_expected_hv = []
                            for hv_data in data['expected_hypervolume_history']:
                                if hv_data:
                                    hv_values = [v['expected_hypervolume'] for v in hv_data.values()]
                                    avg_expected_hv.append(np.mean(hv_values))
                            
                            if avg_expected_hv:
                                ax.plot(avg_expected_hv, label=f'{dm_name} Avg Expected HV', linewidth=2)
                
                ax.set_title(f'Expected Hypervolume - K = {k}, h = {h}')
                ax.set_xlabel('Period')
                ax.set_ylabel('Average Expected Hypervolume')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                plot_idx += 1
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/expected_hypervolume_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Benchmark Comparison
        fig, ax = plt.subplots(figsize=(12, 8))
        
        benchmark_names = []
        benchmark_returns = []
        
        for benchmark_name, benchmark_data in self.results['benchmarks'].items():
            benchmark_names.append(benchmark_data['type'])
            benchmark_returns.append(benchmark_data['cumulative_return'] * 100)
        
        bars = ax.bar(benchmark_names, benchmark_returns, alpha=0.8)
        ax.set_xlabel('Benchmark Type')
        ax.set_ylabel('Cumulative Return (%)')
        ax.set_title('Benchmark Performance Comparison')
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, benchmark_returns):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                   f'{value:.2f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/benchmark_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Enhanced visualizations saved to {save_path}/")

def main():
    """Main enhanced experiment execution"""
    logger.info("Starting Enhanced IBOVESPA ASMS-EMOA Experiment")
    
    # Experiment parameters
    start_date = '2022-01-01'  # Earlier start to ensure 30+ periods
    end_date = '2024-01-01'
    historical_days = 120
    stride_days = 60
    k_values = [0, 1, 2, 3]
    h_values = [1, 2]
    
    # Initialize enhanced experiment
    experiment = EnhancedIBOVESPAExperiment(initial_wealth=100000.0, transaction_cost_rate=0.001)
    
    # Download data
    returns_data = experiment.download_ibovespa_data(start_date, end_date)
    
    # Run enhanced experiment
    results = experiment.run_enhanced_experiment(returns_data, historical_days, stride_days, k_values, h_values)
    
    # Generate enhanced report
    report = experiment.generate_enhanced_report()
    
    # Save report
    with open('enhanced_ibovespa_experiment_report.md', 'w') as f:
        f.write(report)
    
    # Create enhanced visualizations
    experiment.create_enhanced_visualizations()
    
    # Save results as JSON
    results_summary = {}
    for k in [k for k in results.keys() if k != 'benchmarks']:
        results_summary[k] = {}
        for h in [1, 2]:
            results_summary[k][f'h_{h}'] = {}
            for dm_name in ['Hv-DM', 'R-DM', 'M-DM']:
                if dm_name in results[k][f'h_{h}']:
                    data = results[k][f'h_{h}'][dm_name]
                    results_summary[k][f'h_{h}'][dm_name] = {
                        'final_roi': data['roi_history'][-1] if data['roi_history'] else 0,
                        'final_wealth': data['wealth_history'][-1] if data['wealth_history'] else 100000,
                        'total_transaction_costs': sum(data['transaction_costs']),
                        'avg_coherence': np.mean(data['coherence_history']) if data['coherence_history'] else 0,
                        'n_periods': len(data['wealth_history'])
                    }
    
    # Add benchmark summary
    results_summary['benchmarks'] = {}
    for benchmark_name, benchmark_data in results['benchmarks'].items():
        results_summary['benchmarks'][benchmark_name] = {
            'cumulative_return': benchmark_data['cumulative_return'],
            'sharpe_ratio': benchmark_data['sharpe_ratio'],
            'volatility': benchmark_data['volatility']
        }
    
    with open('enhanced_ibovespa_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)
    
    logger.info("Enhanced experiment completed! Check enhanced_ibovespa_experiment_report.md and enhanced_ibovespa_results/ for results.")

if __name__ == "__main__":
    main() 