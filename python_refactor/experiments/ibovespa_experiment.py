#!/usr/bin/env python3
"""
IBOVESPA ASMS-EMOA Experiment
Comparing Hv-DM, R-DM, and M-DM decision makers with different anticipation horizons
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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DecisionMaker:
    """Base class for decision makers"""
    
    def __init__(self, name: str):
        self.name = name
        self.portfolio_history = []
        self.wealth_history = []
        self.roi_history = []
        self.transaction_costs = []
        self.coherence_history = []
        
    def select_portfolio(self, pareto_frontier: List[Solution]) -> Solution:
        """Select portfolio from Pareto frontier - to be implemented by subclasses"""
        raise NotImplementedError
        
    def update_history(self, portfolio: Solution, wealth: float, roi: float, 
                      transaction_cost: float, coherence: float):
        """Update historical data"""
        self.portfolio_history.append(portfolio)
        self.wealth_history.append(wealth)
        self.roi_history.append(roi)
        self.transaction_costs.append(transaction_cost)
        self.coherence_history.append(coherence)

class HvDM(DecisionMaker):
    """Hypervolume Decision Maker - selects AMFC portfolio"""
    
    def __init__(self):
        super().__init__("Hv-DM")
        
    def select_portfolio(self, pareto_frontier: List[Solution]) -> Solution:
        """Select portfolio with maximum expected hypervolume"""
        if not pareto_frontier:
            return None
            
        # Calculate expected hypervolume for each solution
        hypervolumes = []
        for solution in pareto_frontier:
            if hasattr(solution, 'expected_hypervolume'):
                hypervolumes.append(solution.expected_hypervolume)
            else:
                # Fallback: use current hypervolume contribution
                hypervolumes.append(getattr(solution, 'hypervolume_contribution', 0.0))
        
        # Select solution with maximum hypervolume
        max_idx = np.argmax(hypervolumes)
        return pareto_frontier[max_idx]

class RDM(DecisionMaker):
    """Random Decision Maker - randomly selects from Pareto frontier"""
    
    def __init__(self):
        super().__init__("R-DM")
        
    def select_portfolio(self, pareto_frontier: List[Solution]) -> Solution:
        """Randomly select portfolio from Pareto frontier"""
        if not pareto_frontier:
            return None
        return np.random.choice(pareto_frontier)

class MDM(DecisionMaker):
    """Median Decision Maker - selects median portfolio"""
    
    def __init__(self):
        super().__init__("M-DM")
        
    def select_portfolio(self, pareto_frontier: List[Solution]) -> Solution:
        """Select median portfolio by weight vector"""
        if not pareto_frontier:
            return None
            
        # Calculate median weights across all portfolios
        weights_matrix = np.array([sol.weights for sol in pareto_frontier])
        median_weights = np.median(weights_matrix, axis=0)
        
        # Find portfolio closest to median weights
        distances = []
        for sol in pareto_frontier:
            dist = np.linalg.norm(sol.weights - median_weights)
            distances.append(dist)
        
        min_idx = np.argmin(distances)
        return pareto_frontier[min_idx]

class IBOVESPAExperiment:
    """Main experiment class for IBOVESPA ASMS-EMOA comparison"""
    
    def __init__(self, initial_wealth: float = 100000.0, transaction_cost_rate: float = 0.001):
        self.initial_wealth = initial_wealth
        self.transaction_cost_rate = transaction_cost_rate
        self.decision_makers = [HvDM(), RDM(), MDM()]
        self.results = {}
        
    def download_ibovespa_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Download IBOVESPA component stocks data"""
        logger.info("Downloading IBOVESPA data...")
        
        # IBOVESPA component stocks (top 50 by market cap)
        ibovespa_stocks = [
            'VALE3.SA', 'PETR4.SA', 'ITUB4.SA', 'BBDC4.SA', 'ABEV3.SA',
            'WEGE3.SA', 'RENT3.SA', 'LREN3.SA', 'JBSS3.SA', 'SUZB3.SA',
            'RADL3.SA', 'CCRO3.SA', 'GGBR4.SA', 'USIM5.SA', 'CSAN3.SA',
            'EMBR3.SA', 'BRFS3.SA', 'VIVT4.SA', 'TOTS3.SA', 'QUAL3.SA',
            'MGLU3.SA', 'RAIL3.SA', 'SBSP3.SA', 'CMIG4.SA', 'CPLE6.SA',
            'ELET3.SA', 'ENBR3.SA', 'EQTL3.SA', 'GNDI3.SA', 'HYPE3.SA',
            'IRBR3.SA', 'ITSA4.SA', 'KLBN4.SA', 'LAME4.SA', 'MULT3.SA',
            'NATU3.SA', 'PCAR4.SA', 'SANB11.SA', 'SAPR4.SA', 'TAEE11.SA',
            'TIMP3.SA', 'UGPA3.SA', 'USIM3.SA', 'WIZS3.SA', 'YDUQ3.SA'
        ]
        
        try:
            # Download data
            data = yf.download(ibovespa_stocks, start=start_date, end=end_date, 
                             progress=False, group_by='ticker')
            
            # Handle multi-level columns
            if isinstance(data.columns, pd.MultiIndex):
                # Get only 'Adj Close' prices
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
            # Fallback to synthetic data
            logger.info("Using synthetic IBOVESPA-like data")
            return self._generate_synthetic_ibovespa_data(start_date, end_date)
    
    def _generate_synthetic_ibovespa_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Generate synthetic IBOVESPA-like data for testing"""
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        dates = pd.date_range(start, end, freq='D')
        
        # Generate 40 stocks with realistic Brazilian market characteristics
        n_stocks = 40
        n_days = len(dates)
        
        # Base returns with Brazilian market characteristics
        np.random.seed(42)  # For reproducibility
        base_returns = np.random.normal(0.0005, 0.02, (n_days, n_stocks))  # Daily returns
        
        # Add some correlation structure
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
        # Calculate weight changes
        weight_changes = np.abs(new_weights - old_weights)
        
        # Transaction cost = sum of absolute changes * cost rate * portfolio value
        total_cost = np.sum(weight_changes) * self.transaction_cost_rate * portfolio_value
        
        return total_cost
    
    def run_experiment(self, returns_data: pd.DataFrame, 
                      historical_days: int = 120, 
                      stride_days: int = 60,
                      k_values: List[int] = [0, 1, 2, 3]) -> Dict:
        """Run the complete experiment"""
        logger.info("Starting IBOVESPA experiment...")
        
        # Initialize results structure
        for k in k_values:
            self.results[k] = {}
            for dm in self.decision_makers:
                self.results[k][dm.name] = {
                    'wealth_history': [],
                    'roi_history': [],
                    'coherence_history': [],
                    'transaction_costs': [],
                    'anticipative_factors': [],
                    'portfolio_compositions': []
                }
        
        # Calculate number of periods
        total_days = len(returns_data)
        n_periods = (total_days - historical_days) // stride_days
        
        logger.info(f"Running experiment for {n_periods} periods")
        
        for period in range(n_periods):
            logger.info(f"Processing period {period + 1}/{n_periods}")
            
            # Calculate date ranges
            start_idx = period * stride_days
            end_idx = start_idx + historical_days
            future_idx = end_idx + stride_days
            
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
                'returns_data': historical_data,
                'anticipation_horizon': 1  # K=1 for one-step ahead
            }
            
            # Run for each K value
            for k in k_values:
                algorithm_params['anticipation_horizon'] = k
                
                # Initialize ASMS-EMOA
                sms_emoa = SMSEMOA(**algorithm_params)
                
                # Run optimization
                pareto_frontier = sms_emoa.run_optimization()
                
                # Calculate anticipative factor (1-λ)
                anticipative_factor = 1.0 - (1.0 / (1.0 + k))  # Simplified formula
                
                # Calculate coherence
                coherence = self.calculate_coherence(pareto_frontier)
                
                # Test each decision maker
                for dm in self.decision_makers:
                    # Select portfolio
                    selected_portfolio = dm.select_portfolio(pareto_frontier)
                    
                    if selected_portfolio is not None:
                        # Calculate performance
                        if period == 0:
                            # First period: use initial wealth
                            current_wealth = self.initial_wealth
                            old_weights = np.zeros(num_assets)
                        else:
                            # Use previous wealth and weights
                            current_wealth = self.results[k][dm.name]['wealth_history'][-1]
                            old_weights = self.results[k][dm.name]['portfolio_compositions'][-1]
                        
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
                        self.results[k][dm.name]['wealth_history'].append(new_wealth)
                        self.results[k][dm.name]['roi_history'].append(roi)
                        self.results[k][dm.name]['coherence_history'].append(coherence)
                        self.results[k][dm.name]['transaction_costs'].append(transaction_cost)
                        self.results[k][dm.name]['anticipative_factors'].append(anticipative_factor)
                        self.results[k][dm.name]['portfolio_compositions'].append(selected_portfolio.weights.copy())
        
        logger.info("Experiment completed successfully!")
        return self.results
    
    def generate_report(self) -> str:
        """Generate comprehensive experiment report"""
        report = []
        report.append("# IBOVESPA ASMS-EMOA Experiment Report")
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Performance Summary
        report.append("## Performance Summary")
        report.append("")
        
        for k in self.results.keys():
            report.append(f"### Anticipation Horizon K = {k}")
            report.append("")
            report.append("| Decision Maker | Final ROI (%) | Final Wealth (R$) | Total Transaction Costs (R$) | Avg Coherence |")
            report.append("|----------------|---------------|-------------------|------------------------------|---------------|")
            
            for dm_name in ['Hv-DM', 'R-DM', 'M-DM']:
                if dm_name in self.results[k]:
                    data = self.results[k][dm_name]
                    final_roi = data['roi_history'][-1] * 100 if data['roi_history'] else 0
                    final_wealth = data['wealth_history'][-1] if data['wealth_history'] else self.initial_wealth
                    total_costs = sum(data['transaction_costs']) if data['transaction_costs'] else 0
                    avg_coherence = np.mean(data['coherence_history']) if data['coherence_history'] else 0
                    
                    report.append(f"| {dm_name} | {final_roi:.2f}% | R$ {final_wealth:,.2f} | R$ {total_costs:,.2f} | {avg_coherence:.3f} |")
            
            report.append("")
        
        # Statistical Analysis
        report.append("## Statistical Analysis")
        report.append("")
        
        # Compare DMs for each K
        for k in self.results.keys():
            report.append(f"### K = {k} - Decision Maker Comparison")
            report.append("")
            
            # Extract final ROIs for statistical testing
            final_rois = []
            dm_names = []
            
            for dm_name in ['Hv-DM', 'R-DM', 'M-DM']:
                if dm_name in self.results[k] and self.results[k][dm_name]['roi_history']:
                    final_rois.append(self.results[k][dm_name]['roi_history'][-1])
                    dm_names.append(dm_name)
            
            if len(final_rois) >= 2:
                # Wilcoxon test between Hv-DM and others
                if 'Hv-DM' in dm_names:
                    hv_idx = dm_names.index('Hv-DM')
                    for i, name in enumerate(dm_names):
                        if i != hv_idx:
                            try:
                                stat, p_value = wilcoxon([final_rois[hv_idx]], [final_rois[i]])
                                report.append(f"Hv-DM vs {name}: p-value = {p_value:.4f}")
                            except:
                                report.append(f"Hv-DM vs {name}: Insufficient data for statistical test")
            
            report.append("")
        
        # Key Insights
        report.append("## Key Insights")
        report.append("")
        
        # Find best performing DM for each K
        for k in self.results.keys():
            best_dm = None
            best_roi = -float('inf')
            
            for dm_name in ['Hv-DM', 'R-DM', 'M-DM']:
                if dm_name in self.results[k] and self.results[k][dm_name]['roi_history']:
                    roi = self.results[k][dm_name]['roi_history'][-1]
                    if roi > best_roi:
                        best_roi = roi
                        best_dm = dm_name
            
            if best_dm:
                report.append(f"- **K = {k}**: {best_dm} performed best with {best_roi:.2%} ROI")
        
        report.append("")
        report.append("## Conclusions")
        report.append("")
        report.append("1. **Anticipation Impact**: Higher K values show different performance patterns")
        report.append("2. **Decision Maker Comparison**: Hv-DM aims for maximal flexibility")
        report.append("3. **Transaction Costs**: Impact on net returns varies by rebalancing strategy")
        report.append("4. **Coherence Analysis**: Portfolio similarity patterns across DMs")
        
        return "\n".join(report)
    
    def create_visualizations(self, save_path: str = "ibovespa_results"):
        """Create comprehensive visualizations"""
        os.makedirs(save_path, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Wealth Evolution
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, k in enumerate(self.results.keys()):
            ax = axes[i]
            for dm_name in ['Hv-DM', 'R-DM', 'M-DM']:
                if dm_name in self.results[k]:
                    wealth_history = self.results[k][dm_name]['wealth_history']
                    ax.plot(wealth_history, label=dm_name, linewidth=2)
            
            ax.set_title(f'Wealth Evolution - K = {k}')
            ax.set_xlabel('Period')
            ax.set_ylabel('Portfolio Value (R$)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/wealth_evolution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. ROI Comparison
        fig, ax = plt.subplots(figsize=(12, 8))
        
        k_values = list(self.results.keys())
        dm_names = ['Hv-DM', 'R-DM', 'M-DM']
        x = np.arange(len(k_values))
        width = 0.25
        
        for i, dm_name in enumerate(dm_names):
            rois = []
            for k in k_values:
                if dm_name in self.results[k] and self.results[k][dm_name]['roi_history']:
                    rois.append(self.results[k][dm_name]['roi_history'][-1] * 100)
                else:
                    rois.append(0)
            
            ax.bar(x + i*width, rois, width, label=dm_name, alpha=0.8)
        
        ax.set_xlabel('Anticipation Horizon (K)')
        ax.set_ylabel('Final ROI (%)')
        ax.set_title('ROI Comparison by Decision Maker and K')
        ax.set_xticks(x + width)
        ax.set_xticklabels([f'K={k}' for k in k_values])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/roi_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Coherence Heatmap
        fig, ax = plt.subplots(figsize=(10, 6))
        
        coherence_data = []
        for k in k_values:
            row = []
            for dm_name in dm_names:
                if dm_name in self.results[k]:
                    avg_coherence = np.mean(self.results[k][dm_name]['coherence_history'])
                    row.append(avg_coherence)
                else:
                    row.append(0)
            coherence_data.append(row)
        
        sns.heatmap(coherence_data, annot=True, fmt='.3f', 
                   xticklabels=dm_names, yticklabels=[f'K={k}' for k in k_values],
                   cmap='YlOrRd', ax=ax)
        ax.set_title('Average Coherence by Decision Maker and K')
        ax.set_xlabel('Decision Maker')
        ax.set_ylabel('Anticipation Horizon (K)')
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/coherence_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Transaction Costs
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for k in k_values:
            costs = []
            for dm_name in dm_names:
                if dm_name in self.results[k]:
                    total_cost = sum(self.results[k][dm_name]['transaction_costs'])
                    costs.append(total_cost)
                else:
                    costs.append(0)
            
            x_pos = np.arange(len(dm_names))
            ax.bar(x_pos + k*0.2, costs, 0.2, label=f'K={k}', alpha=0.8)
        
        ax.set_xlabel('Decision Maker')
        ax.set_ylabel('Total Transaction Costs (R$)')
        ax.set_title('Transaction Costs by Decision Maker and K')
        ax.set_xticks(x_pos + 0.3)
        ax.set_xticklabels(dm_names)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/transaction_costs.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualizations saved to {save_path}/")

def main():
    """Main experiment execution"""
    logger.info("Starting IBOVESPA ASMS-EMOA Experiment")
    
    # Experiment parameters
    start_date = '2023-01-01'
    end_date = '2024-01-01'
    historical_days = 120
    stride_days = 60
    k_values = [0, 1, 2, 3]
    
    # Initialize experiment
    experiment = IBOVESPAExperiment(initial_wealth=100000.0, transaction_cost_rate=0.001)
    
    # Download data
    returns_data = experiment.download_ibovespa_data(start_date, end_date)
    
    # Run experiment
    results = experiment.run_experiment(returns_data, historical_days, stride_days, k_values)
    
    # Generate report
    report = experiment.generate_report()
    
    # Save report
    with open('ibovespa_experiment_report.md', 'w') as f:
        f.write(report)
    
    # Create visualizations
    experiment.create_visualizations()
    
    # Save results as JSON
    results_summary = {}
    for k in results:
        results_summary[k] = {}
        for dm_name in results[k]:
            results_summary[k][dm_name] = {
                'final_roi': results[k][dm_name]['roi_history'][-1] if results[k][dm_name]['roi_history'] else 0,
                'final_wealth': results[k][dm_name]['wealth_history'][-1] if results[k][dm_name]['wealth_history'] else 100000,
                'total_transaction_costs': sum(results[k][dm_name]['transaction_costs']),
                'avg_coherence': np.mean(results[k][dm_name]['coherence_history']) if results[k][dm_name]['coherence_history'] else 0
            }
    
    with open('ibovespa_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)
    
    logger.info("Experiment completed! Check ibovespa_experiment_report.md and ibovespa_results/ for results.")

if __name__ == "__main__":
    main() 