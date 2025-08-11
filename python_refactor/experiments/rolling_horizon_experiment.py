#!/usr/bin/env python3
"""
Rolling Horizon Experimental Setup for Anticipatory SMS-EMOA

This script implements the rolling horizon approach described in the thesis:
- 50 days of data for training
- Hold portfolios for 30 days
- Update data every 30 days
- Real asset data with proper risk calculation
- Dynamic alpha learning
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import yfinance as yf
import warnings
import ssl
warnings.filterwarnings('ignore')

# Disable SSL verification for yfinance
ssl._create_default_https_context = ssl._create_unverified_context

# Add src to path
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.algorithms.sms_emoa import SMSEMOA
from src.algorithms.anticipatory_learning import AnticipatoryLearning
from src.portfolio.portfolio import Portfolio
from src.algorithms.solution import Solution

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class RollingHorizonExperiment:
    """Rolling horizon experimental setup as described in the thesis."""
    
    def __init__(self, num_assets=5, population_size=30, generations=100, 
                 training_window=50, holding_period=30):
        self.num_assets = num_assets
        self.population_size = population_size
        self.generations = generations
        self.training_window = training_window  # 50 days as in thesis
        self.holding_period = holding_period    # 30 days as in thesis
        self.results = {
            'rolling_periods': [],
            'pareto_fronts': [],
            'hypervolume_history': [],
            'alpha_history': [],
            'prediction_errors': [],
            'portfolio_performance': []
        }
        
    def download_real_data(self):
        """Download real market data for major stocks."""
        print("📊 Downloading real market data...")
        
        # Use major stocks that are likely to have good data
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'][:self.num_assets]
        
        # Download 2 years of data to have enough for rolling windows
        end_date = datetime.now()
        start_date = end_date - timedelta(days=730)  # 2 years
        
        data = {}
        successful_tickers = []
        
        for ticker in tickers:
            try:
                stock = yf.download(ticker, start=start_date, end=end_date, progress=False)
                if not stock.empty and len(stock) > 100:  # Ensure sufficient data
                    data[ticker] = stock['Adj Close']
                    successful_tickers.append(ticker)
                    print(f"✅ Downloaded {ticker}: {len(stock)} days")
                else:
                    print(f"❌ Insufficient data for {ticker}")
            except Exception as e:
                print(f"❌ Error downloading {ticker}: {e}")
        
        if len(successful_tickers) < 3:
            print("⚠️  Insufficient data, falling back to synthetic data")
            return self._generate_synthetic_data()
        
        # Create returns dataframe
        df = pd.DataFrame(data)
        returns_df = df.pct_change().dropna()
        
        print(f"📈 Real dataset shape: {returns_df.shape}")
        print(f"📅 Date range: {returns_df.index[0].date()} to {returns_df.index[-1].date()}")
        print(f"📊 Assets: {list(returns_df.columns)}")
        
        return returns_df
    
    def _generate_synthetic_data(self):
        """Generate synthetic data as fallback."""
        print("📊 Generating synthetic data as fallback...")
        
        np.random.seed(42)
        n_days = 730  # 2 years
        n_assets = self.num_assets
        
        # Generate correlated returns with realistic parameters
        base_returns = np.random.normal(0.0005, 0.02, (n_days, n_assets))
        
        # Realistic correlation matrix
        correlation_matrix = np.array([
            [1.0, 0.3, 0.2, 0.1, 0.05],
            [0.3, 1.0, 0.4, 0.2, 0.1],
            [0.2, 0.4, 1.0, 0.3, 0.15],
            [0.1, 0.2, 0.3, 1.0, 0.25],
            [0.05, 0.1, 0.15, 0.25, 1.0]
        ])[:n_assets, :n_assets]
        
        # Apply correlation
        L = np.linalg.cholesky(correlation_matrix)
        correlated_returns = base_returns @ L.T
        
        # Create date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=n_days-1)
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Create returns dataframe
        asset_names = [f'Asset_{i+1}' for i in range(n_assets)]
        returns_df = pd.DataFrame(correlated_returns, index=date_range, columns=asset_names)
        
        print(f"✅ Generated synthetic data for {n_assets} assets")
        return returns_df
    
    def setup_rolling_windows(self, returns_df):
        """Set up rolling windows as per thesis specification."""
        print("🔄 Setting up rolling windows...")
        
        total_days = len(returns_df)
        windows = []
        
        # Calculate number of complete windows
        window_start = 0
        while window_start + self.training_window + self.holding_period <= total_days:
            training_end = window_start + self.training_window
            holding_end = training_end + self.holding_period
            
            window = {
                'training_start': window_start,
                'training_end': training_end,
                'holding_start': training_end,
                'holding_end': holding_end,
                'training_data': returns_df.iloc[window_start:training_end],
                'holding_data': returns_df.iloc[training_end:holding_end],
                'period': len(windows) + 1
            }
            
            windows.append(window)
            window_start += self.holding_period  # Slide by holding period
        
        print(f"📊 Created {len(windows)} rolling windows")
        print(f"📅 Each window: {self.training_window} days training + {self.holding_period} days holding")
        
        return windows
    
    def run_rolling_horizon_experiment(self, windows):
        """Run the complete rolling horizon experiment."""
        print("🚀 Starting Rolling Horizon Experiment")
        print("=" * 60)
        
        for i, window in enumerate(windows):
            print(f"\n📊 Rolling Period {i+1}/{len(windows)}")
            print(f"📅 Training: {window['training_data'].index[0].date()} to {window['training_data'].index[-1].date()}")
            print(f"📅 Holding: {window['holding_data'].index[0].date()} to {window['holding_data'].index[-1].date()}")
            
            # Run optimization for this window
            period_results = self._run_window_optimization(window, i)
            
            # Store results
            self.results['rolling_periods'].append(period_results)
            
            print(f"✅ Period {i+1} completed")
        
        print(f"\n🎉 Rolling horizon experiment completed!")
        print(f"📊 Total periods: {len(self.results['rolling_periods'])}")
        
        return self.results
    
    def _run_window_optimization(self, window, period_idx):
        """Run optimization for a single rolling window."""
        # Setup portfolio data for this window
        training_data = window['training_data'].values
        self._setup_portfolio_data(training_data)
        
        # Create SMS-EMOA with enhanced anticipatory learning
        sms_emoa = SMSEMOA(
            population_size=self.population_size,
            generations=self.generations,
            crossover_rate=0.9,
            mutation_rate=0.1,
            tournament_size=3
        )
        
        # Create anticipatory learning with dynamic alpha
        anticipatory_learning = AnticipatoryLearning(
            learning_rate=0.01,
            prediction_horizon=1,
            monte_carlo_simulations=1000,
            adaptive_learning=True,
            window_size=10
        )
        sms_emoa.set_learning(anticipatory_learning)
        
        # Prepare data
        data = {
            'num_assets': self.num_assets,
            'returns_data': training_data
        }
        
        # Run optimization
        start_time = datetime.now()
        population = sms_emoa.run(data)
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Get Pareto front
        pareto_front = sms_emoa.get_pareto_front()
        
        # Evaluate on holding period
        holding_performance = self._evaluate_holding_performance(pareto_front, window['holding_data'])
        
        # Collect learning metrics
        learning_metrics = anticipatory_learning.get_learning_metrics()
        
        # Store period results
        period_results = {
            'period': period_idx + 1,
            'training_data': window['training_data'],
            'holding_data': window['holding_data'],
            'population': population,
            'pareto_front': pareto_front,
            'hypervolume_history': sms_emoa.hypervolume_history.copy(),
            'stochastic_hypervolume_history': sms_emoa.stochastic_hypervolume_history.copy(),
            'learning_metrics': learning_metrics,
            'holding_performance': holding_performance,
            'duration': duration,
            'sms_emoa': sms_emoa,
            'anticipatory_learning': anticipatory_learning
        }
        
        return period_results
    
    def _setup_portfolio_data(self, returns_data):
        """Set up Portfolio static variables for current window."""
        n_assets = returns_data.shape[1]
        
        # Compute statistics
        Portfolio.available_assets_size = n_assets
        Portfolio.mean_ROI = np.mean(returns_data, axis=0) * 252  # Annualized
        Portfolio.median_ROI = np.median(returns_data, axis=0) * 252
        Portfolio.covariance = np.cov(returns_data.T) * 252
        Portfolio.robust_covariance = np.cov(returns_data.T) * 252
        Portfolio.complete_returns_data = returns_data
    
    def _evaluate_holding_performance(self, pareto_front, holding_data):
        """Evaluate Pareto front performance on holding period."""
        if not pareto_front:
            return {}
        
        holding_returns = holding_data.values
        n_days = len(holding_returns)
        
        performance = {}
        
        for i, solution in enumerate(pareto_front):
            weights = solution.P.investment
            
            # Calculate portfolio returns for holding period
            portfolio_returns = np.sum(holding_returns * weights, axis=1)
            
            # Calculate performance metrics
            total_return = np.prod(1 + portfolio_returns) - 1
            annualized_return = (1 + total_return) ** (252 / n_days) - 1
            volatility = np.std(portfolio_returns) * np.sqrt(252)
            sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
            max_drawdown = self._calculate_max_drawdown(portfolio_returns)
            
            performance[f'solution_{i}'] = {
                'weights': weights,
                'total_return': total_return,
                'annualized_return': annualized_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'roi': solution.P.ROI,
                'risk': solution.P.risk
            }
        
        return performance
    
    def _calculate_max_drawdown(self, returns):
        """Calculate maximum drawdown from returns."""
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return np.min(drawdown)
    
    def analyze_results(self):
        """Analyze and display comprehensive results."""
        print("\n📊 Analyzing Rolling Horizon Results")
        print("=" * 50)
        
        periods = self.results['rolling_periods']
        print(f"📈 Total Periods: {len(periods)}")
        
        # Aggregate metrics across periods
        all_hypervolumes = []
        all_alphas = []
        all_prediction_errors = []
        all_sharpe_ratios = []
        
        for period in periods:
            # Hypervolume
            if period['hypervolume_history']:
                all_hypervolumes.append(period['hypervolume_history'][-1])
            
            # Learning metrics
            learning_metrics = period['learning_metrics']
            if learning_metrics:
                all_alphas.append(learning_metrics.get('mean_alpha', 0))
                all_prediction_errors.append(learning_metrics.get('mean_prediction_error', 0))
            
            # Performance metrics
            holding_perf = period['holding_performance']
            for solution_perf in holding_perf.values():
                all_sharpe_ratios.append(solution_perf['sharpe_ratio'])
        
        # Summary statistics
        if all_hypervolumes:
            print(f"📊 Final Hypervolume: {np.mean(all_hypervolumes):.4f} ± {np.std(all_hypervolumes):.4f}")
        if all_alphas:
            print(f"🧠 Mean Alpha: {np.mean(all_alphas):.4f} ± {np.std(all_alphas):.4f}")
        if all_prediction_errors:
            print(f"📊 Mean Prediction Error: {np.mean(all_prediction_errors):.4f} ± {np.std(all_prediction_errors):.4f}")
        if all_sharpe_ratios:
            print(f"📈 Mean Sharpe Ratio: {np.mean(all_sharpe_ratios):.4f} ± {np.std(all_sharpe_ratios):.4f}")
    
    def create_comprehensive_visualizations(self):
        """Create comprehensive visualizations for rolling horizon results."""
        print("\n🎨 Creating comprehensive visualizations...")
        
        periods = self.results['rolling_periods']
        if not periods:
            print("❌ No data to visualize")
            return
        
        # Create large figure with multiple subplots
        fig = plt.figure(figsize=(24, 16))
        
        # 1. Hypervolume Evolution Across Periods
        ax1 = plt.subplot(3, 4, 1)
        self._plot_hypervolume_evolution(ax1, periods)
        
        # 2. Alpha Learning Rate Evolution
        ax2 = plt.subplot(3, 4, 2)
        self._plot_alpha_evolution(ax2, periods)
        
        # 3. Prediction Error Evolution
        ax3 = plt.subplot(3, 4, 3)
        self._plot_prediction_error_evolution(ax3, periods)
        
        # 4. Sharpe Ratio Performance
        ax4 = plt.subplot(3, 4, 4)
        self._plot_sharpe_ratio_performance(ax4, periods)
        
        # 5. Pareto Front Evolution
        ax5 = plt.subplot(3, 4, 5)
        self._plot_pareto_front_evolution(ax5, periods)
        
        # 6. Portfolio Weights Evolution
        ax6 = plt.subplot(3, 4, 6)
        self._plot_portfolio_weights_evolution(ax6, periods)
        
        # 7. Risk-Return Scatter (All Periods)
        ax7 = plt.subplot(3, 4, 7)
        self._plot_risk_return_scatter(ax7, periods)
        
        # 8. Learning Events Distribution
        ax8 = plt.subplot(3, 4, 8)
        self._plot_learning_events_distribution(ax8, periods)
        
        # 9. Convergence Analysis
        ax9 = plt.subplot(3, 4, 9)
        self._plot_convergence_analysis(ax9, periods)
        
        # 10. Performance Comparison
        ax10 = plt.subplot(3, 4, 10)
        self._plot_performance_comparison(ax10, periods)
        
        # 11. Volatility Analysis
        ax11 = plt.subplot(3, 4, 11)
        self._plot_volatility_analysis(ax11, periods)
        
        # 12. Summary Statistics
        ax12 = plt.subplot(3, 4, 12)
        self._plot_summary_statistics(ax12, periods)
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"rolling_horizon_experiment_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"💾 Saved visualization: {filename}")
        
        plt.show()
    
    def _plot_hypervolume_evolution(self, ax, periods):
        """Plot hypervolume evolution across periods."""
        for i, period in enumerate(periods):
            if period['hypervolume_history']:
                ax.plot(period['hypervolume_history'], 
                       label=f'Period {i+1}', alpha=0.7)
        
        ax.set_xlabel('Generation')
        ax.set_ylabel('Hypervolume')
        ax.set_title('Hypervolume Evolution')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
    
    def _plot_alpha_evolution(self, ax, periods):
        """Plot alpha learning rate evolution."""
        alphas = []
        for period in periods:
            learning_metrics = period['learning_metrics']
            if learning_metrics:
                alphas.append(learning_metrics.get('mean_alpha', 0))
        
        if alphas:
            ax.plot(alphas, 'b-o', linewidth=2, markersize=6)
            ax.set_xlabel('Period')
            ax.set_ylabel('Mean Alpha')
            ax.set_title('Dynamic Alpha Evolution')
            ax.grid(True, alpha=0.3)
    
    def _plot_prediction_error_evolution(self, ax, periods):
        """Plot prediction error evolution."""
        errors = []
        for period in periods:
            learning_metrics = period['learning_metrics']
            if learning_metrics:
                errors.append(learning_metrics.get('mean_prediction_error', 0))
        
        if errors:
            ax.plot(errors, 'r-o', linewidth=2, markersize=6)
            ax.set_xlabel('Period')
            ax.set_ylabel('Mean Prediction Error')
            ax.set_title('Prediction Error Evolution')
            ax.grid(True, alpha=0.3)
    
    def _plot_sharpe_ratio_performance(self, ax, periods):
        """Plot Sharpe ratio performance across periods."""
        all_sharpe_ratios = []
        for period in periods:
            holding_perf = period['holding_performance']
            for solution_perf in holding_perf.values():
                all_sharpe_ratios.append(solution_perf['sharpe_ratio'])
        
        if all_sharpe_ratios:
            ax.hist(all_sharpe_ratios, bins=15, alpha=0.7, color='green')
            ax.axvline(np.mean(all_sharpe_ratios), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(all_sharpe_ratios):.3f}')
            ax.set_xlabel('Sharpe Ratio')
            ax.set_ylabel('Frequency')
            ax.set_title('Sharpe Ratio Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    def _plot_pareto_front_evolution(self, ax, periods):
        """Plot Pareto front evolution."""
        for i, period in enumerate(periods):
            pareto_front = period['pareto_front']
            if pareto_front:
                rois = [s.P.ROI for s in pareto_front]
                risks = [s.P.risk for s in pareto_front]
                ax.scatter(risks, rois, alpha=0.6, label=f'Period {i+1}')
        
        ax.set_xlabel('Risk (Std Dev)')
        ax.set_ylabel('ROI')
        ax.set_title('Pareto Front Evolution')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
    
    def _plot_portfolio_weights_evolution(self, ax, periods):
        """Plot portfolio weights evolution."""
        # Show weights for first Pareto solution across periods
        weights_evolution = []
        for period in periods:
            pareto_front = period['pareto_front']
            if pareto_front:
                weights_evolution.append(pareto_front[0].P.investment)
        
        if weights_evolution:
            weights_array = np.array(weights_evolution)
            im = ax.imshow(weights_array.T, aspect='auto', cmap='viridis')
            ax.set_xlabel('Period')
            ax.set_ylabel('Asset')
            ax.set_title('Portfolio Weights Evolution')
            plt.colorbar(im, ax=ax)
    
    def _plot_risk_return_scatter(self, ax, periods):
        """Plot risk-return scatter for all periods."""
        all_rois = []
        all_risks = []
        
        for period in periods:
            pareto_front = period['pareto_front']
            for solution in pareto_front:
                all_rois.append(solution.P.ROI)
                all_risks.append(solution.P.risk)
        
        if all_rois:
            ax.scatter(all_risks, all_rois, alpha=0.6, color='blue')
            ax.set_xlabel('Risk (Std Dev)')
            ax.set_ylabel('ROI')
            ax.set_title('Risk-Return Scatter (All Periods)')
            ax.grid(True, alpha=0.3)
    
    def _plot_learning_events_distribution(self, ax, periods):
        """Plot learning events distribution."""
        learning_events = []
        for period in periods:
            learning_metrics = period['learning_metrics']
            if learning_metrics:
                events = learning_metrics.get('total_learning_events', 0)
                learning_events.append(events)
        
        if learning_events:
            ax.bar(range(len(learning_events)), learning_events, alpha=0.7)
            ax.set_xlabel('Period')
            ax.set_ylabel('Learning Events')
            ax.set_title('Learning Events Distribution')
            ax.grid(True, alpha=0.3)
    
    def _plot_convergence_analysis(self, ax, periods):
        """Plot convergence analysis."""
        convergence_rates = []
        for period in periods:
            if len(period['hypervolume_history']) > 1:
                # Calculate improvement rate
                improvements = np.diff(period['hypervolume_history'])
                convergence_rate = np.mean(improvements[-10:]) if len(improvements) >= 10 else np.mean(improvements)
                convergence_rates.append(convergence_rate)
        
        if convergence_rates:
            ax.plot(convergence_rates, 'g-o', linewidth=2, markersize=6)
            ax.set_xlabel('Period')
            ax.set_ylabel('Convergence Rate')
            ax.set_title('Convergence Analysis')
            ax.grid(True, alpha=0.3)
    
    def _plot_performance_comparison(self, ax, periods):
        """Plot performance comparison."""
        returns = []
        volatilities = []
        
        for period in periods:
            holding_perf = period['holding_performance']
            for solution_perf in holding_perf.values():
                returns.append(solution_perf['annualized_return'])
                volatilities.append(solution_perf['volatility'])
        
        if returns:
            ax.scatter(volatilities, returns, alpha=0.6, color='purple')
            ax.set_xlabel('Volatility')
            ax.set_ylabel('Annualized Return')
            ax.set_title('Performance Comparison')
            ax.grid(True, alpha=0.3)
    
    def _plot_volatility_analysis(self, ax, periods):
        """Plot volatility analysis."""
        volatilities = []
        for period in periods:
            holding_perf = period['holding_performance']
            for solution_perf in holding_perf.values():
                volatilities.append(solution_perf['volatility'])
        
        if volatilities:
            ax.hist(volatilities, bins=15, alpha=0.7, color='orange')
            ax.axvline(np.mean(volatilities), color='red', linestyle='--',
                      label=f'Mean: {np.mean(volatilities):.3f}')
            ax.set_xlabel('Volatility')
            ax.set_ylabel('Frequency')
            ax.set_title('Volatility Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    def _plot_summary_statistics(self, ax, periods):
        """Plot summary statistics."""
        ax.axis('off')
        
        # Calculate summary statistics
        total_periods = len(periods)
        total_solutions = sum(len(p['pareto_front']) for p in periods)
        avg_hypervolume = np.mean([p['hypervolume_history'][-1] for p in periods if p['hypervolume_history']])
        avg_alpha = np.mean([p['learning_metrics'].get('mean_alpha', 0) for p in periods if p['learning_metrics']])
        
        summary_text = f"""
        Rolling Horizon Experiment Summary
        
        Total Periods: {total_periods}
        Total Solutions: {total_solutions}
        Avg Hypervolume: {avg_hypervolume:.4f}
        Avg Alpha: {avg_alpha:.4f}
        
        Training Window: {self.training_window} days
        Holding Period: {self.holding_period} days
        Population Size: {self.population_size}
        Generations: {self.generations}
        """
        
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, 
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    def run_full_experiment(self):
        """Run the complete rolling horizon experiment."""
        print("🧪 Starting Rolling Horizon Experimental Setup")
        print("=" * 60)
        
        # Download data
        returns_df = self.download_real_data()
        
        # Setup rolling windows
        windows = self.setup_rolling_windows(returns_df)
        
        # Run experiment
        results = self.run_rolling_horizon_experiment(windows)
        
        # Analyze results
        self.analyze_results()
        
        # Create visualizations
        self.create_comprehensive_visualizations()
        
        print("\n🎉 Rolling horizon experiment completed successfully!")
        return results


def main():
    """Main function to run the rolling horizon experiment."""
    # Create and run experiment
    experiment = RollingHorizonExperiment(
        num_assets=5,
        population_size=30,
        generations=100,
        training_window=50,  # As per thesis
        holding_period=30    # As per thesis
    )
    
    results = experiment.run_full_experiment()
    return results


if __name__ == "__main__":
    main() 