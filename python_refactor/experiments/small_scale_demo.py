#!/usr/bin/env python3
"""
Small Scale Experimental Demonstration of Anticipatory SMS-EMOA

This script demonstrates the anticipatory SMS-EMOA algorithm on a small dataset
to produce quick results and nice visualizations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# Add src to path
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import with absolute paths
from src.algorithms.sms_emoa import SMSEMOA
from src.algorithms.anticipatory_learning import AnticipatoryLearning
from src.portfolio.portfolio import Portfolio
from src.algorithms.solution import Solution

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class SmallScaleExperiment:
    """Small scale experimental demonstration."""
    
    def __init__(self, num_assets=5, population_size=20, generations=50):
        self.num_assets = num_assets
        self.population_size = population_size
        self.generations = generations
        self.results = {}
        
    def download_small_dataset(self):
        """Generate synthetic dataset for demonstration."""
        print("📊 Generating synthetic dataset...")
        
        # Generate synthetic returns data
        np.random.seed(42)  # For reproducibility
        
        # Create synthetic returns for 5 assets over 120 days
        n_days = 120
        n_assets = self.num_assets
        
        # Generate correlated returns
        base_returns = np.random.normal(0.001, 0.02, (n_days, n_assets))  # Daily returns
        
        # Add some correlation structure
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
        print(f"📈 Dataset shape: {returns_df.shape}")
        print(f"📅 Date range: {returns_df.index[0].date()} to {returns_df.index[-1].date()}")
        print(f"📊 Mean daily returns: {returns_df.mean().values}")
        print(f"📊 Return std: {returns_df.std().values}")
        
        return returns_df
    
    def setup_portfolio_data(self, returns_df):
        """Set up Portfolio static variables."""
        print("🔧 Setting up Portfolio data...")
        
        returns_data = returns_df.values
        n_assets = returns_data.shape[1]
        
        # Compute statistics
        Portfolio.available_assets_size = n_assets
        Portfolio.mean_ROI = np.mean(returns_data, axis=0) * 252  # Annualized
        Portfolio.median_ROI = np.median(returns_data, axis=0) * 252
        Portfolio.covariance = np.cov(returns_data.T) * 252
        Portfolio.robust_covariance = np.cov(returns_data.T) * 252  # Simplified
        Portfolio.complete_returns_data = returns_data
        
        print(f"📊 Mean ROI: {Portfolio.mean_ROI}")
        print(f"📊 ROI Std: {np.std(returns_data, axis=0) * np.sqrt(252)}")
        
        return returns_data
    
    def run_experiment(self, returns_data):
        """Run the anticipatory SMS-EMOA experiment."""
        print("🚀 Running anticipatory SMS-EMOA experiment...")
        
        # Create SMS-EMOA with anticipatory learning
        sms_emoa = SMSEMOA(
            population_size=self.population_size,
            generations=self.generations,
            crossover_rate=0.9,
            mutation_rate=0.1,
            tournament_size=3
        )
        
        # Create and set anticipatory learning
        anticipatory_learning = AnticipatoryLearning(
            learning_rate=0.01,
            prediction_horizon=1,
            monte_carlo_simulations=500,  # Reduced for speed
            adaptive_learning=True,
            window_size=10
        )
        sms_emoa.set_learning(anticipatory_learning)
        
        # Prepare data
        data = {
            'num_assets': self.num_assets,
            'returns_data': returns_data
        }
        
        # Run optimization
        print(f"🔄 Running {self.generations} generations...")
        start_time = datetime.now()
        
        population = sms_emoa.run(data)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"✅ Optimization completed in {duration:.2f} seconds")
        
        # Collect results
        self.results = {
            'population': population,
            'pareto_front': sms_emoa.get_pareto_front(),
            'hypervolume_history': sms_emoa.hypervolume_history,
            'stochastic_hypervolume_history': sms_emoa.stochastic_hypervolume_history,
            'function_evaluations': sms_emoa.get_function_evaluations(),
            'learning_metrics': anticipatory_learning.get_learning_metrics(),
            'duration': duration,
            'sms_emoa': sms_emoa,
            'anticipatory_learning': anticipatory_learning
        }
        
        return self.results
    
    def analyze_results(self):
        """Analyze and display results."""
        print("\n📊 Analyzing results...")
        
        pareto_front = self.results['pareto_front']
        hypervolume_history = self.results['hypervolume_history']
        learning_metrics = self.results['learning_metrics']
        
        print(f"🎯 Pareto Front Size: {len(pareto_front)}")
        print(f"📈 Final Hypervolume: {hypervolume_history[-1]:.6f}")
        print(f"🔄 Function Evaluations: {self.results['function_evaluations']}")
        print(f"⏱️  Duration: {self.results['duration']:.2f} seconds")
        
        if learning_metrics:
            print(f"🧠 Learning Events: {learning_metrics.get('total_learning_events', 0)}")
            print(f"📊 Mean Alpha: {learning_metrics.get('mean_alpha', 0):.4f}")
            print(f"📊 Mean Prediction Error: {learning_metrics.get('mean_prediction_error', 0):.4f}")
        
        # Pareto front analysis
        if pareto_front:
            rois = [s.P.ROI for s in pareto_front]
            risks = [s.P.risk for s in pareto_front]
            
            print(f"📊 ROI Range: {min(rois):.4f} - {max(rois):.4f}")
            print(f"📊 Risk Range: {min(risks):.4f} - {max(risks):.4f}")
    
    def create_visualizations(self):
        """Create comprehensive visualizations."""
        print("\n🎨 Creating visualizations...")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Pareto Front
        ax1 = plt.subplot(2, 3, 1)
        self._plot_pareto_front(ax1)
        
        # 2. Hypervolume Evolution
        ax2 = plt.subplot(2, 3, 2)
        self._plot_hypervolume_evolution(ax2)
        
        # 3. Population Distribution
        ax3 = plt.subplot(2, 3, 3)
        self._plot_population_distribution(ax3)
        
        # 4. Learning Metrics
        ax4 = plt.subplot(2, 3, 4)
        self._plot_learning_metrics(ax4)
        
        # 5. Portfolio Weights
        ax5 = plt.subplot(2, 3, 5)
        self._plot_portfolio_weights(ax5)
        
        # 6. Convergence Analysis
        ax6 = plt.subplot(2, 3, 6)
        self._plot_convergence_analysis(ax6)
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"small_scale_experiment_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"💾 Saved visualization: {filename}")
        
        plt.show()
    
    def _plot_pareto_front(self, ax):
        """Plot Pareto front."""
        pareto_front = self.results['pareto_front']
        population = self.results['population']
        
        # Plot all solutions
        all_rois = [s.P.ROI for s in population]
        all_risks = [s.P.risk for s in population]
        ax.scatter(all_risks, all_rois, alpha=0.3, color='lightblue', label='Population')
        
        # Plot Pareto front
        if pareto_front:
            pareto_rois = [s.P.ROI for s in pareto_front]
            pareto_risks = [s.P.risk for s in pareto_front]
            ax.scatter(pareto_risks, pareto_rois, color='red', s=100, label='Pareto Front', zorder=5)
        
        ax.set_xlabel('Risk')
        ax.set_ylabel('ROI')
        ax.set_title('Pareto Front')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_hypervolume_evolution(self, ax):
        """Plot hypervolume evolution."""
        hypervolume_history = self.results['hypervolume_history']
        stochastic_history = self.results['stochastic_hypervolume_history']
        
        generations = range(len(hypervolume_history))
        ax.plot(generations, hypervolume_history, 'b-', label='Hypervolume', linewidth=2)
        
        if stochastic_history:
            ax.plot(generations, stochastic_history, 'r--', label='Expected Future Hypervolume', linewidth=2)
        
        ax.set_xlabel('Generation')
        ax.set_ylabel('Hypervolume')
        ax.set_title('Hypervolume Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_population_distribution(self, ax):
        """Plot population distribution."""
        population = self.results['population']
        
        rois = [s.P.ROI for s in population]
        risks = [s.P.risk for s in population]
        
        ax.hist2d(risks, rois, bins=15, cmap='Blues', alpha=0.7)
        ax.set_xlabel('Risk')
        ax.set_ylabel('ROI')
        ax.set_title('Population Distribution')
        ax.grid(True, alpha=0.3)
    
    def _plot_learning_metrics(self, ax):
        """Plot learning metrics."""
        learning_metrics = self.results['learning_metrics']
        
        if not learning_metrics:
            ax.text(0.5, 0.5, 'No Learning Data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Learning Metrics')
            return
        
        # Create bar plot of key metrics
        metrics = ['mean_alpha', 'mean_prediction_error', 'mean_state_quality']
        values = [learning_metrics.get(m, 0) for m in metrics]
        labels = ['Mean Alpha', 'Mean Prediction Error', 'Mean State Quality']
        
        bars = ax.bar(labels, values, color=['skyblue', 'lightcoral', 'lightgreen'])
        ax.set_ylabel('Value')
        ax.set_title('Learning Metrics')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                   f'{value:.3f}', ha='center', va='bottom')
    
    def _plot_portfolio_weights(self, ax):
        """Plot portfolio weights for Pareto front solutions."""
        pareto_front = self.results['pareto_front']
        
        if not pareto_front:
            ax.text(0.5, 0.5, 'No Pareto Front', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Portfolio Weights')
            return
        
        # Plot weights for first few Pareto solutions
        num_solutions = min(3, len(pareto_front))
        x = np.arange(self.num_assets)
        width = 0.8 / num_solutions
        
        for i in range(num_solutions):
            solution = pareto_front[i]
            weights = solution.P.investment
            ax.bar(x + i * width, weights, width, 
                   label=f'Solution {i+1} (ROI: {solution.P.ROI:.3f})', alpha=0.7)
        
        ax.set_xlabel('Asset Index')
        ax.set_ylabel('Weight')
        ax.set_title('Portfolio Weights (Pareto Solutions)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_convergence_analysis(self, ax):
        """Plot convergence analysis."""
        hypervolume_history = self.results['hypervolume_history']
        
        if len(hypervolume_history) < 2:
            ax.text(0.5, 0.5, 'Insufficient Data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Convergence Analysis')
            return
        
        # Calculate improvement rate
        improvements = np.diff(hypervolume_history)
        generations = range(1, len(hypervolume_history))
        
        ax.plot(generations, improvements, 'g-', linewidth=2)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax.set_xlabel('Generation')
        ax.set_ylabel('Hypervolume Improvement')
        ax.set_title('Convergence Analysis')
        ax.grid(True, alpha=0.3)
    
    def run_full_experiment(self):
        """Run the complete experiment."""
        print("🧪 Starting Small Scale Experimental Demonstration")
        print("=" * 60)
        
        # Download data
        returns_df = self.download_small_dataset()
        
        # Setup portfolio data
        returns_data = self.setup_portfolio_data(returns_df)
        
        # Run experiment
        results = self.run_experiment(returns_data)
        
        # Analyze results
        self.analyze_results()
        
        # Create visualizations
        self.create_visualizations()
        
        print("\n🎉 Experiment completed successfully!")
        return results


def main():
    """Main function to run the experiment."""
    # Create and run experiment
    experiment = SmallScaleExperiment(
        num_assets=5,
        population_size=20,
        generations=50
    )
    
    results = experiment.run_full_experiment()
    return results


if __name__ == "__main__":
    main() 