#!/usr/bin/env python3
"""
Traditional Portfolio Benchmarks
Implements standard portfolio optimization strategies for comparison
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TraditionalBenchmarks:
    """Traditional portfolio optimization benchmarks"""
    
    def __init__(self, returns_data):
        self.returns_data = returns_data
        self.n_assets = returns_data.shape[1]
        
    def equal_weighted_portfolio(self):
        """Equal-weighted portfolio (1/N strategy)"""
        weights = np.ones(self.n_assets) / self.n_assets
        return weights
    
    def minimum_variance_portfolio(self, historical_data):
        """Minimum variance portfolio optimization"""
        
        # Calculate covariance matrix
        cov_matrix = historical_data.cov().values
        
        # Objective function: minimize portfolio variance
        def objective(weights):
            portfolio_variance = weights.T @ cov_matrix @ weights
            return portfolio_variance
        
        # Constraints: weights sum to 1
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        
        # Bounds: weights between 0 and 1 (no short selling)
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        
        # Initial guess: equal weights
        initial_weights = np.ones(self.n_assets) / self.n_assets
        
        # Optimize
        result = minimize(objective, initial_weights, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        
        if result.success:
            return result.x
        else:
            logger.warning("Minimum variance optimization failed, using equal weights")
            return self.equal_weighted_portfolio()
    
    def sharpe_optimal_portfolio(self, historical_data, risk_free_rate=0.02):
        """Sharpe ratio optimal portfolio (maximum Sharpe ratio)"""
        
        # Calculate expected returns and covariance
        expected_returns = historical_data.mean().values
        cov_matrix = historical_data.cov().values
        
        # Objective function: maximize Sharpe ratio (minimize negative Sharpe)
        def objective(weights):
            portfolio_return = np.sum(weights * expected_returns)
            portfolio_volatility = np.sqrt(weights.T @ cov_matrix @ weights)
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
            return -sharpe_ratio  # Minimize negative Sharpe
        
        # Constraints: weights sum to 1
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        
        # Bounds: weights between 0 and 1 (no short selling)
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        
        # Initial guess: equal weights
        initial_weights = np.ones(self.n_assets) / self.n_assets
        
        # Optimize
        result = minimize(objective, initial_weights, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        
        if result.success:
            return result.x
        else:
            logger.warning("Sharpe optimization failed, using equal weights")
            return self.equal_weighted_portfolio()
    
    def maximum_diversification_portfolio(self, historical_data):
        """Maximum diversification portfolio"""
        
        # Calculate covariance matrix
        cov_matrix = historical_data.cov().values
        
        # Calculate individual asset volatilities
        volatilities = np.sqrt(np.diag(cov_matrix))
        
        # Objective function: maximize diversification ratio
        def objective(weights):
            portfolio_volatility = np.sqrt(weights.T @ cov_matrix @ weights)
            weighted_volatility = np.sum(weights * volatilities)
            diversification_ratio = weighted_volatility / portfolio_volatility
            return -diversification_ratio  # Minimize negative diversification
        
        # Constraints: weights sum to 1
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        
        # Bounds: weights between 0 and 1
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        
        # Initial guess: equal weights
        initial_weights = np.ones(self.n_assets) / self.n_assets
        
        # Optimize
        result = minimize(objective, initial_weights, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        
        if result.success:
            return result.x
        else:
            logger.warning("Maximum diversification optimization failed, using equal weights")
            return self.equal_weighted_portfolio()
    
    def risk_parity_portfolio(self, historical_data):
        """Risk parity portfolio (equal risk contribution)"""
        
        # Calculate covariance matrix
        cov_matrix = historical_data.cov().values
        
        # Objective function: minimize variance of risk contributions
        def objective(weights):
            portfolio_volatility = np.sqrt(weights.T @ cov_matrix @ weights)
            risk_contributions = (weights * (cov_matrix @ weights)) / portfolio_volatility
            risk_contribution_variance = np.var(risk_contributions)
            return risk_contribution_variance
        
        # Constraints: weights sum to 1
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        
        # Bounds: weights between 0 and 1
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        
        # Initial guess: equal weights
        initial_weights = np.ones(self.n_assets) / self.n_assets
        
        # Optimize
        result = minimize(objective, initial_weights, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        
        if result.success:
            return result.x
        else:
            logger.warning("Risk parity optimization failed, using equal weights")
            return self.equal_weighted_portfolio()
    
    def calculate_portfolio_metrics(self, weights, historical_data):
        """Calculate portfolio performance metrics"""
        
        # Calculate expected returns and covariance
        expected_returns = historical_data.mean().values
        cov_matrix = historical_data.cov().values
        
        # Portfolio metrics
        portfolio_return = np.sum(weights * expected_returns)
        portfolio_volatility = np.sqrt(weights.T @ cov_matrix @ weights)
        sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0
        
        # Maximum drawdown simulation
        cumulative_returns = (1 + historical_data @ weights).cumprod()
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # Value at Risk (95% confidence)
        portfolio_returns = historical_data @ weights
        var_95 = np.percentile(portfolio_returns, 5)
        
        # Conditional Value at Risk (Expected Shortfall)
        cvar_95 = np.mean(portfolio_returns[portfolio_returns <= var_95])
        
        return {
            'return': portfolio_return,
            'volatility': portfolio_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'weights': weights
        }
    
    def run_all_benchmarks(self, historical_data):
        """Run all traditional benchmarks"""
        
        logger.info("Running traditional benchmarks...")
        
        benchmarks = {}
        
        # 1. Equal-weighted portfolio
        logger.info("Calculating equal-weighted portfolio...")
        eq_weights = self.equal_weighted_portfolio()
        benchmarks['Equal_Weighted'] = self.calculate_portfolio_metrics(eq_weights, historical_data)
        
        # 2. Minimum variance portfolio
        logger.info("Calculating minimum variance portfolio...")
        min_var_weights = self.minimum_variance_portfolio(historical_data)
        benchmarks['Minimum_Variance'] = self.calculate_portfolio_metrics(min_var_weights, historical_data)
        
        # 3. Sharpe optimal portfolio
        logger.info("Calculating Sharpe optimal portfolio...")
        sharpe_weights = self.sharpe_optimal_portfolio(historical_data)
        benchmarks['Sharpe_Optimal'] = self.calculate_portfolio_metrics(sharpe_weights, historical_data)
        
        # 4. Maximum diversification portfolio
        logger.info("Calculating maximum diversification portfolio...")
        max_div_weights = self.maximum_diversification_portfolio(historical_data)
        benchmarks['Maximum_Diversification'] = self.calculate_portfolio_metrics(max_div_weights, historical_data)
        
        # 5. Risk parity portfolio
        logger.info("Calculating risk parity portfolio...")
        risk_parity_weights = self.risk_parity_portfolio(historical_data)
        benchmarks['Risk_Parity'] = self.calculate_portfolio_metrics(risk_parity_weights, historical_data)
        
        return benchmarks
    
    def create_benchmark_report(self, benchmarks):
        """Create a comprehensive benchmark report"""
        
        report = []
        report.append("# Traditional Portfolio Benchmarks Report")
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Performance comparison table
        report.append("## Performance Comparison")
        report.append("")
        report.append("| Strategy | Return (%) | Volatility (%) | Sharpe Ratio | Max Drawdown (%) | VaR (95%) | CVaR (95%) |")
        report.append("|----------|------------|----------------|--------------|------------------|-----------|------------|")
        
        for strategy, metrics in benchmarks.items():
            report.append(f"| {strategy} | {metrics['return']*100:.2f} | {metrics['volatility']*100:.2f} | "
                         f"{metrics['sharpe_ratio']:.3f} | {metrics['max_drawdown']*100:.2f} | "
                         f"{metrics['var_95']*100:.2f} | {metrics['cvar_95']*100:.2f} |")
        
        report.append("")
        
        # Portfolio weights
        report.append("## Portfolio Weights")
        report.append("")
        
        for strategy, metrics in benchmarks.items():
            report.append(f"### {strategy}")
            report.append("")
            weights_str = ", ".join([f"{w:.3f}" for w in metrics['weights']])
            report.append(f"Weights: [{weights_str}]")
            report.append("")
        
        return '\n'.join(report)
    
    def create_benchmark_visualizations(self, benchmarks, save_dir="benchmark_results"):
        """Create visualizations for benchmark comparison"""
        
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. Performance comparison bar chart
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Traditional Portfolio Benchmarks Comparison', fontsize=16)
        
        strategies = list(benchmarks.keys())
        returns = [benchmarks[s]['return'] * 100 for s in strategies]
        volatilities = [benchmarks[s]['volatility'] * 100 for s in strategies]
        sharpe_ratios = [benchmarks[s]['sharpe_ratio'] for s in strategies]
        max_drawdowns = [benchmarks[s]['max_drawdown'] * 100 for s in strategies]
        
        # Returns
        axes[0, 0].bar(strategies, returns, color='blue', alpha=0.7)
        axes[0, 0].set_title('Annualized Returns (%)')
        axes[0, 0].set_ylabel('Return (%)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Volatilities
        axes[0, 1].bar(strategies, volatilities, color='red', alpha=0.7)
        axes[0, 1].set_title('Annualized Volatility (%)')
        axes[0, 1].set_ylabel('Volatility (%)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Sharpe Ratios
        axes[1, 0].bar(strategies, sharpe_ratios, color='green', alpha=0.7)
        axes[1, 0].set_title('Sharpe Ratio')
        axes[1, 0].set_ylabel('Sharpe Ratio')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Maximum Drawdowns
        axes[1, 1].bar(strategies, max_drawdowns, color='orange', alpha=0.7)
        axes[1, 1].set_title('Maximum Drawdown (%)')
        axes[1, 1].set_ylabel('Drawdown (%)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/benchmark_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Risk-return scatter plot
        plt.figure(figsize=(10, 8))
        for strategy in strategies:
            plt.scatter(benchmarks[strategy]['volatility'] * 100, 
                       benchmarks[strategy]['return'] * 100, 
                       s=100, label=strategy, alpha=0.7)
        
        plt.xlabel('Volatility (%)')
        plt.ylabel('Return (%)')
        plt.title('Risk-Return Profile of Traditional Benchmarks')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{save_dir}/risk_return_scatter.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Portfolio weights heatmap
        weights_matrix = np.array([benchmarks[s]['weights'] for s in strategies])
        
        plt.figure(figsize=(12, 8))
        plt.imshow(weights_matrix, cmap='YlOrRd', aspect='auto')
        plt.colorbar(label='Weight')
        plt.xticks(range(self.n_assets), [f'Asset {i+1}' for i in range(self.n_assets)], rotation=45)
        plt.yticks(range(len(strategies)), strategies)
        plt.title('Portfolio Weights Heatmap')
        plt.xlabel('Assets')
        plt.ylabel('Strategies')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/weights_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Benchmark visualizations saved to {save_dir}/")

def main():
    """Main function to test traditional benchmarks"""
    
    # Create sample data for testing
    np.random.seed(42)
    n_assets = 30
    n_days = 1000
    
    # Generate synthetic returns
    returns_data = pd.DataFrame(
        np.random.normal(0.001, 0.02, (n_days, n_assets)),
        columns=[f'Asset_{i+1}' for i in range(n_assets)]
    )
    
    # Initialize benchmarks
    benchmarks = TraditionalBenchmarks(returns_data)
    
    # Run all benchmarks
    results = benchmarks.run_all_benchmarks(returns_data)
    
    # Create report
    report = benchmarks.create_benchmark_report(results)
    
    # Save report
    with open('traditional_benchmarks_report.md', 'w') as f:
        f.write(report)
    
    # Create visualizations
    benchmarks.create_benchmark_visualizations(results)
    
    logger.info("Traditional benchmarks analysis completed!")

if __name__ == "__main__":
    main() 