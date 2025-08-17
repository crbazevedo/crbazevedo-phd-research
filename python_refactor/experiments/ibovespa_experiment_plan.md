# IBOVESPA ASMS-EMOA Experiment Plan

## Experiment Overview

### Objective
Compare three decision-makers (DMs) using ASMS-EMOA on Brazilian IBOVESPA data:
- **Hv-DM**: Anticipated Maximal Flexible Choice (AMFC) portfolio
- **R-DM**: Random portfolio from anticipated stochastic Pareto frontier
- **M-DM**: Median portfolio from anticipated stochastic Pareto frontier

### Experimental Parameters
- **Period**: 120 days historical data
- **Stride**: 60 days (rebalancing every 2 months)
- **Prediction Horizon**: K = 1 (one-step ahead)
- **Portfolio Constraints**: 1-10 assets (cardinality constraint)
- **Initial Investment**: R$ 100,000
- **Anticipation Rates**: λ(i)_t = 1 for all i (static market assumption)

## Experimental Design

### Independent Variables
1. **K (Anticipation Horizon)**: {0, 1, 2, 3}
2. **Decision Maker Type**: {R-DM, M-DM, Hv-DM}

### Dependent Variables
1. **Accumulated ROI**: Total return over experiment period
2. **Confidence Distributions**: (1-λ) anticipative factor
3. **Coherence**: Cosine similarity to population centroid
4. **Transaction Costs**: Rebalancing costs
5. **Accumulated Wealth**: Portfolio value over time

## Implementation Plan

### 1. Data Collection
- Download IBOVESPA component stocks (top 50 by market cap)
- Historical data: 120 days + experiment period
- Handle missing data and corporate actions

### 2. ASMS-EMOA Configuration
- Population size: 100
- Generations: 50
- Crossover rate: 0.8
- Mutation rate: 0.1
- Cardinality constraint: 1-10 assets

### 3. Decision Makers Implementation
- **Hv-DM**: Select portfolio maximizing expected hypervolume
- **R-DM**: Random selection from Pareto frontier
- **M-DM**: Median portfolio by weight vector

### 4. Metrics Computation
- **ROI**: (Final Value - Initial Value) / Initial Value
- **Coherence**: Cosine similarity to centroid (Eq. 7.14)
- **Transaction Costs**: 0.1% per trade
- **Anticipative Factor**: (1-λ) confidence distribution

## Expected Tables and Visualizations

### Tables
1. **Performance Summary**: ROI, Sharpe Ratio, Max Drawdown by DM and K
2. **Coherence Analysis**: Average coherence by DM and K
3. **Transaction Cost Analysis**: Total costs by DM and K
4. **Anticipative Factor**: Confidence distributions by K

### Visualizations
1. **Wealth Evolution**: Portfolio value over time for each DM
2. **ROI Comparison**: Bar chart comparing DMs across K values
3. **Coherence Heatmap**: Coherence values by DM and K
4. **Portfolio Composition**: Asset allocation evolution
5. **Anticipative Factor**: Confidence distribution over time
6. **Transaction Cost Impact**: Cumulative costs over time

## Analysis Framework

### Statistical Tests
- Wilcoxon signed-rank test for DM comparisons
- ANOVA for K value effects
- Correlation analysis between coherence and performance

### Key Insights to Extract
1. **DM Performance**: Which DM performs best across different K values
2. **Anticipation Impact**: How K affects performance
3. **Coherence-Performance**: Relationship between portfolio similarity and returns
4. **Transaction Cost Impact**: Cost-benefit analysis of rebalancing frequency

## Success Criteria
1. All DMs successfully implemented and tested
2. Comprehensive metrics computed for all combinations
3. Statistical significance in DM comparisons
4. Clear visualization of results
5. Actionable insights for portfolio management 