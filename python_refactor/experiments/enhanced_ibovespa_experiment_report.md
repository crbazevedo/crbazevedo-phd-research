# Enhanced IBOVESPA ASMS-EMOA Experiment Report
Generated on: 2025-08-17 21:33:37

## Experiment Overview

- **Assets**: Top 70 IBOVESPA stocks by market cap
- **Periods**: 90-day investment periods
- **Historical Data**: 120 days
- **Stride**: 60 days (rebalancing every 2 months)
- **Anticipation Horizons**: K = {0, 1, 2, 3}
- **Prediction Steps**: h = {1, 2}
- **Initial Investment**: R$ 100,000

## Performance Summary

### Anticipation Horizon K = 0

#### Prediction Step h = 1

| Decision Maker | Final ROI (%) | Final Wealth (R$) | Total Transaction Costs (R$) | Avg Coherence |
|----------------|---------------|-------------------|------------------------------|---------------|
| Hv-DM | -13.76% | R$ 86,243.40 | R$ 2,057.40 | 0.875 |
| R-DM | -14.90% | R$ 85,101.11 | R$ 2,127.33 | 0.875 |
| M-DM | -12.70% | R$ 87,302.77 | R$ 1,636.09 | 0.875 |

#### Prediction Step h = 2

| Decision Maker | Final ROI (%) | Final Wealth (R$) | Total Transaction Costs (R$) | Avg Coherence |
|----------------|---------------|-------------------|------------------------------|---------------|
| Hv-DM | -12.41% | R$ 87,594.64 | R$ 2,158.41 | 0.873 |
| R-DM | -15.08% | R$ 84,916.35 | R$ 2,145.99 | 0.873 |
| M-DM | -13.00% | R$ 86,995.30 | R$ 1,664.89 | 0.873 |

### Anticipation Horizon K = 1

#### Prediction Step h = 1

| Decision Maker | Final ROI (%) | Final Wealth (R$) | Total Transaction Costs (R$) | Avg Coherence |
|----------------|---------------|-------------------|------------------------------|---------------|
| Hv-DM | -14.81% | R$ 85,191.74 | R$ 2,123.80 | 0.876 |
| R-DM | -12.25% | R$ 87,750.81 | R$ 2,180.26 | 0.876 |
| M-DM | -13.64% | R$ 86,362.61 | R$ 1,624.11 | 0.876 |

#### Prediction Step h = 2

| Decision Maker | Final ROI (%) | Final Wealth (R$) | Total Transaction Costs (R$) | Avg Coherence |
|----------------|---------------|-------------------|------------------------------|---------------|
| Hv-DM | -12.59% | R$ 87,406.91 | R$ 2,264.09 | 0.872 |
| R-DM | -13.82% | R$ 86,176.30 | R$ 2,029.19 | 0.872 |
| M-DM | -14.77% | R$ 85,227.53 | R$ 1,582.08 | 0.872 |

### Anticipation Horizon K = 2

#### Prediction Step h = 1

| Decision Maker | Final ROI (%) | Final Wealth (R$) | Total Transaction Costs (R$) | Avg Coherence |
|----------------|---------------|-------------------|------------------------------|---------------|
| Hv-DM | -11.97% | R$ 88,028.23 | R$ 2,312.01 | 0.874 |
| R-DM | -12.95% | R$ 87,049.57 | R$ 2,236.14 | 0.874 |
| M-DM | -12.63% | R$ 87,369.07 | R$ 1,545.65 | 0.874 |

#### Prediction Step h = 2

| Decision Maker | Final ROI (%) | Final Wealth (R$) | Total Transaction Costs (R$) | Avg Coherence |
|----------------|---------------|-------------------|------------------------------|---------------|
| Hv-DM | -9.82% | R$ 90,184.38 | R$ 2,259.16 | 0.872 |
| R-DM | -13.80% | R$ 86,201.97 | R$ 2,187.62 | 0.872 |
| M-DM | -12.68% | R$ 87,323.76 | R$ 1,584.39 | 0.872 |

### Anticipation Horizon K = 3

#### Prediction Step h = 1

| Decision Maker | Final ROI (%) | Final Wealth (R$) | Total Transaction Costs (R$) | Avg Coherence |
|----------------|---------------|-------------------|------------------------------|---------------|
| Hv-DM | -15.85% | R$ 84,152.87 | R$ 2,217.02 | 0.873 |
| R-DM | -12.67% | R$ 87,331.43 | R$ 2,188.98 | 0.873 |
| M-DM | -12.64% | R$ 87,363.91 | R$ 1,619.97 | 0.873 |

#### Prediction Step h = 2

| Decision Maker | Final ROI (%) | Final Wealth (R$) | Total Transaction Costs (R$) | Avg Coherence |
|----------------|---------------|-------------------|------------------------------|---------------|
| Hv-DM | -14.88% | R$ 85,117.06 | R$ 2,108.78 | 0.873 |
| R-DM | -12.80% | R$ 87,200.48 | R$ 2,208.86 | 0.873 |
| M-DM | -14.38% | R$ 85,618.42 | R$ 1,518.26 | 0.873 |

## Benchmark Comparison

| Benchmark | Cumulative Return (%) | Sharpe Ratio | Volatility (%) |
|-----------|---------------------|--------------|----------------|
| Equal-Weighted Index | -11.02% | -0.040 | 1.15% |
| Sharpe Optimal | -93.48% | -0.365 | 3.22% |
| Minimum Variance | -11.47% | -0.077 | 0.68% |

## Expected Future Hypervolume Analysis

### K = 0, h = 1 - Expected Hypervolume Distributions

- **Hv-DM**: Average Expected Hypervolume = 0.000000
- **R-DM**: Average Expected Hypervolume = 0.000000
- **M-DM**: Average Expected Hypervolume = 0.000000

### K = 0, h = 2 - Expected Hypervolume Distributions

- **Hv-DM**: Average Expected Hypervolume = 0.000000
- **R-DM**: Average Expected Hypervolume = 0.000000
- **M-DM**: Average Expected Hypervolume = 0.000000

### K = 1, h = 1 - Expected Hypervolume Distributions

- **Hv-DM**: Average Expected Hypervolume = 0.000000
- **R-DM**: Average Expected Hypervolume = 0.000000
- **M-DM**: Average Expected Hypervolume = 0.000000

### K = 1, h = 2 - Expected Hypervolume Distributions

- **Hv-DM**: Average Expected Hypervolume = 0.000000
- **R-DM**: Average Expected Hypervolume = 0.000000
- **M-DM**: Average Expected Hypervolume = 0.000000

### K = 2, h = 1 - Expected Hypervolume Distributions

- **Hv-DM**: Average Expected Hypervolume = 0.000000
- **R-DM**: Average Expected Hypervolume = 0.000000
- **M-DM**: Average Expected Hypervolume = 0.000000

### K = 2, h = 2 - Expected Hypervolume Distributions

- **Hv-DM**: Average Expected Hypervolume = 0.000000
- **R-DM**: Average Expected Hypervolume = 0.000000
- **M-DM**: Average Expected Hypervolume = 0.000000

### K = 3, h = 1 - Expected Hypervolume Distributions

- **Hv-DM**: Average Expected Hypervolume = 0.000000
- **R-DM**: Average Expected Hypervolume = 0.000000
- **M-DM**: Average Expected Hypervolume = 0.000000

### K = 3, h = 2 - Expected Hypervolume Distributions

- **Hv-DM**: Average Expected Hypervolume = 0.000000
- **R-DM**: Average Expected Hypervolume = 0.000000
- **M-DM**: Average Expected Hypervolume = 0.000000

## Investment Period Data Ranges

| Period | Historical Data | Future Data |
|--------|----------------|-------------|
| 1 | 2012-10-10 to 2012-04-13 | 2012-04-12 to 2012-04-10 |
| 2 | 2012-10-04 to 2012-04-10 | 2012-04-05 to 2012-04-03 |
| 3 | 2012-10-01 to 2012-04-03 | 2012-04-02 to 2012-03-29 |
| 4 | 2012-09-25 to 2012-03-29 | 2012-03-28 to 2012-03-26 |
| 5 | 2012-09-20 to 2012-03-26 | 2012-03-23 to 2012-03-21 |
| 6 | 2012-09-17 to 2012-03-21 | 2012-03-20 to 2012-03-16 |
| 7 | 2012-09-12 to 2012-03-16 | 2012-03-15 to 2012-03-13 |
| 8 | 2012-09-07 to 2012-03-13 | 2012-03-12 to 2012-03-08 |
| 9 | 2012-09-03 to 2012-03-08 | 2012-03-07 to 2012-03-05 |
| 10 | 2012-08-29 to 2012-03-05 | 2012-03-02 to 2012-02-29 |
| ... | ... | ... |
| 33 | 2012-05-18 to 2011-11-24 | 2011-11-23 to 2011-11-21 |

## Key Insights

1. **N-Step Prediction Impact**: h=1 vs h=2 prediction accuracy
2. **Anticipation Horizon Effect**: K values impact on performance
3. **Decision Maker Comparison**: AMFC vs Random vs Median strategies
4. **Expected Hypervolume Analysis**: Future flexibility assessment
5. **Benchmark Performance**: Traditional vs anticipatory approaches
6. **Transaction Cost Impact**: Rebalancing frequency effects