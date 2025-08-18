# Real FTSE Data: ASMS-EMOA vs Traditional Benchmarks Report
Generated on: 2025-08-18 01:41:12

## Experiment Overview

- **Data**: Real FTSE 100 component data from repository
- **Assets**: 30 FTSE assets
- **Historical Window**: 120 days
- **Rebalancing**: Every 30 days
- **Anticipation Horizons**: K = {0, 1, 2, 3}
- **Prediction Steps**: h = {1, 2}
- **Decision Makers**: Hv-DM, R-DM, M-DM
- **Initial Investment**: R$ 100,000
- **Number of Runs**: 5

## Performance Summary (Mean ± Std across runs)

| Strategy | Total ROI (%) | Avg ROI/Period (%) | Final Wealth (R$) |
|----------|---------------|-------------------|-------------------|
| ASMS_EMOA_K0_h1_Hv-DM | 36.74 ± 0.02 | 1.6815 ± 0.0008 | R$ 136,741 ± 22 |
| ASMS_EMOA_K0_h1_R-DM | 109.69 ± 0.00 | 3.9934 ± 0.0000 | R$ 209,685 ± 0 |
| ASMS_EMOA_K0_h1_M-DM | 116.34 ± 0.00 | 4.1637 ± 0.0000 | R$ 216,335 ± 0 |
| ASMS_EMOA_K0_h2_Hv-DM | 36.73 ± 0.00 | 1.6811 ± 0.0000 | R$ 136,730 ± 0 |
| ASMS_EMOA_K0_h2_R-DM | 109.69 ± 0.00 | 3.9934 ± 0.0000 | R$ 209,685 ± 0 |
| ASMS_EMOA_K0_h2_M-DM | 115.83 ± 0.00 | 4.1511 ± 0.0000 | R$ 215,833 ± 0 |
| ASMS_EMOA_K1_h1_Hv-DM | 36.73 ± 0.00 | 1.6811 ± 0.0000 | R$ 136,730 ± 0 |
| ASMS_EMOA_K1_h1_R-DM | 109.69 ± 0.00 | 3.9934 ± 0.0000 | R$ 209,685 ± 0 |
| ASMS_EMOA_K1_h1_M-DM | 116.34 ± 0.00 | 4.1637 ± 0.0000 | R$ 216,335 ± 0 |
| ASMS_EMOA_K1_h2_Hv-DM | 36.73 ± 0.00 | 1.6811 ± 0.0000 | R$ 136,730 ± 0 |
| ASMS_EMOA_K1_h2_R-DM | 109.69 ± 0.00 | 3.9934 ± 0.0000 | R$ 209,685 ± 0 |
| ASMS_EMOA_K1_h2_M-DM | 116.13 ± 0.00 | 4.1585 ± 0.0000 | R$ 216,132 ± 0 |
| ASMS_EMOA_K2_h1_Hv-DM | 36.73 ± 0.00 | 1.6811 ± 0.0000 | R$ 136,730 ± 0 |
| ASMS_EMOA_K2_h1_R-DM | 109.69 ± 0.00 | 3.9934 ± 0.0000 | R$ 209,685 ± 0 |
| ASMS_EMOA_K2_h1_M-DM | 116.34 ± 0.00 | 4.1637 ± 0.0000 | R$ 216,335 ± 0 |
| ASMS_EMOA_K2_h2_Hv-DM | 36.73 ± 0.00 | 1.6811 ± 0.0000 | R$ 136,730 ± 0 |
| ASMS_EMOA_K2_h2_R-DM | 109.69 ± 0.00 | 3.9934 ± 0.0000 | R$ 209,685 ± 0 |
| ASMS_EMOA_K2_h2_M-DM | 116.34 ± 0.00 | 4.1637 ± 0.0000 | R$ 216,335 ± 0 |
| ASMS_EMOA_K3_h1_Hv-DM | 36.73 ± 0.00 | 1.6811 ± 0.0000 | R$ 136,730 ± 0 |
| ASMS_EMOA_K3_h1_R-DM | 109.69 ± 0.00 | 3.9934 ± 0.0000 | R$ 209,685 ± 0 |
| ASMS_EMOA_K3_h1_M-DM | 116.34 ± 0.00 | 4.1637 ± 0.0000 | R$ 216,335 ± 0 |
| ASMS_EMOA_K3_h2_Hv-DM | 36.73 ± 0.00 | 1.6811 ± 0.0000 | R$ 136,730 ± 0 |
| ASMS_EMOA_K3_h2_R-DM | 109.69 ± 0.00 | 3.9934 ± 0.0000 | R$ 209,685 ± 0 |
| ASMS_EMOA_K3_h2_M-DM | 116.17 ± 0.00 | 4.1594 ± 0.0000 | R$ 216,169 ± 0 |
| Equal_Weighted | 118.09 ± 0.00 | 4.2076 ± 0.0000 | R$ 218,089 ± 0 |
| Minimum_Variance | 32.78 ± 0.00 | 1.5278 ± 0.0000 | R$ 132,779 ± 0 |
| Sharpe_Optimal | 73.59 ± 0.00 | 2.9699 ± 0.0000 | R$ 173,587 ± 0 |

## Decision Maker Comparison (K=1, h=1)

| Decision Maker | Total ROI (%) | Avg ROI/Period (%) | Final Wealth (R$) |
|----------------|---------------|-------------------|-------------------|
| Hv-DM | 36.73 ± 0.00 | 1.6811 ± 0.0000 | R$ 136,730 ± 0 |
| R-DM | 109.69 ± 0.00 | 3.9934 ± 0.0000 | R$ 209,685 ± 0 |
| M-DM | 116.34 ± 0.00 | 4.1637 ± 0.0000 | R$ 216,335 ± 0 |

## H-value Comparison (K=1, Hv-DM)

| H-value | Total ROI (%) | Avg ROI/Period (%) | Final Wealth (R$) |
|---------|---------------|-------------------|-------------------|
| h1 | 36.73 ± 0.00 | 1.6811 ± 0.0000 | R$ 136,730 ± 0 |
| h2 | 36.73 ± 0.00 | 1.6811 ± 0.0000 | R$ 136,730 ± 0 |

## K-value Comparison (h=1, Hv-DM)

| K-value | Total ROI (%) | Avg ROI/Period (%) | Final Wealth (R$) |
|---------|---------------|-------------------|-------------------|
| K0 | 36.74 ± 0.02 | 1.6815 ± 0.0008 | R$ 136,741 ± 22 |
| K1 | 36.73 ± 0.00 | 1.6811 ± 0.0000 | R$ 136,730 ± 0 |
| K2 | 36.73 ± 0.00 | 1.6811 ± 0.0000 | R$ 136,730 ± 0 |
| K3 | 36.73 ± 0.00 | 1.6811 ± 0.0000 | R$ 136,730 ± 0 |

## Best Performing Strategy

**Best Overall Strategy**: Equal_Weighted
- Total ROI: 118.09% ± 0.00%
- Average ROI per Period: 4.2076% ± 0.0000%
- Final Wealth: R$ 218,089 ± 0

## Statistical Analysis

### ASMS-EMOA vs Traditional Benchmarks
- Average ASMS-EMOA ROI: 87.55%
- Average Traditional Benchmark ROI: 74.82%
- Performance Difference: 12.73%