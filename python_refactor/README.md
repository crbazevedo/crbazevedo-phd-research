# Python Refactor of Portfolio Optimization Research

This is a complete Python refactor of the original C++ portfolio optimization research project. The goal is to maintain functional equivalence while leveraging Python's scientific computing ecosystem.

## Project Structure

```
python_refactor/
├── src/
│   ├── __init__.py
│   ├── portfolio/
│   │   ├── __init__.py
│   │   ├── asset.py          # Asset data structure and loading
│   │   ├── portfolio.py      # Portfolio class and calculations
│   │   └── statistics.py     # Statistical calculations
│   ├── algorithms/
│   │   ├── __init__.py
│   │   ├── nsga2.py          # NSGA-II algorithm
│   │   ├── sms_emoa.py       # SMS-EMOA algorithm
│   │   └── operators.py      # Genetic operators
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── kalman_filter.py  # Kalman filter implementation
│   │   └── helpers.py        # Utility functions
│   └── main.py               # Main execution script
├── tests/
│   ├── __init__.py
│   ├── test_portfolio.py
│   ├── test_algorithms.py
│   └── test_utils.py
├── data/
│   └── ftse-original/        # CSV data files
├── requirements.txt
├── setup.py
└── README.md
```

## Dependencies

```bash
pip install -r requirements.txt
```

### Core Dependencies:
- `numpy` - Numerical computing
- `pandas` - Data manipulation
- `scipy` - Scientific computing
- `matplotlib` - Plotting
- `scikit-learn` - Machine learning utilities

## Usage

### Basic Usage
```bash
python src/main.py market num_assets algorithm regularization_type robustness max_cardinality start_date end_date training_percentage window_size exp_id
```

### Example
```bash
python src/main.py ftse 50 NSGA2 L1 1 10 2010-01-01 2015-12-31 0.7 20 1
```

## Testing

Run tests to verify functional equivalence:
```bash
python -m pytest tests/
```

## Refactoring Progress

- [x] Project structure setup
- [ ] Asset data loading (CSV parsing)
- [ ] Portfolio calculations (ROI, risk, covariance)
- [ ] Statistical functions
- [ ] NSGA-II algorithm
- [ ] SMS-EMOA algorithm
- [ ] Genetic operators (crossover, mutation, selection)
- [ ] Kalman filter
- [ ] Main execution flow
- [ ] Output validation and comparison

## Validation Strategy

Each component will be tested against the original C++ implementation to ensure:
1. **Numerical equivalence** - Same results within floating-point precision
2. **Functional equivalence** - Same behavior and outputs
3. **Performance comparison** - Execution time analysis

## Notes

- The refactor maintains the same mathematical formulations and algorithms
- Python's numpy/scipy provide equivalent linear algebra operations to Eigen
- Date handling uses pandas datetime instead of Boost date_time
- Random number generation uses numpy.random with equivalent seeds 