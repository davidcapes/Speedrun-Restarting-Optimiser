# Preset Distributions

This module defines custom probability distributions for a synthetic speedrun game, including their PDFs, CDFs, and samplers. It also provides utilities for visualization and summary metrics of these distributions.

## Overview
- **Example Distributions** (`example_case.py`): Implements six task distributions with associated PDFs, CDFs, and optimized random samplers.
- **Visualization Utilities** (`distribution_plotter.py`): Functions to compute statistical metrics and generate legend-annotated plots.

## File Description: `example_case.py`

### Constants:
- `W = 75` (goal run time)
- `N = 6` (number of tasks/distributions)

### Key Functions:
- **`pdf_task(i)`**
  Returns the `i`th PDF function.

- **`cdf_task(i)`** 
  Returns the `i`th CDF function.

- **`sample_task(i)`**: 
  Samples from the `i`th distribution using Numba-accelerated samplers.

### Task Distributions
| Task | Distribution      | Parameters                              | Mean   | Variance |
|------|-------------------|-----------------------------------------|--------|----------|
| 1    | Truncated Normal  | μ=19, σ²=70, x≥0                        | ~20.7  | ~65.2    |
| 2    | Exponential       | θ=20                                    | 20.0   | 400.0    |
| 3    | Uniform           | a=4, b=47                               | 25.5   | 150.1    |
| 4    | Gamma             | α=2, β=9                                | 18.0   | 162.0    |
| 5    | Truncated Mixture | μ₁=25, σ₁²=50, μ₂=13, σ₂²=130, x≥0      | ~19.0  | ~155.5   |
| 6    | Triangular        | a=0, c=15, b=45                         | 20.0   | 75.0     |

The goal is to complete all tasks with total time < 75.

## File Description: `distribution_plotter.py`

### Key Functions:
- **`get_bounds(cdf, portion_lwr=0, portion_upr=1)`**  
  Computes lower and upper bounds for a given CDF and probability range.
- **`get_distribution_metrics(pdf, cdf, x_min, x_max)`**  
  Computes statistics: mean, variance, std, median, IQR, and mode.
- **`plot_pdfs(pdfs, cdfs)`**  
  Plots PDFs for multiple distributions and includes summary stats in legends.

### Example Usage
```python
from src.preset_distributions.distribution_plotter import get_bounds, get_distribution_metrics, plot_pdfs
from src.preset_distributions.example_case import PDFS, CDFS, sample_task

# Access PDF and CDF of the first distribution
pdf1 = PDFS[0]
cdf1 = CDFS[0]

# Compute 90% coverage bounds
bounds = get_bounds(cdf1, portion_lwr=0.05, portion_upr=0.95)

# Compute metrics for a range
mean, var, std, median, iqr, mode = get_distribution_metrics(pdf1, cdf1, 0, 500)

# Sample from the 3rd distribution
value = sample_task(3)

# Plot PDFs for all defined distributions
plot_pdfs(PDFS, CDFS)
```