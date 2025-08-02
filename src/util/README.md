# Utilities
This module provides general utility functions for statistical and numerical operations.

---

## Functions

### Truncated Normal Distribution
- **`trunc_normal_pdf(x, mu, sigma, lwr=-np.inf, upr=np.inf)`**  
  Probability density function (PDF) for a truncated normal distribution.

- **`trunc_normal_cdf(x, mu, sigma, lwr=-np.inf, upr=np.inf)`**  
  Cumulative distribution function (CDF) for a truncated normal distribution.

### Numerical Integration
- **`cumulative_trapezoid(y_array, x_array)`**  
  Cumulative trapezoidal integration, Numba-compatible.

### Statistical Utilities
- **`sample_std(array, ddof=1)`**  
  Sample standard deviation (default `ddof=1`).

---

## Features
- Numba JIT-compiled for performance and compatibility.
- Support for infinite bound for truncation.
---

## Example
```python
import numpy as np
from src.util.math_support import trunc_normal_pdf, trunc_normal_cdf, cumulative_trapezoid, sample_std

x = 0.5
mu, sigma = 0, 1
pdf_val = trunc_normal_pdf(x, mu, sigma, lwr=-1, upr=1)
cdf_val = trunc_normal_cdf(x, mu, sigma, lwr=-1, upr=1)

x_array = np.linspace(0, 10, 100)
y_array = np.sin(x_array)
integral = cumulative_trapezoid(y_array, x_array)

data = np.array([1.0, 2.0, 3.0, 4.0])
std_val = sample_std(data)
```