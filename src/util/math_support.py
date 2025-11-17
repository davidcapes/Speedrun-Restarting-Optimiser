from math import erf

import numpy as np
from numba import njit


@njit
def trunc_normal_cdf(x, mu, sigma, lwr=-np.inf, upr=np.inf):
    """Numba-compatible truncated normal CDF calculation."""

    if lwr > upr:
        raise ValueError
    elif lwr == upr:
        return np.float64(x >= lwr)

    if x < lwr:
        return 0.0
    elif x > upr:
        return 1.0

    if sigma < 0:
        raise ValueError
    elif sigma == 0:
        return np.float64(x >= mu)

    if lwr == -np.inf and upr == np.inf:
        return 0.5 * (1 + erf((x - mu) / (sigma * np.sqrt(2))))

    cdf_lwr = trunc_normal_cdf(lwr, mu, sigma, -np.inf, np.inf)
    cdf_upr = trunc_normal_cdf(upr, mu, sigma, -np.inf, np.inf)

    if cdf_upr - cdf_lwr <= 0:
        if abs(lwr) == np.inf or abs(upr) == np.inf:
            raise ValueError
        return (x - lwr) / (upr - lwr)  # Uniform approximation
    return (trunc_normal_cdf(x, mu, sigma, -np.inf, np.inf) - cdf_lwr) / (cdf_upr - cdf_lwr)


@njit
def trunc_normal_pdf(x, mu, sigma, lwr=-np.inf, upr=np.inf):
    """Numba-compatible truncated normal PDF calculation."""

    if lwr >= upr:
        raise ValueError

    if not (lwr <= x <= upr):
        return 0.0

    if sigma <= 0:
        raise ValueError

    if lwr == -np.inf and upr == np.inf:
        return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

    cdf_lwr = trunc_normal_cdf(lwr, mu, sigma, -np.inf, np.inf)
    cdf_upr = trunc_normal_cdf(upr, mu, sigma, -np.inf, np.inf)

    if cdf_upr - cdf_lwr <= 0:
        if abs(lwr) == np.inf or abs(upr) == np.inf:
            raise ValueError
        return 1 / (upr - lwr)  # Uniform approximation
    return trunc_normal_pdf(x, mu, sigma, -np.inf, np.inf) / (cdf_upr - cdf_lwr)


@njit
def cumulative_trapezoid(y_array: np.ndarray, x_array: np.ndarray):
    """Numba-compatible cumulative trapezoidal integration."""

    result = np.zeros(len(y_array))
    for i in range(1, len(y_array)):
        y = (y_array[i] + y_array[i - 1]) / 2
        dx = x_array[i] - x_array[i - 1]
        result[i] = result[i - 1] + y * dx
    return result


@njit
def sample_std(array, ddof=1):
    """Numba-compatible sample standard deviation calculator."""

    n = len(array)
    if n <= ddof:
        return np.inf

    mean = 0.0
    for i in range(n):
        mean += array[i]
    mean /= n

    var = 0.0
    for i in range(n):
        var += (array[i] - mean) ** 2
    var /= (n - ddof)

    return np.sqrt(var)
