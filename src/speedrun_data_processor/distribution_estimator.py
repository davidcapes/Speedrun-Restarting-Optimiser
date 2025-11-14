import os
import sys

import numpy as np
from numba import njit, prange
from scipy.optimize import minimize

# Load relevant files.
REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_DIR)
try:
    from src.util.math_support import trunc_normal_pdf, trunc_normal_cdf, sample_std, cumulative_trapezoid
finally:
    if sys.path[0] == REPO_DIR:
        sys.path.pop(0)


@njit(parallel=True)
def kde_method(D1, D2, bins=1000, x_lwr=0, x_upr=75):
    """
    Calculates PDF and CDF tables for the Random Variable with the following PDF.

    This function implements a modified kernel density estimation that accounts for
    right-censored observations in the data, where we know that the true
    value is greater than the observed value in D2. It uses truncated normal distributions
    as kernels and applies Silverman's rule for bandwidth selection.

    :param D1: Primary dataset used for kernel density estimation
    :type D1: numpy.ndarray[float]
    :param D2: Secondary dataset representing right-censored observations
    :type D2: numpy.ndarray[float]
    :param bins: Number of bins for the output PDF and CDF tables
    :type bins: int
    :param x_lwr: Lower bound of the x-axis range for density estimation
    :type x_lwr: float
    :param x_upr: Upper bound of the x-axis range for density estimation
    :type x_upr: float
    :return: A tuple containing PDF and CDF tables
    :rtype: tuple[numpy.ndarray[float], numpy.ndarray[float]]

    Mathematical formulation:
    • fₜₙ(x) = φ(x) / (1 − Φ(0)) - Truncated normal PDF
    • Fₜₙ(x) = (Φ(x) − Φ(0)) / (1 − Φ(0)) - Truncated normal CDF
    • h = s×⁻⁵√(3n₁/4) - Silverman's bandwidth rule where s is unbiased std
    • f(x) = (1 + Σ_{d₂ ∈ D₂} 𝟏(x > d₂)/Σ_{d₁ ∈ D₁} (1 − Fₜₙ((d₂−d₁)/h)) ) × Σ_{d₁ ∈ D₁} (fₜₙ((x−d₁)/h))/h) / (n₁ + n₂)
    """

    h = sample_std(D1) * (3/4 * len(D1)) ** (-1/5)

    # Calculate base kernal density estimate.
    x_values = np.linspace(x_lwr, x_upr, bins + 1)
    pdf_table = np.zeros(len(x_values))
    for i in prange(len(x_values)):
        x = x_values[i]
        for d1 in D1:
            pdf_table[i] += trunc_normal_pdf(x, d1, h, 0)
        pdf_table[i] /= len(D1)
    cdf_table = cumulative_trapezoid(pdf_table, x_values)

    # Re-scale based on restarts.
    for i in prange(len(x_values)):
        x = x_values[i]
        multiplier = len(D1)
        for d2 in D2:
            if x > d2:
                multiplier += 1/(1 - np.interp(d2, xp=x_values, fp=cdf_table))
        pdf_table[i] *= multiplier
    pdf_table /= len(D1) + len(D2)
    cdf_table = cumulative_trapezoid(pdf_table, x_values)

    return pdf_table, cdf_table


def gmm_method(D1, D2, bins, x_lwr=0, x_upr=75, m=5, iterations=1, seed=42):
    np.random.seed(seed)

    def generate_parameters():
        mu = np.random.uniform(0, max(np.max(D1), np.max(D2)), size=m)
        sigma = np.random.uniform(0, np.std(D1, ddof=1) * (len(D1) + len(D2)) / len(D1), size=m)
        weights = np.random.uniform(0, 1, size=m)
        weights /= np.sum(weights)
        return np.concatenate((mu, sigma, weights))

    def f_gmm(x, parameters):
        mu, sigma, weights = parameters[:m], parameters[m:2*m], parameters[2*m:]
        weights = np.abs(weights) + 1e-10
        sigma = np.abs(sigma)
        weights /= weights.sum()
        return sum([wi * trunc_normal_pdf(x, mi, si, 0) for mi, si, wi in zip(mu, sigma, weights)])

    def F_gmm(x, parameters):
        mu, sigma, weights = parameters[:m], parameters[m:2*m], parameters[2*m:]
        weights = np.abs(weights) + 1e-10
        sigma = np.abs(sigma)
        weights /= weights.sum()
        return sum([wi * trunc_normal_cdf(x, mi, si, 0) for mi, si, wi in zip(mu, sigma, weights)])

    def loss_function(parameters):
        component1 = np.sum([np.log(f_gmm(d1, parameters)) for d1 in D1])
        component2 = np.sum([np.log(1 - F_gmm(d2, parameters)) for d2 in D2])
        return (-1) * (component1 + component2)

    # Obtain Parameters.
    parameters = generate_parameters()
    for _ in range(iterations):
        parameters_new = minimize(loss_function, x0=generate_parameters()).x
        if loss_function(parameters_new) < loss_function(parameters):
            parameters = parameters_new

    # Create Tables.
    x_values = np.linspace(x_lwr, x_upr, bins + 1)
    pdf_table = np.array([f_gmm(x, parameters) for x in x_values])
    cdf_table = np.array([F_gmm(x, parameters) for x in x_values])

    return pdf_table, cdf_table


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    D1 = np.array([19, 21, 22, 8, 16, 17, 18, 12])
    D2 = np.array([23, 24, 19, 23])

    x_lwr, x_upr = 0, 75
    bins = 10000

    x_values = np.linspace(x_lwr, x_upr, bins + 1)
    pdf_table, cdf_table = kde_method(D1, D2, bins, x_lwr=x_lwr, x_upr=x_upr)
    plt.plot(x_values, pdf_table)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Plot of y vs x')
    plt.grid(True)
    plt.show()
