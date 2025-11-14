import os
import sys
from math import floor, ceil, log

import numpy as np
from numba import njit, prange


def create_probability_tables(pdf_functions, bin_count, x_min, x_max, calc_bin_split=0):
    """
    Compute discretized PDF and CDF tables from a list of probability density functions. The CDF calculation is
    performed using trapezoidal rule integration, with optional sub-bin evaluations for improved accuracy.

    :param pdf_functions: A list of PDF functions.
    :type pdf_functions: Sequence[Callable[[float], float]]
    :param bin_count: Number of bins to divide `[x_min, x_max]` into, giving `bin_count + 1` evaluation points.
    :type bin_count: int
    :param x_min: Lowest domain value to use in the output table.
    :type x_min: float
    :param x_max: Highest domain value to use in the output table.
    :type x_max: float
    :param calc_bin_split: The number of times to split each table bin (in calculations) for improved precision.
    :type calc_bin_split: int
    :return: A tuple `(pdf_tables, cdf_tables)` containing 2D NumPy arrays of shape `(n, bin_count + 1)`. The first
             table contains PDF values, and the second contains calculated CDF values.
    :rtype: tuple[np.ndarray[np.float64], np.ndarray[np.float64]]
    """

    # Initialize.
    n = len(pdf_functions)
    bin_width = (x_max - x_min) / bin_count
    pdf_tables = np.zeros((n, bin_count + 1), dtype=np.float64)
    cdf_tables = np.zeros((n, bin_count + 1), dtype=np.float64)

    # Fill Tables.
    for k, f in enumerate(pdf_functions):
        pdf_tables[k][0] = f(x_min)
        cdf_tables[k][0] = 0.0

        for j in range(1, bin_count + 1):
            pdf_tables[k][j] = f(x_min + j * bin_width)

            cdf_tables[k][j] = cdf_tables[k][j - 1]
            cdf_tables[k][j] += (pdf_tables[k][j] + pdf_tables[k][j - 1]) / 2
            for h in range(1, calc_bin_split + 1):
                x = x_min + bin_width * (j - 1 + h / (calc_bin_split + 1))
                cdf_tables[k][j] += f(x)

        cdf_tables[k] *= bin_width / (calc_bin_split + 1)

    return pdf_tables, cdf_tables


@njit(parallel=True)
def get_expected_time_linear(pdf_tables, r_vector):
    """
    Calculates A₁(0, 𝐫) / B₁(0, 𝐫), where:

    • 0 < r₁ ≤ ... ≤ rₙ
    • fₖ(x) = 0 if x < 0 for all k ∈ {1, ..., n}
    • Aₖ(t, 𝐫) = rₖ + ∫_{0}^{rₖ-t} fₖ(x)×(Aₖ₊₁(t + x, 𝐫) − rₖ) dx if k ≤ n and Aₖ(t, 𝐫) = t otherwise
    • Bₖ(t, 𝐫) = ∫_{0}^{rₖ-t} fₖ(x)×Bₖ₊₁(t + x, 𝐫) dx if k ≤ n and Bₖ(t, 𝐫) = 1 otherwise

    This implementation uses tabular dynamic programming with linear interpolation and trapezoidal rule integration.
    Let (n, b) be the dimensions of pdf_tables, this functions runs in O(nb²).

    :param pdf_tables: 2D rectangular array (shape = (n, table_size)) of PDF tables, where
                       pdf_tables[k-1][j] represents fₖ(rₙ × j / (table_size - 1)).
    :type pdf_tables: numpy.ndarray
    :param r_vector: 1D sorted array (length = n) of restart thresholds, where r_vector[k - 1] represents rₖ.
    :type r_vector: numpy.ndarray
    :returns: A₁(0, 𝐫) / B₁(0, 𝐫) if B₁(0, 𝐫) ≠ 0, otherwise infinity
    :rtype: float
    """

    # Pre-calculate values.
    table_size = len(pdf_tables[0])
    r_max = max(r_vector)
    n = len(r_vector)

    # Initialize tables for both A and B.
    A_table = np.zeros((n + 1, table_size), dtype=np.float64)
    B_table = np.zeros((n + 1, table_size), dtype=np.float64)

    # Base case handling.
    for j in range(table_size):
        A_table[n][j] = r_max * (j / (table_size - 1))
        B_table[n][j] = 1.0

    # Backward induction through all stages
    for i in range(n - 1, -1, -1):

        # Get table index of restart threshold.
        r = r_vector[i]
        r_index_approx = (table_size - 1) * (r / r_max)
        r_index_lwr, r_index_upr = floor(r_index_approx), ceil(r_index_approx)
        r_alpha = r_index_approx - r_index_lwr

        # Get Interpolated A_{i+1}(r) and B_{i+1}(r).
        A_prev_r = (1 - r_alpha) * A_table[i + 1][r_index_lwr] + r_alpha * A_table[i + 1][r_index_upr]
        B_prev_r = (1 - r_alpha) * B_table[i + 1][r_index_lwr] + r_alpha * B_table[i + 1][r_index_upr]

        # Iterate table.
        iteration_limit = r_index_lwr if i > 0 else 0  # Avoid unnecessary calculations.
        for j in prange(iteration_limit + 1):

            # Calculate index of r - t.
            rmt_index_approx = r_index_approx - j
            rmt_index_lwr, rmt_index_upr = floor(rmt_index_approx), ceil(rmt_index_approx)
            rmt_alpha = rmt_index_approx - rmt_index_lwr
            f_rmt = (1 - rmt_alpha) * pdf_tables[i][rmt_index_lwr] + rmt_alpha * pdf_tables[i][rmt_index_upr]

            # Trapezoidal rule integration for both A and B
            # Main integration loop over interior points
            for k in range(1, rmt_index_lwr + 1):

                # A table integration.
                y1_A = pdf_tables[i][k - 1] * (A_table[i + 1][j + k - 1] - r)
                y2_A = pdf_tables[i][k] * (A_table[i + 1][j + k] - r)
                A_table[i][j] += (y1_A + y2_A) / 2

                # B table integration.
                y1_B = pdf_tables[i][k - 1] * B_table[i + 1][j + k - 1]
                y2_B = pdf_tables[i][k] * B_table[i + 1][j + k]
                B_table[i][j] += (y1_B + y2_B) / 2

            # Handle area remainder.
            if rmt_index_upr > r_index_lwr:

                # A table remainder contribution.
                y1_A = pdf_tables[i][rmt_index_lwr] * (A_table[i + 1][j + rmt_index_lwr] - r)
                y2_A = f_rmt * (A_prev_r - r)
                A_table[i][j] += rmt_alpha * (y1_A + y2_A) / 2

                # B table remainder contribution.
                y1_B = pdf_tables[i][rmt_index_lwr] * B_table[i + 1][j + rmt_index_lwr]
                y2_B = f_rmt * B_prev_r
                B_table[i][j] += rmt_alpha * (y1_B + y2_B) / 2

            # Multiply by dx.
            if table_size > 1:
                dx = r_max / (table_size - 1)
                A_table[i][j] *= dx
                B_table[i][j] *= dx

            # Add constant term for A
            A_table[i][j] += r

        # Fill interpolation point using reverse interpolation for both tables
        if r_index_upr > r_index_lwr:
            A_table[i][r_index_upr] = (0 - (1 - r_alpha) * A_table[i][r_index_lwr]) / r_alpha
            B_table[i][r_index_upr] = (0 - (1 - r_alpha) * B_table[i][r_index_lwr]) / r_alpha

    # Return expected time, handling division by zero
    if B_table[0][0] == 0:
        return np.inf
    return A_table[0][0] / B_table[0][0]


# TODO: Add fractional reduction of binning to get closer to the true answer quicker.
# TODO: If only altering early rs, integrals don't need to be recalculated.
def get_restarts(pdf_tables, w, rounding=2, log_step=2):

    # Initialize.
    n = len(pdf_tables)
    r_vector = np.array([round(w * (j / n), rounding) for j in range(1, n + 1)])
    EY = get_expected_time_linear(pdf_tables, r_vector)
    accuracy = 10**(-rounding)
    step_sizes =  np.unique([max(2**i, accuracy) for i in range(floor(log(w, log_step)), floor(log(accuracy, log_step)) - 1, -1)])

    # Loop.
    s = 0
    updated = False
    while s < len(step_sizes):

        k = 0
        while k < n - 1:
            for delta in (-step_sizes[s], step_sizes[s]):

                # Get new r_vector.
                r_vector_new = r_vector.copy()
                r_vector_new[k] += delta
                r_vector_new[k] = np.clip(r_vector_new[k], accuracy, w)
                for k2 in range(k):
                    r_vector_new[k2] = min(r_vector_new[k], r_vector_new[k2])
                for k2 in range(k + 1, n - 1):
                    r_vector_new[k2] = max(r_vector_new[k], r_vector_new[k2])
                EY_new = get_expected_time_linear(pdf_tables, r_vector_new)

                # Update if better.
                if EY_new < EY:
                    EY = EY_new
                    r_vector = r_vector_new
                    k = 0
                    updated = True
                    break
            else:
                k += 1

        s += 1
        if s == len(step_sizes) and updated:
            updated = False
            s = 0

    return np.round(r_vector, 2)


def get_restarts_gd(pdf_tables, w, max_iter=200):

    n = len(pdf_tables)
    r_vector = w * (np.arange(1, n + 1) / n)  # Consider np.sort(np.random.uniform(epsilon, w, n)) AND r_vector[-1] = w
    step_sizes = np.unique([max(2 ** i, 10**(-2)) for i in range(floor(log(w, 2)), -3, -1)])

    epsilon = 0.1
    for _ in range(max_iter):
        EY_current = get_expected_time_linear(pdf_tables, r_vector)
        gradients = np.zeros(n)

        for i in range(n):

            # Forward difference.
            r_plus = r_vector.copy()
            r_plus[i] += epsilon
            r_plus[i] = np.clip(r_plus[i], epsilon, w)
            EY_plus = EY_current
            if r_plus[i] != r_vector[i]:
                EY_plus = get_expected_time_linear(pdf_tables, r_plus)

            # Backward difference.
            r_minus = r_vector.copy()
            r_minus[i] -= epsilon
            r_minus[i] = np.clip(r_minus[i], epsilon, w)
            EY_minus = EY_current
            if r_plus[i] != r_vector[i]:
                EY_minus = get_expected_time_linear(pdf_tables, r_minus)

            # Calculate gradient.
            if EY_current <= min(EY_minus, EY_plus):
                gradients[i] = 0
            else:
                gradients[i] = (EY_plus - EY_minus) / (2 * epsilon)
        if np.all(gradients == 0):
            break

        for s in step_sizes:
            dr = s * (gradients / np.max(np.abs(gradients)))
            new_r = r_vector - dr

            # Constrain vector
            new_r[0] = max(epsilon, new_r[0])
            for i in range(1, len(new_r)):
                new_r[i] = max(new_r[i], new_r[i - 1])
            new_r[n - 1] = min(w, new_r[n - 1])

            EY_new = get_expected_time_linear(pdf_tables, new_r)
            if EY_new < EY_current:
                r_vector = new_r
                break
        else:
            break

    return np.round(r_vector, 2)


# TODO: Ask AI, is this correct? Create unit tests for this?
@njit(parallel=True)
def _step_expected_time_graph(pdf_tables, r_vector, prev_expectation, s_tables, graph_array, iter_stop=None):

    # Pre-define variables.
    table_size = len(pdf_tables[0])
    r_max = np.max(r_vector)
    if iter_stop is None:
        iter_stop = len(pdf_tables)

    for i in range(iter_stop - 1, -1, -1):

        # Calculate previous table.
        s_table_prev = np.full(table_size, np.inf)
        for j in prange(table_size):
            for h, connected in enumerate(graph_array[i]):
                if connected:
                    s_table_prev[j] = min(s_table_prev[j], s_tables[h][j])

        # Get table index of restart threshold.
        r = r_vector[i]
        r_index_approx = (table_size - 1) * (r / r_max)
        r_index_lwr, r_index_upr = floor(r_index_approx), ceil(r_index_approx)
        r_alpha = r_index_approx - r_index_lwr
        s_prev_r = (1 - r_alpha) * s_table_prev[r_index_lwr] + r_alpha * s_table_prev[r_index_upr]

        # Iterate table.
        iteration_limit = r_index_lwr if i > 0 else 0  # Avoid unnecessary calculations.
        for j in prange(iteration_limit + 1):
            t = j * r_max / (table_size - 1)
            s_tables[i][j] = 0

            # Calculate index of r - t.
            rmt_index_approx = r_index_approx - j
            rmt_index_lwr, rmt_index_upr = floor(rmt_index_approx), ceil(rmt_index_approx)
            rmt_alpha = rmt_index_approx - rmt_index_lwr
            f_rmt = (1 - rmt_alpha) * pdf_tables[i][rmt_index_lwr] + rmt_alpha * pdf_tables[i][rmt_index_upr]

            # Trapezoidal rule integration for both A and B
            # Main integration loop over interior points
            for k in range(1, rmt_index_lwr + 1):
                x1 = (k - 1) * r_max / (table_size - 1)
                x2 = k * r_max / (table_size - 1)
                y1 = pdf_tables[i][k - 1] * (x1 + s_table_prev[j + k - 1] - (prev_expectation + (r - t)))
                y2 = pdf_tables[i][k] * (x2 + s_table_prev[j + k] - (prev_expectation + (r - t)))
                s_tables[i][j] += (y1 + y2) / 2

            # Handle area remainder.
            if rmt_index_upr > r_index_lwr:
                x1 = rmt_index_lwr * r_max / (table_size - 1)
                x2 = r - t
                y1 = pdf_tables[i][rmt_index_lwr - 1] * (x1 + s_table_prev[j + rmt_index_lwr - 1] - (prev_expectation + (r - t)))
                y2 = f_rmt * (x2 + s_prev_r - (prev_expectation + (r - t)))
                s_tables[i][j] += rmt_alpha * (y1 + y2) / 2

            # Multiply by dx.
            if table_size > 1:
                dx = r_max / (table_size - 1)
                s_tables[i][j] *= dx

            # Add constant term for A
            s_tables[i][j] += prev_expectation + (r - t)

        # Fill interpolation point using reverse interpolation for both tables
        if r_index_upr > r_index_lwr:
            s_tables[i][r_index_upr] = (0 - (1 - r_alpha) * s_tables[i][r_index_lwr]) / r_alpha

    return s_tables


@njit
def get_expected_time_graph(pdf_tables, r_vector, graph_array=None, epsilon=10**(-4), max_iters=100):

    # Pre-define / Initialize.
    n = len(r_vector)
    table_size = len(pdf_tables[0])
    s_tables = np.zeros((n + 1, table_size), dtype=np.float64)
    if graph_array is None:
        graph_array = np.zeros((n + 1, n + 1), dtype=np.int16)
        for h in range(n):
            graph_array[h][h + 1] = 1

    # Exponential Search.
    lwr, upr = n * np.max(r_vector) + epsilon, n * np.max(r_vector) + epsilon
    for _ in range(max_iters):
        if lwr <  _step_expected_time_graph(pdf_tables, r_vector, lwr, s_tables, graph_array)[0][0]:
            break
        lwr /= 2
    for _ in range(max_iters):
        if upr >  _step_expected_time_graph(pdf_tables, r_vector, upr, s_tables, graph_array)[0][0]:
            break
        upr *= 2

    # Bisect Method.
    for _ in range(max_iters):
        if upr - lwr < epsilon:
            break

        mid = (lwr + upr) / 2
        new = _step_expected_time_graph(pdf_tables, r_vector, mid, s_tables, graph_array)[0][0]

        if new == mid:
            break
        elif new < mid:
            upr = mid
        elif new > mid:
            lwr = mid

    return (lwr + upr) / 2


if __name__ == "__main__":
    REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, REPO_DIR)
    try:
        from src.preset_distributions.example_case import W, PDFS
    finally:
        if sys.path[0] == REPO_DIR:
            sys.path.pop(0)

    bin_count = 2000
    pdf_tables, cdf_tables = create_probability_tables(PDFS, bin_count, 0, W)

    #r_vector = get_r_gd(pdf_tables, W)
    #print("R-Vector (gradient descent):", r_vector)
    #print("Expected Time (gradient descent):", get_expected_time_linear(pdf_tables, r_vector))

    r_vector = get_restarts(pdf_tables, W)
    print("R-Vector (gradient descent):", r_vector)
    print("Expected Time (gradient descent):", get_expected_time_linear(pdf_tables, r_vector))