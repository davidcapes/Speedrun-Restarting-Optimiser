import os
import sys
import unittest

import numpy as np
from scipy.stats import t, uniform, expon
from numba import njit

# Import test functions.
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)
try:
    from src.restart_analysis.restart_solver import create_probability_tables, get_expected_time_linear, get_restarts, get_restarts_gd
    from src.restart_analysis.restart_simulator import game_simulator
    from src.preset_distributions.example_case import PDFS, sample_task, W, N
finally:
    if sys.path[0] == parent_dir:
        sys.path.pop(0)


class TestSolverSimulatorAgreement(unittest.TestCase):
    """ Test suite to ensure solver and simulator agree on expected times. """

    def test_basic_exponential(self):

        # Set random seed.
        def set_seed(seed):
            np.random.seed(seed)
        set_seed(42)
        njit(set_seed)(42)

        # Define parameters.
        bin_count = 5000
        n_simulations = 1000000

        @njit
        def sampler(i):
            if i == 1:
                return np.random.exponential(scale=1/2)
            elif i == 2:
                return np.random.exponential(scale=1/3)
            elif i == 3:
                return np.random.exponential(scale=1/0.7)
            raise ValueError

        pdfs = [lambda x: expon.pdf(x, scale=1/2),
                lambda x: expon.pdf(x, scale=1/3),
                lambda x: expon.pdf(x, scale=1/0.7)]

        r_vectors = [np.array([(j + 1) * (i / 3) for j in range(3)]) for i in range(1, 5 + 1)]

        # Calculate statistics.
        for r_vector in r_vectors:
            w = r_vector[-1]
            pdf_tables, _ = create_probability_tables(pdfs, bin_count, 0, w)
            expected_mn = get_expected_time_linear(pdf_tables, r_vector)
            mn, std = game_simulator(sampler, r_vector, n_simulations, parallel=False, goal_time=w)

            # Perform t-test.
            t_stat = (mn - expected_mn) / (std / np.sqrt(n_simulations))
            p_value = 2 * t.sf(abs(t_stat), n_simulations - 1)
            self.assertTrue(p_value > 0.05,
                            f"Solver and simulator disagree: expected={expected_mn:.4f}, simulated={mn:.4f}, "
                            f"p-value={p_value:.6f}, t-stat={t_stat:.4f}")


    def test_basic_uniform(self):

        # Set random seed.
        def set_seed(seed):
            np.random.seed(seed)
        set_seed(42)
        njit(set_seed)(42)

        # Define parameters.
        w = 1
        bin_count = 5000
        n_simulations = 1000000

        @njit
        def sampler(i): return np.random.uniform(low=0, high=w/2)

        n_values = range(2, 5 + 1)
        pdfs_list = [[lambda x: uniform.pdf(x, loc=0, scale=w/2)] * i for i in n_values]
        r_vectors = [np.array([w] * i, dtype=np.float64) for i in n_values]

        # Calculate statistics.
        for r_vector, pdfs in zip(r_vectors, pdfs_list):
            pdf_tables, _ = create_probability_tables(pdfs, bin_count, 0, w)
            expected_mn = get_expected_time_linear(pdf_tables, r_vector)
            mn, std = game_simulator(sampler, r_vector, n_simulations, parallel=False, goal_time=w)

            # Perform t-test.
            t_stat = (mn - expected_mn) / (std / np.sqrt(n_simulations))
            p_value = 2 * t.sf(abs(t_stat), n_simulations - 1)
            self.assertTrue(p_value > 0.05,
                            f"Solver and simulator disagree: expected={expected_mn:.4f}, simulated={mn:.4f}, "
                            f"p-value={p_value:.6f}, t-stat={t_stat:.4f}")


    def test_example_individual(self):

        # Set random seed.
        def set_seed(seed):
            np.random.seed(seed)
        set_seed(42)
        njit(set_seed)(42)

        for i in range(N):

            # Define parameters.
            w = 10
            bin_count = 5000
            n_simulations = 250000

            @njit
            def sampler(j): return sample_task(i + 1)

            r_vector = np.array([w])
            pdf_tables, _ = create_probability_tables([PDFS[i]], bin_count, 0, w)

            # Calculate statistics.
            expected_mn = get_expected_time_linear(pdf_tables, r_vector)
            mn, std = game_simulator(sampler, r_vector, n_simulations, parallel=False, goal_time=w)

            # Perform t-test.
            t_stat = (mn - expected_mn) / (std / np.sqrt(n_simulations))
            p_value = 2 * t.sf(abs(t_stat), n_simulations - 1)
            self.assertTrue(p_value > 0.05,
                            f"Solver and simulator disagree (task {i}): "
                            "expected={expected_mn:.4f}, simulated={mn:.4f}, "
                            f"p-value={p_value:.6f}, t-stat={t_stat:.4f}")


    def test_example_broad_r(self):

        # Set random seed.
        def set_seed(seed):
            np.random.seed(seed)
        set_seed(42)
        njit(set_seed)(42)

        # Define parameters.
        bin_count = 5000
        n_simulations = 250000

        r_vectors = np.array([[15, 30, 45, 60, 75, W],
                              [10, 20, 30, 40, 50, W],
                              [17, 27, 47, 60, 72, W],
                              [30, 40, 50, 60, 70, W],
                              [55, 59, 63, 67, 71, W],
                              [W, W, W, W, W, W]], dtype=np.float64)

        # Calculate statistics.
        pdf_tables, _ = create_probability_tables(PDFS, bin_count, 0, W)
        for r_vector in r_vectors:
            with self.subTest(r_vector=r_vector):
                expected_mn = get_expected_time_linear(pdf_tables, r_vector)
                mn, std = game_simulator(sample_task, r_vector, n_simulations, parallel=False, goal_time=W)

                # Perform t-test.
                t_stat = (mn - expected_mn) / (std / np.sqrt(n_simulations))
                p_value = 2 * t.sf(abs(t_stat), n_simulations - 1)
                self.assertTrue(p_value > 0.05,
                                f"Solver and simulator disagree (r-vector = {r_vector}): "
                                "expected={expected_mn:.4f}, simulated={mn:.4f}, "
                                f"p-value={p_value:.6f}, t-stat={t_stat:.4f}")


    def test_example_solved(self):

        # Set random seed.
        def set_seed(seed):
            np.random.seed(seed)
        set_seed(42)
        njit(set_seed)(42)

        # Solve R-vectors.
        bin_count = 1000
        pdf_tables, cdf_tables = create_probability_tables(PDFS, bin_count, 0, W)
        r_vectors = (get_restarts(pdf_tables, W), get_restarts_gd(pdf_tables, W))

        # Define parameters.
        bin_count = 5000
        n_simulations = 250000

        # Calculate statistics.
        pdf_tables, cdf_tables = create_probability_tables(PDFS, bin_count, 0, W)
        for r_vector in r_vectors:
            with self.subTest(r_vector=r_vector):
                expected_mn = get_expected_time_linear(pdf_tables, r_vector)
                mn, std = game_simulator(sample_task, r_vector, n_simulations, parallel=False, goal_time=W)

                # Perform t-test.
                t_stat = (mn - expected_mn) / (std / np.sqrt(n_simulations))
                p_value = 2 * t.sf(abs(t_stat), n_simulations - 1)
                self.assertTrue(p_value > 0.05,
                                f"Solver and simulator disagree (r-vector = {r_vector}): "
                                "expected={expected_mn:.4f}, simulated={mn:.4f}, "
                                f"p-value={p_value:.6f}, t-stat={t_stat:.4f}")


if __name__ == "__main__":
    unittest.main()
