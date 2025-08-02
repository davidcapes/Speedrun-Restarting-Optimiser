import unittest
import numpy as np

from src.preset_distributions.example_case import PDFS, CDFS, SAMPLERS, sample_task, W


class TestExampleCase(unittest.TestCase):

    def test_consistency(self, n_bins=1000000, precision=0.01, n_quantiles=11):

        # Initialize.
        np.random.seed(42)
        quantiles = np.linspace(0, 1, n_quantiles)
        assert (len(PDFS) == len(CDFS) == len(SAMPLERS))

        # Test matching PDFS match CDFS.
        x_array = np.linspace(0, W, n_bins)
        for f, F in zip(PDFS, CDFS):
            f_x_array = f(x_array)
            for q in quantiles:
                cutoff = int(q * (len(x_array) - 1))
                integral = np.trapezoid(f_x_array[:cutoff], x_array[:cutoff])
                cdf = F(W * q)
                assert abs(integral - cdf) <= precision

        # Match CDFS and SAMPLERS.
        for i, F in enumerate(CDFS):
            sample = np.sort([sample_task(i + 1) for _ in range(n_bins)])
            for q in quantiles:
                cutoff = int(q * (len(x_array) - 1))
                cdf = F(sample[cutoff])
                assert abs(q - cdf) <= precision
