from scipy.stats import expon, gamma, truncnorm, uniform, triang, norm
import numpy as np
from numba import njit


C = 1 - 0.5 * (norm.cdf(0, loc=25, scale=np.sqrt(50)) + norm.cdf(0, loc=13, scale=np.sqrt(130)))

# PDFs.
PDFS = (
    lambda x: truncnorm.pdf(x, a=-19/np.sqrt(70), b=np.inf, loc=19, scale=np.sqrt(70)),
    lambda x: expon.pdf(x, scale=20),
    lambda x: uniform.pdf(x, loc=4, scale=43),
    lambda x: gamma.pdf(x, a=2, scale=9),
    lambda x: (0.5 / C) * (norm.pdf(x, loc=25, scale=np.sqrt(50)) + norm.pdf(x, loc=13, scale=np.sqrt(130))) * (x >= 0),
    lambda x: triang.pdf(x, c=1/3, loc=0, scale=45)
)

def pdf_task(i, pdfs=PDFS):
    return pdfs[i - 1]

# CDFs.
CDFS = (
    lambda x: truncnorm.cdf(x, a=-19/np.sqrt(70), b=np.inf, loc=19, scale=np.sqrt(70)),
    lambda x: expon.cdf(x, scale=20),
    lambda x: uniform.cdf(x, loc=4, scale=43),
    lambda x: gamma.cdf(x, a=2, scale=9),
    lambda x: (x >= 0) * ((0.5 / C) * (norm.cdf(x, loc=25, scale=np.sqrt(50)) +
                                       norm.cdf(x, loc=13, scale=np.sqrt(130))) - (1 - C)),
    lambda x: triang.cdf(x, c=1/3, loc=0, scale=45)
)

def cdf_task(i, cdfs=CDFS):
    return cdfs[i - 1]

# Samplers.
@njit
def sampler1():
    result = np.random.normal(loc=19, scale=np.sqrt(70))
    return result if result >= 0 else sampler1()
@njit
def sampler2(): return np.random.exponential(scale=20)
@njit
def sampler3(): return np.random.uniform(low=4, high=47)
@njit
def sampler4(): return np.random.gamma(shape=2, scale=9)
@njit
def sampler5():
    mean, std = (25, np.sqrt(50)) if np.random.rand() < 0.5 else (13, np.sqrt(130))
    result = np.random.normal(loc=mean, scale=std)
    return result if result >= 0 else sampler5()
@njit
def sampler6(): return np.random.triangular(left=0, mode=15, right=45)

SAMPLERS = (sampler1, sampler2, sampler3, sampler4, sampler5, sampler6)

@njit
def sample_task(i):
    if i == 1:
        return sampler1()
    elif i == 2:
        return sampler2()
    elif i == 3:
        return sampler3()
    elif i == 4:
        return sampler4()
    elif i == 5:
        return sampler5()
    elif i == 6:
        return sampler6()
    raise ValueError


# Constants.
W = 75
N = 6
