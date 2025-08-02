import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import bisect


def get_bounds(cdf, portion_lwr=0, portion_upr=1, sig_fig_rounding=2,tolerance=1e-10):
    # TODO: Add docstring

    # Initialize.
    x_lwr, x_upr = 0, 0
    portion_lwr += tolerance
    portion_upr -= tolerance

    # Exponential search.
    while cdf(x_lwr) > portion_lwr:
        x_lwr = x_lwr * 2 - 1
    while cdf(x_upr) < portion_upr:
        x_upr = x_upr * 2 + 1

    # Bisect search.
    bound_lwr = bisect(lambda x: cdf(x) - portion_lwr, x_lwr, x_upr, xtol=tolerance, maxiter=100)
    bound_upr = bisect(lambda x: cdf(x) - portion_upr, x_lwr, x_upr, xtol=tolerance, maxiter=100)

    # Significant figure rounding.
    l = int(np.ceil(max(np.log10(abs(bound_lwr)), np.log10(abs(bound_upr))))) - sig_fig_rounding
    bound_lwr = 10**l * int(np.floor(bound_lwr * 10**(-l)))
    bound_upr = 10**l * int(np.ceil(bound_upr * 10**(-l)))

    return bound_lwr, bound_upr


def get_distribution_metrics(pdf, cdf, x_min, x_max, rounding=2, n_bins=250000):
    # TODO: Add docstring

    # Initialize.
    x_array = np.linspace(x_min, x_max, n_bins)
    fx_array = pdf(x_array)

    # Extract moment statistics.
    mean = np.trapezoid(y=x_array * fx_array, x=x_array)
    var = np.trapezoid(y=x_array ** 2 * fx_array, x=x_array) - mean ** 2
    std = np.round(np.sqrt(var), rounding)
    mean = np.round(mean, rounding)
    var = np.round(var, rounding)

    # Extract quantile statistics.
    tolerance = 10 ** (-rounding - 1)
    median = bisect(lambda x: cdf(x) - 0.5, 0, 500, xtol=tolerance, maxiter=100)
    median = np.round(median, rounding)
    q1 = bisect(lambda x: cdf(x) - 0.25, 0, 500, xtol=tolerance, maxiter=100)
    q3 = bisect(lambda x: cdf(x) - 0.75, 0, 500, xtol=tolerance, maxiter=100)
    iqr = np.round(q3 - q1, rounding)

    # Extract mode.
    fx_argmax = np.argmax(fx_array)
    mode = np.round(x_array[fx_argmax], rounding) if len(np.where(fx_array == fx_array[fx_argmax])[0]) == 1 else None

    return mean, var, std, median, iqr, mode


# TODO: Add cdf plotting capabilities too.
def plot_pdfs(pdfs, cdfs, save_directory='plots/', colors=('blue', 'green', 'red', 'gold', 'darkturquoise', 'purple')):
    # TODO: Add docstring

    x = np.linspace(0, 60, 5000)
    for i in range(len(pdfs)):

        # Generate y values.
        y = pdfs[i](x)

        # Calculate statistics.
        mean, var, std, median, iqr, mode = get_distribution_metrics(pdfs[i], cdfs[i], 0, 500)  # TODO: Determine magic numbers via CDF percentage argument.
        stat_text = f"mean={mean}\nvariance={var}\nstand dev={std}\nmedian={median}\nIQR={iqr}\nmode={mode}"

        # Plot figure.
        plt.figure(figsize=(8, 5))
        plt.plot(x, y, color=colors[i], label=stat_text)
        plt.xlabel('x')
        plt.ylabel(f'f{i + 1}(x)')
        plt.xlim(0, 60)  # TODO: Determine magic numbers via percentage argument (and clean decimals).
        plt.ylim(0, 0.05)  # TODO: Determine magic numbers via percentage argument (and clean decimals).
        plt.grid()
        plt.title(f"Visualisation of the PDF for Task {i + 1}.")
        plt.fill_between(x, y, color=colors[i], alpha=0.3)
        plt.legend()
        width, height = plt.gcf().get_size_inches()
        plt.gcf().set_size_inches(width, height * 0.9)  # Reduce height by 10%
        plt.savefig(save_directory + f"task{i + 1}_pdf.png")


if __name__ == "__main__":
    from src.preset_distributions.example_case import PDFS, CDFS

    print(get_bounds(CDFS[0]))
    plot_pdfs(PDFS, CDFS)
