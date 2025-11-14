import os
import sys
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from _pickle import PicklingError

import numpy as np
from numba import njit, prange
from numba.core.registry import CPUDispatcher as numba_function


def _game_simulator_single(sampler_function, r_vector, goal_time):
    n = len(r_vector)

    # Game parameters.
    total_time = 0.0
    current_time = 0.0
    task = 1

    while True:

        # Do next task.
        current_time += float(sampler_function(task))

        if task == n or current_time >= r_vector[task - 1]:
            total_time += min(current_time, r_vector[task - 1])

            # Success.
            if task == n and current_time < goal_time:
                break

            # Restart.
            task = 1
            current_time = 0.0

        # Next task.
        else:
            task += 1

    return total_time


def _game_simulator_default(sampler_function, r_vector, goal_time, n_simulations,
                            single_simulator=_game_simulator_single):

    total_sum = 0.0
    total_square_sum = 0.0
    for _ in prange(n_simulations):
        v = single_simulator(sampler_function, r_vector, goal_time)
        total_sum += v
        total_square_sum += v**2

    mn = total_sum / n_simulations
    var = (total_square_sum - mn ** 2) / (n_simulations - 1) if n_simulations > 1 else np.inf
    return mn, np.sqrt(var)


def game_simulator(sampler_function, r_vector, n_simulations, goal_time=None, parallel=True, seed=None):
    """
    Simulates n_simulations different speedruns and gives the average amount of time for a run to beat the goal_time.

    :param sampler_function: A function that takes a task number and returns a RV's realization for that task's time.
    :type sampler_function: Callable[[int], float]
    :param r_vector: An array of restart thresholds, where if the runtime at task i exceeds r[i-1], the run is restarted.
    :type r_vector: numpy.ndarray[float]
    :param n_simulations: The number of successful runs to simulate.
    :type n_simulations: int
    :param goal_time: The threshold where, if runtime is faster than goal_time at the final task, the run is a success.
    :type goal_time:
    :param parallel: Whether to use parallel processing to simulate. Set to false for deterministic results.
    :type parallel: boolean
    :param seed: The random seed to use for calculation. For no seeding, set equal to None or set parallel=True.
    :type seed: int
    :return: The mean and standard deviation of total time taken for a successful run.
    :type: Tuple[float]
    """

    # Input validation.
    for i in range(1, len(r_vector) + 1):
        float(sampler_function(i))

    r_vector = np.array(r_vector, dtype=np.float64)
    if not np.all(r_vector > 0):
        raise ValueError

    n_simulations = int(n_simulations)
    if n_simulations < 1:
        raise ValueError

    if goal_time is None:
        goal_time = r_vector[-1]
    goal_time = float(goal_time)

    parallel = bool(parallel)

    if seed is not None and not parallel:
        def set_seed(seed):
            np.random.seed(seed)
        set_seed(int(seed))
        njit(set_seed)(int(seed))

    # Numba optimization.
    if isinstance(sampler_function, numba_function):
        single_simulator = njit(_game_simulator_single)
        multi_simulator = njit(_game_simulator_default, parallel=parallel)
        return multi_simulator(sampler_function, r_vector, goal_time, n_simulations, single_simulator)

    # Default parallelization.
    if parallel:
        try:
            n_threads = multiprocessing.cpu_count()
            with ProcessPoolExecutor(max_workers=n_threads) as executor:
                chunks = [(n_simulations // n_threads) + (1 if i < n_simulations % n_threads else 0)
                           for i in range(n_threads)]
                results = executor.map(_game_simulator_default, [sampler_function] * n_threads,
                                       [r_vector] * n_threads, [goal_time] * n_threads, chunks)
                return np.mean(list(results))
        except (PicklingError, BrokenProcessPool, AttributeError):
            pass
    return _game_simulator_default(sampler_function, r_vector, goal_time, n_simulations)

# TODO: Strategy-based simulator

if __name__ == "__main__":

    # Load relevant files.
    REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, REPO_DIR)
    try:
        from src.preset_distributions.example_case import W, sample_task
    finally:
        if sys.path[0] == REPO_DIR:
            sys.path.pop(0)

    n_simulations = 1000000
    r_vector = np.array([16.55, 26.63, 46.99, 60.3, 71.86, 75])
    mn, std = game_simulator(sample_task, r_vector, n_simulations, parallel=False, goal_time=W)
    print(mn, std)

