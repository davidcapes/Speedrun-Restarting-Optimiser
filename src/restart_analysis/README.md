# Restart Analysis

This module offers tools to estimate the expected time to beat a speedrun goal, using either numerical methods or Monte Carlo simulation. 
It also provides restart threshold optimization by minimizing expected time.

---

## **Overview**
- **Restarting Solver** (`restart_solver.py`): Provides analytical methods to calculate the expected completion time and determine optimal restart thresholds using dynamic programming.
- **Restarting Simulator** (`restart_simulator.py`): Simulates speedruns under different restart strategies using Monte Carlo methods.


---


## Files


### 1. `restart_solver.py`

#### **Key Functions**:
- `create_probability_tables()`: Create discretized PDF/CDF tables from continuous distributions
- `get_restarts()`: Find optimal restart thresholds using dynamic programming
- `get_expected_time_linear()`: Calculate expected completion time for given restart thresholds

#### **Features**:
- **Discretized probability tables**: Converts continuous PDFs to discrete tables for efficient computation
- **Dynamic programming algorithm**: Finds optimal restart thresholds that minimize expected completion time
- **Gradient descent alternative**: Alternative optimization method using gradient descent
- **Graph-based computation**: Efficient computation for graph structured games
- **Optimization**: Uses parallelization and JIT compilation (using Numba) for faster calculations


### 2. `restart_simulator.py`

#### **Key Functions**:
- `simulate_speedrun()`: Run single speedrun simulation
- `simulate_speedrun_parallel()`: Run multiple simulations in parallel
- `analyze_results()`: Statistical analysis of simulation results

#### **Features**:
- **Optimization**: Uses parallelization and JIT compilation (using Numba) for faster calculations
- **Statistical analysis**: Provides mean and variance for t-testing analysis.

---

## **Example Usage**

### Finding Optimal Restart Thresholds
```python
from src.restart_analysis.restart_solver import create_probability_tables, get_restarts
from src.preset_distributions.example_case import PDFS, W

# Create probability tables
pdf_tables, cdf_tables = create_probability_tables(PDFS, bin_count=2000, x_min=0, x_max=W)

# Find optimal restart thresholds
restart_thresholds = get_restarts(pdf_tables, W)
print("Optimal restart thresholds:", restart_thresholds)
```

### Running Simulations
```python
from src.restart_analysis.restart_simulator import game_simulator
from src.preset_distributions.example_case import sample_task, W

# Define restart strategy
restart_thresholds = [20, 25, 30, 35, 40, 45]

# Run simulations
results = game_simulator(
    sample_task,
    restart_thresholds,
    goal_time=W,
    n_simulations=10000
)

print(f"Average completion time: {results.mean_time:.2f}")
print(f"Success rate: {results.success_rate:.2%}")
```
