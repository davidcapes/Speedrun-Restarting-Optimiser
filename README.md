# Speedrun Restarting Optimization Framework

## Overview

This project implements a framework for optimizing speedrun restarting strategies in multi-task speedrunning scenarios, where players can choose to restart their runs at any time. The aim is to devise a restarting strategy that minimizes the expected time to beat some goal run time.

## Features

- **🎮 Interactive Game** (`src/game_simulator/game.py`): Pygame-based interface for simulating speedrun mechanics
- **🔬 Dynamic Programming Solver** (`src/restart_analysis/restart_solver.py`): Mathematical restart threshold calculation
- **📊 Monte Carlo Simulator** (`src/restart_analysis/restart_simulator.py`): Validate restarting strategies through statistical simulation
- **📉 Visualization Suite** (`src/preset_distributions/distribution_plotter.py`): Generate plots of distributions, performance metrics, and optimization results
- **📈 Data Analysis Tools** (`src/game_simulator/game_data_analyzer.py` & `src/speedrun_data_processor/`): Analyze player performance and estimate empirical thresholds

## Installation

### Requirements
- Python 3.7+
- NumPy
- SciPy  
- Matplotlib
- Pygame
- Numba (for JIT compilation)
- Pandas
- NetworkX

```bash
pip install -r requirements.txt
```

## Project Structure

```
speedrun-optimization/
├── src/                              # Source code
│   ├── game_simulator/               # Core game logic
│   │   ├── game.py                   # Interactive speedrun simulator game
│   │   └── game_data_analyzer.py     # Player performance analysis
│   ├── preset_distributions/         # Core game logic
│   │   ├── example_case.py           # Example scenario with 6 tasks
│   │   └── distribution_plotter.py   # Distribution plotter
│   ├── restart_analysis/             # Restarting threshold analysis
│   │   ├── restart_solver.py         # Dynamic programming solver
│   │   └── restart_simulator.py      # Monte Carlo simulator
│   ├── speedrun_data_processor/      # Real speedrun data processor
│   │   ├── distribution_estimator.py # PDF estimator
│   │   ├── graph_processor.py        # Data export utilities
│   └── util/                         # Miscelabous utility functions
│       └── math_support.py           # Mathematical utilities
├── tests/                            # Unit tests
├── data/                             # Data storage
│   ├── speedrun_data/                # Real speedrun data
│   └── game_simulator_data/          # Game simulator data
├── assets/                           # Media files
│   ├── audio/                        # Sound effects
│   └── images/                       # Icons and game graphics
└── plots/                            # Generated visualizations
    └── task_distributions/           # Plots of tasks PDFs
```

## Quick Start

### 1. Play the Interactive Game
Experience the speedrun restart mechanics firsthand:
```bash
python src/game_simulator/game.py
```

#### **Player Actions**:
- **Left Click** → Attempt the next task.
- **Right Click** → Restart from the first task.

**Goal:** Complete all tasks with total time < 75, as efficiently as possible.

### 2. Find Optimal Strategy
Calculate mathematically optimal restart thresholds:

```python
from src.restart_analysis.restart_solver import create_probability_tables, get_expected_time_linear, get_restarts
from src.preset_distributions.example_case import PDFS, W

# Create probability tables
pdf_tables, cdf_tables = create_probability_tables(
    PDFS, bin_count=2000, x_min=0, x_max=W
)

# Find optimal restart thresholds
optimal_thresholds = get_restarts(pdf_tables, W)
print("Optimal restart thresholds:", optimal_thresholds)

# Calculate expected time.
expected_time = get_expected_time_linear(pdf_tables, optimal_thresholds)
print("Expected time:", expected_time)
```

### 3. Run Simulations
Validate strategies through Monte Carlo simulation:

```python
from src.restart_analysis.restart_simulator import game_simulator
from src.preset_distributions.example_case import sample_task, W

# Test a strategy
r_vector = [25, 35, 45, 55, 65, W]
results = game_simulator(
    sample_task, r_vector, goal_time=W, n_simulations=10000
)

print(f"Expected time: {results.mean_time:.2f}")
print(f"Success rate: {results.success_rate:.2%}")
```

### 4. Visualize Results
Create plots of task distributions:
```python
from src.preset_distributions.distribution_plotter import plot_pdfs
from src.preset_distributions.example_case import PDFS, CDFS

plot_pdfs(PDFS, CDFS, save_directory='plots/')
```

## Speedrun Data Format
```json
{
  "<task_name>": {
    "Completed": [<float>, ...],   
    "Restarted": [<float>, ...], 
    "Reachable": ["<task_name2>", ...],
    "Start": false,              
    "End": false  
  },
  ...
}
```

## Analysis Pipeline
 1. Load real speedrun data
 2. Estimate probability density functions (PDFs) for task times
 3. Compute optimized restart thresholds

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## Citation

If you use this framework in your research, please cite:
```bibtex
@software{speedrun_optimization,
  title = {Speedrun Optimization Framework},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/speedrun-optimization}
}
```

## Contact
- **Email**: davidmcapes@gmail.com

---

**Happy Speedrunning! 🎮🏃‍♂️💨**
