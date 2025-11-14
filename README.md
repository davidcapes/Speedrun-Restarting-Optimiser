# Speedrun Restart Optimisation

## Overview

This project implements a framework for optimising speedrun restarting strategies, where players can choose to restart their runs at any time. The aim is to devise a restarting strategy that minimises the expected time to beat some goal run time.

## Features

- **🎮 Interactive Game**: Pygame-based interface for simulating speedrun mechanics
- **🔬 Dynamic Programming Solver**: Mathematical restart threshold calculation
- **📊 Monte Carlo Simulator**: Validate restarting strategies through statistical simulation
- **📉 Visualization Suite**: Generate plots of distributions, performance metrics, and optimization results
- **🔍 Data Analysis Tools**: Analyze player performance and estimate empirical thresholds

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

## Requirements

Python version 3.12.10 (or equivalent) is required to run the scripts in this project.

### Packages
- NumPy
- SciPy
- MatPlotLib
- PyGame
- Pandas
- Numba
- NetworkX

### Installation
```bash
pip install -r requirements.txt
```

## Game Simulation

See `docs/Game_Information.pdf` for more details about the game simulator.

### Description

- **Left Click** → Attempt the next task.
- **Right Click** → Restart from the first task.

**Goal:** Complete all tasks with total time < 75, as efficiently as possible.

### Running the Game
Navigate to the project root, and run the following in command line:

```bash
python src/game_simulator/game.py
```

## Author Contact
- **Email**: davidmcapes@gmail.com
- **GitHub**: https://github.com/davidcapes
- **LinkedIn**: https://www.linkedin.com/in/david-capes-/
