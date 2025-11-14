# Game Simulator

This module provides a speedrunning game simulator (a playable game) along with tools for analyzing the resulting gameplay data. It is designed to simulate task-based speedruns for later study and analysis.

## Overview
- **Game Simulator** (`game.py`): A Pygame-based interactive game where players aim to beat the goal time as efficiently as possible.
- **Data Analysis Tools** (`game_data_analysis.py`): Functions for analyzing gameplay data, estimating expected times and empirical restart thresholds.

## File Description: `game.py`

See `docs/Game_Information.pdf` for more details about the game simulator.

### Overview

- Each task randomly accrues a `task_score`, representing the time taken to complete a section of the run.
- When the `task_score` threshold is reached, the task completes, and the next one becomes available.
- The goal is to complete all tasks while minimizing `total_score` and staying under the `goal_score`.

### Player Actions:
- ***Left Click*** → Attempt the next task.
- ***Right Click*** → Restart from the first task.

### Running the Game
From the project root, run:
```bash
python src/game_simulator/game.py
```

## File Description: `game_data_analysis.py`

### Key Functions

- **`get_empirical_expected_time(files, n, w)`**  
  Calculates the player's average expected time to beat the goal score using recorded runs.
- **`estimate_restart_thresholds(files, n)`**  
  Estimates restart thresholds used by the player based on their gameplay data.
