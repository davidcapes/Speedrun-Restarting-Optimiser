# **Game Simulator**

This module provides a speedrunning game simulator (a playable game) along with tools for analyzing the resulting gameplay data. It is designed to simulate task-based speedruns for later study and analysis.

---

## **Overview**
- **Game Simulator** (`game.py`): A Pygame-based interactive game where players aim to beat the goal time as efficiently as possible.
- **Data Analysis Tools** (`game_data_analysis.py`): Functions for analyzing gameplay data, estimating expected times and empirical restart thresholds.

---

## **File Descriptions**

### **1. `game.py`**
An interactive speedrunning simulator built with Pygame.

- Each task randomly accrues a `task_score`, representing the time taken to complete a section of the run.
- When the `task_score` threshold is reached, the task completes, and the next one becomes available.
- The goal is to complete all tasks while minimizing `total_score` and staying under the `goal_score`.

**Player Actions**:
- **Left Click** → Attempt the next task.
- **Right Click** → Restart from the first task.

#### **Features**:
- Visual interface using Pygame
- Audio feedback on task completion
- Data export to CSV for empirical analysis

### **2. `game_data_analysis.py`**
Provides functions for analyzing recorded gameplay data:
- **`get_empirical_expected_time(files, n, w)`**  
  Calculates the player's average expected time to beat the goal score using recorded runs.
- **`estimate_restart_thresholds(files, n)`**  
  Estimates restart thresholds used by the player based on their gameplay data.

---

## **Running the Game**
From the project root, run:
```bash
python src/game_simulator/game.py
```

---

## **Example Output**
Gameplay data is saved as:
```
data/game_simulator_data/raw/game_data_<username>_<timestamp>.csv
```

**Columns**:
- **task_number** → The index of the task (starting from 1)
- **task_score** → Time accrued for that task
- **restarted_mid_task** → `0` or `1` indicating if the task was restarted before completion

**Sample CSV**:
```csv
task_number,task_score,restarted_mid_task
1,4.4718053341,1
1,24.6243132945,0
2,9.4709936437,0
3,33.2009675503,1
```

---

## **Example Analysis**
Example usage of the data analysis functions:
```python
from src.preset_distributions.example_case import N, W
from src.game_simulator.game_data_analyzer import get_empirical_expected_time, estimate_restart_thresholds

files = ["../game_simulator_data/raw/game_data_test_user1_1751644777.2417738.csv"]
print(get_empirical_expected_time(files, N, W))
print(estimate_restart_thresholds(files, N))
```