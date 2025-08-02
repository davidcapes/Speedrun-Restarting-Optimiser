# Data Directory

This directory contains all data files used by the Speedrun Optimization Framework.

## Directory Structure

```
data/
├── game_sessions/           # Game simulator session recordings
│   ├── raw/                 # Raw CSV files from game sessions
│   │   └── game_data_*.csv
│   └── annotated/           # Game session data labelled by participant number
│       └── participant*.csv
└── speedrun_data/           # Real speedrunner's data for anaysis
    └── example_data.json    # Example data used for testing
```

## Data Formats

### Game Session Data (CSV)
Raw and annotated session files are saved in `game_sessions/raw/` and `game_sessions/anotated/` with the following format:

```csv
task_number,task_score,restarted_mid_task
1,15.234567,0
2,22.345678,0
3,35.456789,1
1,12.345678,0
...
```

**Columns:**
- `task_number`: Which task was attempted (int)
- `task_score`: Time taken for the task (float)
- `restarted_mid_task`: Whether the task was interrupted by a restart (0/1)

**File naming:**
- Raw: `game_data_{username}_{timestamp}.csv`
- Annotated: `participant{number}.csv`

### Speedrun Data (JSON)
Data from real speedrunners `speedrun_data/`:

```json
{
    "1": {
        "Completed": [3.48, 16.10, ...],
        "Restarted": [16.41, 16.41, ...],
        "Reachable": ["2"],
        "Start": true,
        "End": false
    },
  ...
}
```

The file `example_data.json` provides synthetic data of the provided format that can be used for testing scripts.