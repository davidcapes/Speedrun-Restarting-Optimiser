# Data Directory

This directory contains all data files produced by the code.

## Directory Structure

```
data/
├── game_sessions/
│   ├── raw/
│   │   └── game_data_*.csv
│   └── annotated/
│       └── participant*.csv
└── speedrun_data/
    ├── example_data.json
    └── *.json
```

## Game Session Data
Raw and annotated session files are saved in `game_sessions/raw/` or `game_sessions/anotated/`.


### Format (CSV)
```csv
task_number,task_score,restarted_mid_task
1,15.234567,0
2,22.345678,0
3,35.456789,1
1,12.345678,0
...
```

### Columns
- `task_number`: Which task was attempted (int)
- `task_score`: Time taken for the task (float)
- `restarted_mid_task`: Whether the task was interrupted by a restart (0/1)

### File naming
- Raw: `game_data_{username}_{timestamp}.csv`
- Annotated: `participant{number}.csv`

### Code Usage
Located at `src/game_simulator/game.py`:

## Speedrun Data
Typically for data from real speedruns saved in `speedrun_data/`. 

Also contains the file `example_data.json`, which provides synthetic data of the provided format for testing purposes.

### Format (JSON)
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