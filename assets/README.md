# Assets Directory

This directory contains media and resource files used by the game simulator.

## Directory Structure

```
assets/
├── audio/
│   ├── task1.mp3
│   ├── task2.mp3
│   ├── task3.mp3
│   ├── task4.mp3
│   ├── task5.mp3
│   └── task6.mp3
└── images/
    └── game_icon.png
```

## Audio Files

### Task Completion Sounds
Each task has a unique musical note that plays upon completion:

| Task | File | Musical Note | Frequency (Hz) |
|------|------|-------------|-----------|
| 1 | `task1.mp3` | G3 | 196.00 |
| 2 | `task2.mp3` | A3 | 220.00 |
| 3 | `task3.mp3` | B3 | 246.94 |
| 4 | `task4.mp3` | C#4 | 277.18 |
| 5 | `task5.mp3` | D#4 | 311.13 |
| 6 | `task6.mp3` | E4 | 329.63 |

### Audio Specifications
- **Format**: MP3
- **Sample Rate**: 44100 Hz
- **Channels**: Monaural
- **Duration**: ~1 second each

### Code Usage
Located at `src/game_simulator/game.py`.


## Image Files

### Game Icon
- **File**: `game_icon.png`
- **Size**: 100x100 pixels
- **Format**: PNG with transparency
- **Usage**: Game window icon

### Code Usage
Located at `src/game_simulator/game.py`.