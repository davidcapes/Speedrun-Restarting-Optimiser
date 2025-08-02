# Assets Directory

This directory contains media and resource files used by the game simulator.

## Directory Structure

```
assets/
├── audio/             # Sound effects for the game
│   ├── task1.mp3      # G3 note
│   ├── task2.mp3      # A3 note
│   ├── task3.mp3      # B3 note
│   ├── task4.mp3      # C#4 note
│   ├── task5.mp3      # D#4 note
│   └── task6.mp3      # E4 note
└── images/            # Icons and graphics
    └── game_icon.png  # Application icon
```

## Audio Files

### Task Completion Sounds
Each task has a unique musical note that plays upon completion:

| Task | File | Musical Note | Frequency |
|------|------|-------------|-----------|
| 1 | task1.mp3 | G3 | 196.00 Hz |
| 2 | task2.mp3 | A3 | 220.00 Hz |
| 3 | task3.mp3 | B3 | 246.94 Hz |
| 4 | task4.mp3 | C#4 | 277.18 Hz |
| 5 | task5.mp3 | D#4 | 311.13 Hz |
| 6 | task6.mp3 | E4 | 329.63 Hz |

### Audio Specifications
- **Format**: MP3
- **Sample Rate**: 44100 Hz
- **Channels**: Monaural
- **Duration**: ~1 second each

## Image Files

### Game Icon
- **File**: `game_icon.png`
- **Size**: 100x100 pixels
- **Format**: PNG with transparency
- **Usage**: Window icon for the game

## Usage in Code

### Loading Audio
```python
import pygame

...

# Sounds.
pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
sounds = {i + 1: pygame.mixer.Sound(f"../audio/task{i + 1}.mp3") for i in range(N)}

...

sounds[current_task].play()

...
```

### Loading Images
```python
import pygame

...

pygame.display.set_icon(pygame.image.load('../../assets/images/game_icon.png'))

...
```
