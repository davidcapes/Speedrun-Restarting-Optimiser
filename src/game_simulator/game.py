import os
import sys
import time
import re

import numpy as np
import pygame

# Load relevant files.
REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_DIR)
try:
    from src.preset_distributions.example_case import sample_task, N
    from src.preset_distributions.example_case import W as goal_score
finally:
    if sys.path[0] == REPO_DIR:
        sys.path.pop(0)


# Initialize Pygame.
pygame.init()

# Game Parameters.
np.random.seed(941029)
username = re.sub(r'[^a-zA-Z0-9]', '_', input("Enter Username: "))
requested_game_speed = input("Enter Game Speed (default 3): ")
game_speed = int(requested_game_speed) if requested_game_speed.isdigit() and int(requested_game_speed) > 0 else 3

# Screen Set-up.
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Restarting Game")
pygame.display.set_icon(pygame.image.load(os.path.join(REPO_DIR, "assets/images/game_icon.png")))
font = pygame.font.Font(None, 36)

# Colors.
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
LIGHT_GRAY = (200, 200, 200)
TASK_COLORS = (
    (0, 0, 255),   # Blue
    (0, 255, 0),   # Green
    (255, 0, 0),   # Red
    (255, 215, 0), # Gold
    (0, 255, 255), # Cyan
    (128, 0, 128)  # Purple
)
BACKGROUND_COLOUR = (30, 0, 25) # What also looks good is (0, 28, 7)

# Sounds.
pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
sounds = {i + 1: pygame.mixer.Sound(os.path.join(REPO_DIR, f"assets/audio/task{i + 1}.mp3")) for i in range(N)}

# Game State Parameters.
current_task = 1
current_score = 0.0
total_score = 0.0
completion_amount = 0
score_data = []

task_in_progress = False
task_start_time = 0.0
current_task_score = 0.0


def draw_diamond_background():

    # Diamond points
    top = (WIDTH // 2, -175)
    bottom = (WIDTH // 2, HEIGHT + 175)
    left = (0, HEIGHT // 2)
    right = (WIDTH - 0, HEIGHT // 2)

    # Fill background with navy
    screen.fill(BACKGROUND_COLOUR)

    # Draw black diamond
    diamond_points = [top, right, bottom, left]
    pygame.draw.polygon(screen, BLACK, diamond_points)

    # Draw diamond outline
    pygame.draw.polygon(screen, WHITE, diamond_points, 2)


def draw_centered_text(text, y, font_size=36):
    font_obj = pygame.font.Font(None, font_size)
    surface = font_obj.render(text, True, WHITE)
    x = (WIDTH - surface.get_width()) // 2
    screen.blit(surface, (x, y))


def draw_task_squares():
    square_size = 40
    spacing = 50
    total_width = N * spacing - (spacing - square_size)
    start_x = (WIDTH - total_width) // 2
    y = HEIGHT // 2 - 20

    for i in range(N):
        x = start_x + i * spacing
        if i < current_task - 1:
            color = TASK_COLORS[i % len(TASK_COLORS)]
        else:
            color = LIGHT_GRAY

        pygame.draw.rect(screen, color, (x, y, square_size, square_size))
        pygame.draw.rect(screen, WHITE, (x, y, square_size, square_size), 2)


def save_data():
    filename = os.path.join(REPO_DIR, f"data/game_simulator_data/raw/game_data_{username}_{time.time()}.csv")
    np.savetxt(filename, score_data, delimiter=',', header="task_number,task_score,restarted_mid_task",
               fmt=('%d', '%.10f', '%d'), comments='')
    print(f"Data saved to {filename}")


def play_game():
    global current_task, current_score, total_score, completion_amount
    global task_in_progress, task_start_time, current_task_score

    running = True
    clock = pygame.time.Clock()

    while running:
        current_time = time.time()

        # Quit game_simulator.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                save_data()

            if event.type == pygame.MOUSEBUTTONDOWN:

                # Restart.
                if event.button == 3:
                    current_task_score = min(game_speed * (current_time - task_start_time), current_task_score)
                    if task_in_progress:
                        total_score += current_task_score
                        score_data.append((current_task, current_task_score, True))
                    current_task = 1
                    current_score = 0.0
                    task_in_progress = False

                # Start task.
                elif event.button == 1 and not task_in_progress:
                    current_task_score = sample_task(current_task)
                    task_in_progress = True
                    task_start_time = current_time

        # Complete task.
        if task_in_progress and (game_speed * (current_time - task_start_time) >= current_task_score):
            current_score += current_task_score
            total_score += current_task_score

            sounds[current_task].play()
            score_data.append((current_task, current_task_score, False))

            current_task += 1
            task_in_progress = False

            if current_task > N:
                if current_score < goal_score:
                    completion_amount += 1
                current_task = 1
                current_score = 0.0

        current_task_time = task_in_progress * (game_speed * (current_time - task_start_time))

        # Drawing.
        draw_diamond_background()

        # Top section.
        draw_centered_text(f"Goal Score: {goal_score}", 80)
        draw_centered_text(f"Current Score: {current_score + current_task_time:.2f}", 120)
        draw_centered_text(f"Total Score: {total_score + current_task_time:.2f}", 160)

        # Center section.
        if task_in_progress:
            draw_centered_text("Task in progress", HEIGHT // 2 - 80)
        else:
            draw_centered_text("Click to attempt next task", HEIGHT // 2 - 80)
        draw_task_squares()
        draw_centered_text(f"Tasks Completed: {current_task - 1}", HEIGHT // 2 + 40)

        # Bottom section
        draw_centered_text(f"Completions: {completion_amount}", HEIGHT - 100)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    play_game()
