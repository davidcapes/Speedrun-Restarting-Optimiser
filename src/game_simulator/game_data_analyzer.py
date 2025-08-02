# TODO: Load annotated data, and for each person, use single-axis soft-margin SVM to calculate a player's empirical restart thresholds.

# TODO: Estimate a player's PDF for each task using a modified KDE that:
#  - Creates a KDE distribution with all non-restarted values (using silverman's rule).
#  - For each restarted value, add a clipped and scaled version of the top half of the pdf kde.
#  - Clip at 0 and rescale pdf estimate to have an area of 1.
#  Store the result of this KDE in a table similar to what is used in R_Solver.

# TODO: For a KDE Table, calculate average absolute and square distance between actual pdf and kde estimate only for
#  values BELOW the player's estimated restart threshold.

import numpy as np
import pandas as pd


def get_empirical_expected_time(files, n, w):
    win_count = 0
    total_time = 0.0

    for file_path in files:
        df = pd.read_csv(file_path)
        win_count += ((df['task_score'].rolling(window=n).sum() < w) & (df['task_number'] == n)).sum()
        total_time += df['task_score'].sum()

    return total_time / win_count


def _clean_raw_data(files, n):

    data = [[[], []] for _ in range(n)]
    for file_path in files:
        for _, row in pd.read_csv(file_path).iterrows():
            task = int(row['task_number'])
            task_score = float(row['task_score'])
            restarted = int(row['restarted_mid_task'])
            data[task - 1][int(restarted)].append(task_score)

    return data


def estimate_restart_thresholds(files, n):

    restarts = [[] for _ in range(n)]
    for file_path in files:
        df = pd.read_csv(file_path)
        for task in range(1, n + 1):
            df[f"rolling{task}"] = df['task_score'].rolling(window=task).sum()
        for _, row in df.iterrows():
            task = int(row["task_number"])
            restarted = int(row["restarted_mid_task"])
            run_time = row[f"rolling{task}"]
            restarts[task - 1].append((run_time, restarted))

    r_vector = np.full(n, np.inf)
    for task in range(1, n + 1):
        weights = []
        for t, r in restarts[task - 1]:
            if not r:
                weights.append(0)
            else:
                misclassified = sum(1 for t2, r2 in restarts[task - 1] if not r2 and t2 >= t)
                weights.append(1/(1 + misclassified))
        if sum(weights) > 0:
            r_vector[task - 1] = np.average([t for t, _ in restarts[task - 1]], weights=weights)

    return r_vector


if __name__ == "__main__":
    from src.preset_distributions.example_case import W, SAMPLERS
    N = len(SAMPLERS)

    files = ["../game_simulator_data/raw/game_data_test_user1_1751644777.2417738.csv"]
    print(_clean_raw_data(files, N))
    print(get_empirical_expected_time(files, N, W))
    print(estimate_restart_thresholds(files, N))
