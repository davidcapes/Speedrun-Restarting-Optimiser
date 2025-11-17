import os
import sys
import json
from typing import Callable

import networkx as nx
import numpy as np
from numba import njit

# Load relevant files.
REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_DIR)
try:
    from src.preset_distributions.example_case import sample_task, N
finally:
    if sys.path[0] == REPO_DIR:
        sys.path.pop(0)


def create_from_example(
    sampler: Callable[[int], float] = sample_task, 
    n: int = N, 
    file_name: str = os.path.join(REPO_DIR, "data", "speedrun_data", "example_data.json"),
    simulation_count: int = 100,
    seed: int = 42
) -> dict:

    np.random.seed(seed)
    njit(np.random.seed(seed))

    result_file = {}
    for task in range(1, n + 1):
        task_string = str(task)
        simulated = np.array([sampler(task) for _ in range(simulation_count)])
        r = np.random.choice(simulated)

        result_file[task_string] = {}
        result_file[task_string]["Completed"] = [float(s) for s in simulated if s < r]
        result_file[task_string]["Restarted"] = [float(r)] * (simulated >= r).sum()
        result_file[task_string]["Reachable"] = [str(task + 1)] if task < n else []
        result_file[task_string]["Start"] = (task == 1)
        result_file[task_string]["End"] = (task == n)

    with open(file_name, "w") as f:
        json.dump(result_file, f, indent=4)

    return result_file

# topologic sort the graph.
def scan_json(
    file_directory: str = os.path.join(REPO_DIR, "data", "speedrun_data", "example_data.json")
) -> tuple[np.ndarray, dict[str, int]]:
    """
    """

    # Load file.
    with open(file_directory, "r") as f:
        data = json.load(f)
    n = len(data)

    # Get topologically sorted graph.
    #start_values = [k for k, v in data.items() if v["Start"]]
    #if len(start_values) > 1:
    #    raise ValueError("Only 1 start allowed in the game_simulator graph.")
    graph_dict = {k: v["Reachable"] for k, v in data.items()}
    sorted_nodes = list(nx.topological_sort(nx.DiGraph(graph_dict)))
    #sorted_nodes.remove(start_values[0])
    #sorted_nodes.insert(0, start_values[0])
    name_indicis = {name: i for i, name in enumerate(sorted_nodes)}

    # Create graph array.
    graph_array = np.zeros((n + 1, n + 1), dtype=np.int16)
    for name_from, i1 in name_indicis.items():
        for name_to in data[name_from]["Reachable"]:
            i2 = name_indicis[name_to]
            graph_array[i1][i2] = 1
        if data[name_from]["End"]:
            graph_array[i1][n] = 1
    if np.any(np.tril(graph_array) != 0):
        raise ValueError("Graph must be a Directed and Acyclic.")

    return graph_array, name_indicis


if __name__ == "__main__":
    print(create_from_example())
    print(scan_json())
