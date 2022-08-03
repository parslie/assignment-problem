from typing import Callable
import numpy as np
import random
import time

from hungarian import generate_assignments


def generate_cost_matrix(row_count: int, col_count: int, max_cost: float = 10) -> np.ndarray:
    matrix = np.ndarray((row_count, col_count))
    for row in range(0, row_count):
        for col in range(0, col_count):
            matrix[row, col] = random.random() * max_cost
    return matrix


def time_assignments(function: Callable, row_count: int, col_count: int, iterations: int = 10000):
    start_time = time.perf_counter()
    for i in range(0, iterations):
        cost_matrix = generate_cost_matrix(row_count, col_count)
        function(cost_matrix)
    duration = time.perf_counter() - start_time
    print(f"{function.__name__} ({row_count}x{col_count}) ({iterations} iterations):\n\t{duration:.5f} sec")


if __name__ == "__main__":
    for n in range(1, 8):
        time_assignments(generate_assignments, n, n)

    #cost_matrix = generate_cost_matrix(3, 3)
    #print(f"Cost matrix:\n{cost_matrix}\n")
    #assignments = generate_assignments(cost_matrix.copy())
    #print(f"Assignments:\n{assignments}\n")

