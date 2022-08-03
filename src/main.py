import numpy as np
import random


def generate_cost_matrix(row_count: int, col_count: int, max_cost: float = 10) -> np.ndarray:
    matrix = np.ndarray((row_count, col_count))
    for row in range(0, row_count):
        for col in range(0, col_count):
            matrix[row, col] = random.random() * max_cost
    return matrix


if __name__ == "__main__":
    cost_matrix = generate_cost_matrix(10, 10)
    print(cost_matrix)
