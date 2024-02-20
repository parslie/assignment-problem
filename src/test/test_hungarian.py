import numpy
import pytest
from hungarian import hungarian_algorithm, adjust_cost_matrix, mark_cost_matrix, reduce_cost_matrix

@pytest.fixture
def costs():
    return numpy.array([
        [3, 5, 1],
        [2, 1, 7],
        [1, 4, 3],
    ], dtype=float)

@pytest.fixture
def cost_matrices() -> numpy.ndarray:
    return [
        numpy.array([
            [3, 5, 1],
            [2, 1, 7],
            [1, 4, 3],
        ], dtype=float),
        numpy.array([
            [3, 5, 0],
            [2, 0, 7],
            [5, 4, 0],
        ], dtype=float),
        numpy.array([
            [3, 5, 2],
            [2, 1, 7],
            [5, 4, 1],
        ], dtype=float),
    ]

@pytest.fixture
def reduced_matrices() -> numpy.ndarray:
    return [
        numpy.array([
            [2, 4, 0],
            [1, 0, 6],
            [0, 3, 2],
        ], dtype=float),
        numpy.array([
            [1, 5, 0],
            [0, 0, 7],
            [3, 4, 0],
        ], dtype=float),
        numpy.array([
            [0, 3, 0],
            [0, 0, 6],
            [3, 3, 0],
        ], dtype=float),
    ]

@pytest.fixture
def assignments_list() -> list[tuple[int, int]]:
    return [
        [
            (0, 2),
            (1, 1),
            (2, 0),
        ],
        [
            (0, 0),
            (1, 1),
            (2, 2),
        ],
        [
            (0, 0),
            (1, 1),
            (2, 2),
        ],
    ]

def test_reduce(cost_matrices, reduced_matrices):
    for costs, reduced in zip(cost_matrices, reduced_matrices):
        reduce_cost_matrix(costs)
        assert costs.tolist() == reduced.tolist()


def test_algorithm(cost_matrices, assignments_list):
    for costs, assignments in zip(cost_matrices, assignments_list):
        output = hungarian_algorithm(costs)
        for assignment in output:
            assert assignment in assignments
