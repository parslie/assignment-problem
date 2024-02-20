import numpy
from hungarian import hungarian_algorithm, adjust_cost_matrix, mark_cost_matrix, reduce_cost_matrix


def test_reduce():
    costs = numpy.array([
        [3, 5, 1],
        [2, 1, 7],
        [1, 4, 3],
    ], dtype=float)

    reduce_cost_matrix(costs)
    assert costs.tolist() == numpy.array([
        [2, 4, 0],
        [1, 0, 6],
        [0, 3, 2],
    ], dtype=float).tolist()

    costs = numpy.array([
        [3, 5, 0],
        [2, 0, 7],
        [5, 4, 0],
    ], dtype=float)

    reduce_cost_matrix(costs)
    assert costs.tolist() == numpy.array([
        [1, 5, 0],
        [0, 0, 7],
        [3, 4, 0],
    ], dtype=float).tolist()

    costs = numpy.array([
        [3, 5, 2],
        [2, 1, 7],
        [5, 4, 1],
    ], dtype=float)

    reduce_cost_matrix(costs)
    assert costs.tolist() == numpy.array([
        [0, 3, 0],
        [0, 0, 6],
        [3, 3, 0],
    ], dtype=float).tolist()


def test_algorithm():
    costs = numpy.array([
        [3, 5, 1],
        [2, 1, 7],
        [1, 4, 3],
    ], dtype=float)

    assignments = hungarian_algorithm(costs)
    assert assignments == [
        (0, 2),
        (1, 1),
        (2, 0),
    ]
