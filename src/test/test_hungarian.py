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
