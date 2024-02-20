from main import generate_cost_matrix, time_assignments

def test_cost_matrix():
    ROW_COUNT = 5
    COL_COUNT = 7

    for max_cost in range(1, 10):
        matrix = generate_cost_matrix(ROW_COUNT, COL_COUNT, max_cost)

        rows, cols = matrix.shape
        assert rows == ROW_COUNT
        assert cols == COL_COUNT

        for row in range(rows):
            for col in range(cols):
                assert matrix[row, col] <= max_cost
