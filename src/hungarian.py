from os import rename
import numpy as np


def reduce_cost_matrix(cost_matrix: np.ndarray):
    for ri in range(cost_matrix.shape[0]):
        minimum_cost = float("inf")
        for cost in cost_matrix[ri, :]:
            if cost < minimum_cost:
                minimum_cost = cost
        cost_matrix[ri, :] -= minimum_cost
    
    for ci in range(cost_matrix.shape[1]):
        minimum_cost = float("inf")
        for cost in cost_matrix[:, ci]:
            if cost < minimum_cost:
                minimum_cost = cost
        cost_matrix[:, ci] -= minimum_cost


def mark_cost_matrix(cost_matrix: np.ndarray) -> tuple[list, list, list]:
    bool_matrix_1: np.ndarray = cost_matrix == 0
    bool_matrix_2: np.ndarray = cost_matrix == 0

    marked_elements = []
    marked_element_rows = []

    # Mark elements
    is_marking_elements = True
    while is_marking_elements:
        min_zero_ri = -1
        min_zero_count = bool_matrix_1.shape[1]
        for ri, row in enumerate(bool_matrix_1):
            zero_count = np.count_nonzero(row)
            if zero_count != 0 and zero_count < min_zero_count:
                min_zero_ri = ri
                min_zero_count = zero_count
        
        if min_zero_ri != -1:
            min_zero_ci = np.where(bool_matrix_1[min_zero_ri, :])[0][0]
            bool_matrix_1[min_zero_ri, :] = False
            bool_matrix_1[:, min_zero_ci] = False
            marked_elements.append((min_zero_ri, min_zero_ci))
            marked_element_rows.append(min_zero_ri)
        else:
            is_marking_elements = False
    
    # Mark rows and columns
    non_marked_rows = list(set(range(bool_matrix_2.shape[0])) - set(marked_element_rows))
    marked_cols = []

    is_marking_rows_cols = True
    while is_marking_rows_cols:
        is_marking_rows_cols = False

        for ri in non_marked_rows:
            for ci, cost in enumerate(bool_matrix_2[ri, :]):
                if cost and ci not in marked_cols:
                    marked_cols.append(ci)
                    is_marking_rows_cols = True

        for ri, ci in marked_elements:
            if ri not in non_marked_rows and ci in marked_cols:
                non_marked_rows.append(ri)
                is_marking_rows_cols = True

    marked_rows = list(set(range(bool_matrix_2.shape[0])) - set(non_marked_rows))
    return marked_elements, marked_rows, marked_cols


def adjust_cost_matrix(cost_matrix: np.ndarray, marked_rows: list[int], marked_cols: list[int]):
    non_marked_rows = list(set(range(cost_matrix.shape[0])) - set(marked_rows))

    non_zero_costs = []
    non_zero_elements = []
    for ri in non_marked_rows:
        for ci, cost in enumerate(cost_matrix[ri, :]):
            if ci not in marked_cols:
                non_zero_costs.append(cost)
                non_zero_elements.append((ri, ci))
    minimum_cost = min(non_zero_costs)

    for ri, ci in non_zero_elements:
        cost_matrix[ri, ci] -= minimum_cost

    for ri in marked_rows:
        for ci in marked_cols:
            cost_matrix[ri, ci] += minimum_cost


def hungarian_algorithm(cost_matrix: np.ndarray) -> list[tuple[int, int]]:
    reduce_cost_matrix(cost_matrix)

    while True:
        marked_elements, marked_rows, marked_cols = mark_cost_matrix(cost_matrix)

        if len(marked_rows) + len(marked_cols) < cost_matrix.shape[0] and len(marked_rows) + len(marked_cols) < cost_matrix.shape[1]:
            adjust_cost_matrix(cost_matrix, marked_rows, marked_cols)
        else:
            return marked_elements
        
