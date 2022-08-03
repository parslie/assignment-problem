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


def generate_assignments(cost_matrix: np.ndarray) -> list[tuple[int, int]]:
    reduce_cost_matrix(cost_matrix)
    print(f"Reduced cost matrix:\n{cost_matrix}\n")

    while True:
        boolean_matrix = cost_matrix == 0

        # Mark elements on rows with minimum amount of zeroes
        marked_element_rows = []
        marked_elements = []
        while True:
            min_zero_count = boolean_matrix.shape[1]
            min_zero_ri = -1
            for ri in range(boolean_matrix.shape[0]):
                zero_count = 0
                for cost in boolean_matrix[ri, :]:
                    if cost == True:
                        zero_count += 1
                if zero_count != 0 and zero_count < min_zero_count:
                    min_zero_count = zero_count
                    min_zero_ri = ri

            # There are no more zeroes
            if min_zero_ri == -1:
                break

            # Mark zeroed element on row
            min_zero_ci = np.where(boolean_matrix[min_zero_ri, :] == True)[0][0]  # Get first column that has a zero
            boolean_matrix[min_zero_ri, :] = False  # Remove zeroes from row
            boolean_matrix[:, min_zero_ci] = False  # Remove zeroes from column
            marked_elements.append((min_zero_ri, min_zero_ci))
            marked_element_rows.append(min_zero_ri)
        
        # Mark rows and columns that has marked elements minimally
        boolean_matrix = cost_matrix == 0
        marked_cols = []
        non_marked_rows = list(set(range(boolean_matrix.shape[0])) - set(marked_element_rows))

        while True:
            will_break = True

            for ri in non_marked_rows:
                for ci, cost in enumerate(boolean_matrix[ri, :]):
                    if cost and ci not in marked_cols:
                        marked_cols.append(ci)
                        will_break = False
            
            for ri, ci in marked_elements:
                if ri not in non_marked_rows and ci in marked_cols:
                    non_marked_rows.append(ri)
                    will_break = False

            if will_break:
                break

        marked_rows = list(set(range(boolean_matrix.shape[0])) - set(non_marked_rows))

        # Return marked elements if all rows can be assigned to a column OR vice versa
        if len(marked_rows) + len(marked_cols) == cost_matrix.shape[0] or \
                len(marked_rows) + len(marked_cols) == cost_matrix.shape[1]:
            return marked_elements

        # Adjust cost matrix
        non_zero_elements = []
        non_zero_costs = []
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

        print(f"Adjusted cost matrix:\n{cost_matrix}\n")
