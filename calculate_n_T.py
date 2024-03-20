from math import comb
import time
import numpy as np


def all_row_configurations(n_containers, n_ports, configuration=[]):
    if n_ports == 1:
        yield configuration + [n_containers]
    else:
        for to_first_port in range(n_containers + 1):
            yield from all_row_configurations(
                n_containers - to_first_port,
                n_ports - 1,
                configuration + [to_first_port],
            )


def count_possible_transportation(port, matrix):
    width = len(matrix[0])
    if port == width - 1:
        yield np.array(matrix)[1:, 1:]
    else:
        sum_of_column = sum(row[port] for row in matrix)
        for row in all_row_configurations(sum_of_column, width - port - 1):
            new_matrix = matrix + [([0] * (port + 1)) + row]
            yield from count_possible_transportation(port + 1, new_matrix)


count = 0
for matrix in count_possible_transportation(0, [[12, 0, 0, 0, 0, 0]]):
    # print(matrix)
    # time.sleep(0.4)
    count += 1

print(count)
