import os
import numpy as np

def import_data(row_0_path, row_1_path, row_2_path):

    row_paths = [row_0_path, row_1_path, row_2_path]

    row_0_data = []
    row_1_data = []
    row_2_data = []
    row_data = [row_0_data, row_1_data, row_2_data]

    # fill data using numpy arrays
    for i,row_path in enumerate(row_paths):
        for file_name in os.scandir(row_path):
            row_data[i].append(np.genfromtxt(file_name.path, skip_header=2, delimiter=','))

    return row_data