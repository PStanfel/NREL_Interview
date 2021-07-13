import os
import numpy as np

def import_data(row_0_path, row_1_path, row_2_path):
    '''
    Method that imports data in the format required for model training. It is assumed that
    data inside the folders is situated such that iterating through the folder moves
    West to East

    Args:
    row_0_path (str): Path to folder of .csv files corresponding to the furthest North row

    row_1_path (str): Path to folder of .csv files corresponding to the middle row

    row_2_path (str): Path to folder of .csc files corresponding to the furthest South row 
    '''

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

def manipulate_data(row_data, indices, horizon, time_step):
