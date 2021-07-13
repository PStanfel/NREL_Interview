import os
import numpy as np

def import_data(row_0_path, row_1_path, row_2_path):
    '''
    Method that imports data in the format required for model training. It is assumed that
    data inside the folders is situated such that iterating through the folder moves
    West to East

    Args:
    row_0_path (str): Path to folder of .csv files corresponding to the furthest North row.

    row_1_path (str): Path to folder of .csv files corresponding to the middle row.

    row_2_path (str): Path to folder of .csc files corresponding to the furthest South row. 
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

def manipulate_data(row_data, target_index, indices, horizon, time_step):
    '''
    Manipulates data to be in correct format for model training.

    Args:
    row_data (arr): Array of arrays of row data for each of the nine sites.

    target_index (int): Integer specifying which index the target (ie wind direction) is at
        in the csv files.

    indices (arr): Array of integers for selecting data from csv files.

    horizon (double): Value specifying length of standard deviation window in minutes.

    time_step (int): Value specifying length of time step in data in minutes.
    '''

    # initialize X with correct data size
    shape = np.shape(row_data[0][0])
    X = np.zeros((shape[0],1))

    for i in range(len(row_data)):
        for j in range(len(row_data[i])):
            X = np.concatenate((X, row_data[i][j][:,indices]), axis=1)

    # remove initialization column
    X = X[:,1:]

    # calculate indices to shift
    shift_index = int(horizon / time_step)

    # targets are wind direction at center site
    y = row_data[1][1][:,target_index]
    y = np.resize(y, (shape[0],1))
    y = np.concatenate((y, np.zeros_like(y)), axis=1)

    # calculate standard deviation using defined window
    for i in range(shape[0] - shift_index):
        y[i+shift_index][1] = np.std(y[i:i+shift_index][:,0])

    # shift data
    y = y[shift_index:]

    X = X[:shape[0]-shift_index,:]

    return X, y