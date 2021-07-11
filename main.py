import numpy as np
import matplotlib.pyplot as plt
import os

########################### Data Import ###########################

# data should be imported according to the below configuration of wind observation sites
'''
    0   1   2
0   X   X   X           N
1   X   X   X       W       E
2   X   X   X           S
'''

data_path = 'data\\'

row_0_path = data_path + '6e629dbdf103136d68dfac978020d5e4\\'
row_1_path = data_path + 'd51a8eff0736f44bcfaea7195c00a80a\\'
row_2_path = data_path + '0fcb7c7e955e55af9376820c9ba9824f\\'
row_paths = [row_0_path, row_1_path, row_2_path]

row_0_data = []
row_1_data = []
row_2_data = []
row_data = [row_0_data, row_1_data, row_2_data]

# fill data using numpy arrays
for i,row_path in enumerate(row_paths):
    for file_name in os.scandir(row_path):
        row_data[i].append(np.genfromtxt(file_name.path, skip_header=2, delimiter=','))

########################### Data Manipulation ###########################