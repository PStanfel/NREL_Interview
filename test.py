from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from data import import_data

# load saved model
model = keras.models.load_model(".\\wind_direction_model.h5")

########################### Data Import ###########################

# data should be imported according to the below configuration of wind observation sites
# the targeted site to predict wind direction at is marked with a "T"
'''
    0   1   2
0   X   X   X           N
1   X   T   X       W       E
2   X   X   X           S
'''

data_path = 'data\\'
row_0_path = data_path + '4815dec2590432f992a317b2e7ed69ec\\'
row_1_path = data_path + '1efc54797915c2931df2e161f8e6ed14\\'
row_2_path = data_path + '288afbdd498141c7f270a74f8c455605\\'

row_data = import_data(row_0_path, row_1_path, row_2_path)
