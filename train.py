from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from data import import_data, manipulate_data

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
row_0_path = data_path + '6e629dbdf103136d68dfac978020d5e4\\'
row_1_path = data_path + 'd51a8eff0736f44bcfaea7195c00a80a\\'
row_2_path = data_path + '0fcb7c7e955e55af9376820c9ba9824f\\'

row_data = import_data(row_0_path, row_1_path, row_2_path)

# code for plotting wind direction at site (1,1)
# plt.rcParams.update({'font.size': 22})
# plt.plot(row_data[1][1][:,6])
# plt.xlabel("Time")
# plt.ylabel("Wind Direction (deg.)")
# plt.title("Wind Direction vs. Time, Site (1,1)")
# plt.show()

########################### Data Manipulation ###########################
# indices 5,6, and 8 correspond to wind direction, wind speed, and pressure, respectively
ws_index = 5
wd_index = 6
pr_index = 8
indices = [ws_index, wd_index, pr_index]

# define time horizon in minutes
horizon = 10

# define time step in minutes
time_step = 5

X, y = manipulate_data(row_data, wd_index, indices, horizon, time_step)

# if testing dataset is not available, train_test_split can be used to generate one
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

X_train = X
y_train = y

########################### Model Training ###########################
# create neural network
layer0 = Dense(20, input_shape=(len(indices)*len(row_data)*len(row_data[0]),), activation='tanh')
layer1 = Dense(20, activation='relu')
layer2 = Dense(2, activation='relu')
layers = [layer0, layer1, layer2]
model = keras.Sequential(layers)

# compile and train model
model.compile(optimizer='Adam', loss='mae', metrics=['mean_absolute_error', 'mean_squared_error'] )
model.fit(X_train, y_train, epochs=20)

# save model
#model.save(".\\wind_direction_model.h5")