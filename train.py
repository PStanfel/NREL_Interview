from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from data import import_data

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

########################### Data Manipulation ###########################
# indices 5,6, and 8 correspond to wind direction, wind speed, and pressure, respectively
ws_index = 5
wd_index = 6
pr_index = 8
indices = [ws_index, wd_index, pr_index]

# initialize X with correct data size
shape = np.shape(row_data[0][0])
X = np.zeros((shape[0],1))

for i in range(len(row_data)):
    for j in range(len(row_data[i])):
        X = np.concatenate((X, row_data[i][j][:,indices]), axis=1)

# remove initialization column
X = X[:,1:]

# define time horizon in minutes
horizon = 30

# define time step in minutes
time_step = 5

# calculate indices to shift
shift_index = int(horizon / time_step)

# targets are wind direction at center site
y = row_data[1][1][:,wd_index]
y = np.resize(y, (shape[0],1))
y = np.concatenate((y, np.zeros_like(y)), axis=1)

for i in range(shape[0] - shift_index):
    y[i+shift_index][1] = np.std(y[i:i+shift_index][:,0])

# shift data
y = y[shift_index:]

X = X[:shape[0]-shift_index,:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

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

