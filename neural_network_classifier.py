
import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense

def get_model(neurons0, neurons1, neurons2, activation_function0, activation_function1, input_shape):
  model = Sequential()
  model.add(Dense(neurons0, activation = activation_function0, input_shape = (input_shape, )))
  model.add(Dense(neurons1, activation = activation_function1))
  model.add(Dense(neurons2, activation = 'softmax'))
  return model    

def classifier(neurons0, neurons1, activation_function, input_shape, strategy):
  with strategy.scope():
    model = Sequential()
    model.add(Dense(neurons0, activation = activation_function, input_shape = (input_shape, ))) 
    model.add(Dense(neurons1, activation = 'softmax'))
    
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
  print(model.summary())
  return model

def train_model(neurons0, neurons1, activation_function, X_train, y_train, strategy):
  model = classifier(neurons0, neurons1, activation_function, X_train.shape[1], strategy)
  model.fit(X_train, y_train, epochs = epochs, steps_per_epoch = steps_per_epoch, verbose = 2)
  return model

def noise_reducer(X, y, value):
  X = np.asarray(X)
  y = np.asarray(y)
  X_noise = []
  X_normal = []
  y_noise = []
  y_normal = []
  new_X_noise = []
  new_y_noise = []
  for i in range(y.shape[0]):
    if y[i] == 'noise':
      X_noise.append(X[i])
      y_noise.append(y[i])
    else:
      X_normal.append(X[i])
      y_normal.append(y[i])
  
  data_pos = np.arange(len(y_noise))
  chosen_data_pos = np.random.choice(data_pos, value)
  
  for pos in chosen_data_pos:
    new_X_noise.append(X_noise[pos])
    new_y_noise.append(y_noise[pos])

  X_normal = np.asarray(X_normal)
  new_X_noise = np.asarray(new_X_noise)
  y_normal = np.asarray(y_normal)
  new_y_noise = np.asarray(new_y_noise)
  
  X = np.concatenate((X_normal, new_X_noise), axis = 0)
  y = np.concatenate((y_normal, new_y_noise), axis = 0)
  return X, y