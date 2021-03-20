
import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense

def get_model(neurons0, neurons1, activation_function, input_shape):
  model = Sequential()
  model.add(Dense(neurons0, activation = activation_function, input_shape = (input_shape, ))) 
  model.add(Dense(neurons1, activation = 'softmax'))
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