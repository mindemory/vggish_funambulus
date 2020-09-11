
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA
import matplotlib
from datetime import datetime
import os
import pickle
import calendar


'''
This module provides functions to assist with plotting our data
'''

def get_label_order(labels, lab_type):
  '''
  Get sensible order for labels
  '''

  reord = []
  if 'month' in lab_type:
    reord = [list(calendar.month_abbr).index(c) for c in labels]

  elif lab_type == 'land-use-ny':
    reord = np.ones(len(labels))*-1
    for ix, lb in enumerate(labels):
      if lb ==  '$\\leq 1.4$': reord[ix] = 0
      if lb == '1.4 - 1.7': reord[ix] = 1
      if lb == '1.7 - 2.0': reord[ix] = 2
      if lb == '2.0 - 2.3': reord[ix] = 3
      if lb ==  '$\\geq 2.3$': reord[ix] = 4
    for ix, r in enumerate(reord):
      if r == -1: reord[ix] = np.max(reord) + 1

  elif lab_type == 'land-use' :
    reord = np.ones(len(labels))*-1
    for ix, lb in enumerate(labels):
      if lb ==  '$\\leq 2.45$': reord[ix] = 0
      if lb == '2.45 - 2.6': reord[ix] = 1
      if lb == '2.6 - 2.75': reord[ix] = 2
      if lb ==  '$\\geq 2.75$': reord[ix] = 3
    for ix, r in enumerate(reord):
      if r == -1: reord[ix] = np.max(reord) + 1

  elif 'dataset' in lab_type:
    reord = np.ones(len(labels))*-1
    for ix, lb in enumerate(labels):
      if 'borneo' in lb.lower() or 'congo' in lb.lower() or 'sulawesi' in lb.lower():
        reord[ix] = np.max(reord) + 1

    for ix, r in enumerate(reord):
      if r == -1: reord[ix] = np.max(reord) + 1

  if len(reord) >= 1:
    reord = [int(i) for i in reord]
    return np.argsort(reord)
  else: return range(len(labels))

def plot_multi_class_recalls(recalls, labels, average_accuracy, cm_values, label_type, feat):
  '''
  Plot recall for each class as a result of a multiclass classification task

  Inputs:
      recalls (ndarray): vector of recalls for each class
      labels (ndarray): label corresponding to each class
      average_accuracy (float): balanced average recall across all classes
      label_type (str): type of label used (e.g. 'dataset', 'land-use', 'hour', 'month' etc.)
      feat (str): acoustic feature set used
  '''

  # Convert decimals to percentages
  recalls = recalls * 100
  average_accuracy = average_accuracy * 100

  # Get sensible order for labels
  order = get_label_order(labels,label_type)
  recalls = np.asarray(recalls)
  recalls = recalls[order]
  labels = labels[order]

  bar1 = plt.bar(labels, recalls)
  indexing = 0
  for rect in bar1:
    height = rect.get_height()
    #print(rect)
    plt.text(rect.get_x() + rect.get_width()/2.0, height - 5, 'tp = ' + str(round(cm_values[0][indexing])), ha='center', va='bottom')
    plt.text(rect.get_x() + rect.get_width()/2.0, height - 10, 'tn = ' + str(round(cm_values[1][indexing])), ha='center', va='bottom')
    plt.text(rect.get_x() + rect.get_width()/2.0, height - 15, 'fp = ' + str(round(cm_values[2][indexing])), ha='center', va='bottom')
    plt.text(rect.get_x() + rect.get_width()/2.0, height - 20, 'fn = ' + str(round(cm_values[3][indexing])), ha='center', va='bottom')
    indexing+=1
