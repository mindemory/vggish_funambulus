
import numpy as np
import umap
from sklearn import preprocessing
from sklearn.cluster import AffinityPropagation
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import Binarizer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import balanced_accuracy_score, f1_score
import calendar
import collections
import heapq
from operator import itemgetter
from imblearn.under_sampling import RandomUnderSampler


'''
This module provides functions to assist with analysis of our data
'''

def multi_class_classification(X, y, k_fold = 5):
  '''
  Do a multiclass classification task using a random forest classifier
  Accuracy is measured using f1 score

  Inputs:
    X (ndarray): feature data
    y (ndarray): labels associated with feature data
    k_fold (int): number of cross-fold validation runs to use

  Returns:
    (All of the below are averaged from cross-fold validation results)
    cm (ndarray): confusion matrix of results
    cm_labels (ndarray): labels for the confusion matrix
    average_accuracy (float): average accuracy across all classes
    accuracies (ndarray): individual accuracies for each class
  '''
  X = np.asarray(X)
  y = np.asarray(y)

  # dividing X, y into train and test data
  sss = StratifiedShuffleSplit(n_splits=k_fold, test_size=0.2, random_state=0)

  # Do K fold cross validation
  all_cms = []
  all_accuracies = []
  tp_array = []
  fp_array = []
  tn_array = []
  fn_array = []
  print('Doing {} fold cross validation predictions. Classes: {}'.format(k_fold,np.unique(y)))
  for k, (train_index, test_index) in enumerate(sss.split(X, y)):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    #X_train_corrected = 
    species = ['noise', 'ratufa', 'dusky']
    train_res = {}
    test_res = {}
    for sp in species:
      train_res[sp] = 0
      test_res[sp] = 0

    for i in y_train:
      train_res[i] += 1
    
    for i in y_test:
      test_res[i] += 1

    print("Training set = {}".format(train_res))
    print("Testing set = {}".format(test_res))
    # training a classifier
    clf = RandomForestClassifier(random_state=0, n_estimators=100)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    # model accuracy for X_test
    class_scores = f1_score(y_test,predictions,average=None)
    print('{}/{} folds mean accuracy: {}'.format(k+1,k_fold,np.mean(class_scores)))
    all_accuracies.append(class_scores)

    cm_labels = np.unique(y)
    k_cm = confusion_matrix(y_test, predictions, labels=cm_labels)
    FP = k_cm.sum(axis=0) - np.diag(k_cm)  
    FN = k_cm.sum(axis=1) - np.diag(k_cm)
    TP = np.diag(k_cm)
    TN = k_cm.sum().sum() - (FP + FN + TP)  
    tp_array.append(TP)
    fp_array.append(FP)
    tn_array.append(TN)
    fn_array.append(FN)    
    all_cms.append(k_cm)

  # Get averages across K fold cross validation
  final_tp = np.mean(np.asarray(tp_array), axis = 0)
  final_tn = np.mean(np.asarray(tn_array), axis = 0)
  final_fp = np.mean(np.asarray(fp_array), axis = 0)
  final_fn = np.mean(np.asarray(fn_array), axis = 0)
  cm_values = [final_tp, final_tn, final_fp, final_fn]
  accuracies = np.mean(np.asarray(all_accuracies),axis=0)
  average_accuracy = np.mean(accuracies)
  print('Average accuracy = {}'.format(average_accuracy))

  cm = np.mean(np.asarray(all_cms),axis=0)

  return cm, cm_labels, average_accuracy, accuracies, cm_values

def multi_class_classification_a(X, y, value, k_fold):
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
  
  # dividing X, y into train and test data
  sss = StratifiedShuffleSplit(n_splits=k_fold, test_size=0.2, random_state=0)

  # Do K fold cross validation
  all_cms = []
  all_accuracies = []
  tp_array = []
  fp_array = []
  tn_array = []
  fn_array = []
  print('Doing {} fold cross validation predictions. Classes: {}'.format(k_fold,np.unique(y)))
  for k, (train_index, test_index) in enumerate(sss.split(X, y)):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    #X_train_corrected = 
    species = [['noise', 'ratufa', 'dusky']]
    train_res = {}
    test_res = {}
    for sp in species:
      train_res[sp] = 0
      test_res[sp] = 0

    for i in y_train:
      train_res[i] += 1
    
    for i in y_test:
      test_res[i] += 1

    print("Training set = {}".format(train_res))
    print("Testing set = {}".format(test_res))
    # training a classifier
    clf = RandomForestClassifier(random_state=0, n_estimators=200)#, max_features = 100)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    # model accuracy for X_test
    class_scores = f1_score(y_test,predictions,average=None)
    print('{}/{} folds mean accuracy: {}'.format(k+1,k_fold,np.mean(class_scores)))
    all_accuracies.append(class_scores)

    cm_labels = np.unique(y)
    k_cm = confusion_matrix(y_test, predictions, labels=cm_labels)
    FP = k_cm.sum(axis=0) - np.diag(k_cm)  
    FN = k_cm.sum(axis=1) - np.diag(k_cm)
    TP = np.diag(k_cm)
    TN = k_cm.sum().sum() - (FP + FN + TP)  
    tp_array.append(TP)
    fp_array.append(FP)
    tn_array.append(TN)
    fn_array.append(FN)    
    all_cms.append(k_cm)

  # Get averages across K fold cross validation
  final_tp = np.mean(np.asarray(tp_array), axis = 0)
  final_tn = np.mean(np.asarray(tn_array), axis = 0)
  final_fp = np.mean(np.asarray(fp_array), axis = 0)
  final_fn = np.mean(np.asarray(fn_array), axis = 0)
  cm_values = [final_tp, final_tn, final_fp, final_fn]
  accuracies = np.mean(np.asarray(all_accuracies),axis=0)
  average_accuracy = np.mean(accuracies)
  print('Average accuracy = {}'.format(average_accuracy))

  cm = np.mean(np.asarray(all_cms),axis=0)

  return cm, cm_labels, average_accuracy, accuracies, cm_values

def random_forest_regressor(X, y, threshold, value, k_fold = 5):
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
  np.random.seed(0)
  chosen_data_pos = np.random.choice(data_pos, value)#, random_state = 0)
  
  for pos in chosen_data_pos:
    new_X_noise.append(X_noise[pos])
    new_y_noise.append(y_noise[pos])

  X_normal = np.asarray(X_normal)
  new_X_noise = np.asarray(new_X_noise)
  y_normal = np.asarray(y_normal)
  new_y_noise = np.asarray(new_y_noise)
  
  X = np.concatenate((X_normal, new_X_noise), axis = 0)
  y = np.concatenate((y_normal, new_y_noise), axis = 0)
  y[y == 'dusky'] = 'usky'
  y_unique = np.unique(y)
  enc = OneHotEncoder(categories = 'auto', sparse = False, handle_unknown = 'error')
  y_regressor = enc.fit_transform(y.reshape(y.shape[0], 1))
  
  # dividing X, y into train and test data
  sss = StratifiedShuffleSplit(n_splits=k_fold, test_size=0.2, random_state=0)

  # Do K fold cross validation
  all_cms = []
  all_accuracies = []
  tp_array = []
  fp_array = []
  tn_array = []
  fn_array = []
  print('Doing {} fold cross validation predictions. Classes: {}'.format(k_fold,np.unique(y)))
  for k, (train_index, test_index) in enumerate(sss.split(X, y_regressor)):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y_regressor[train_index], y_regressor[test_index]
    y_test_cat = enc.inverse_transform(y_test)
    # training a classifier
    clf = RandomForestRegressor(random_state=0, n_estimators=500)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    predictions = Binarizer(threshold = threshold).fit_transform(predictions)
    predictions_cat = enc.inverse_transform(predictions)
    y_test_cat[y_test_cat == 'usky'] = 'dusky'
    predictions_cat[predictions_cat == 'usky'] = 'dusky'
    # model accuracy for X_test
    class_scores = f1_score(y_test_cat,predictions_cat,average=None)
    print('{}/{} folds mean accuracy: {}'.format(k+1,k_fold,np.mean(class_scores)))
    all_accuracies.append(class_scores)
    y[y == 'usky'] = 'dusky'
    cm_labels = np.unique(y)
    k_cm = confusion_matrix(y_test_cat, predictions_cat, labels=cm_labels)
    FP = k_cm.sum(axis=0) - np.diag(k_cm)  
    FN = k_cm.sum(axis=1) - np.diag(k_cm)
    TP = np.diag(k_cm)
    TN = k_cm.sum().sum() - (FP + FN + TP)  
    tp_array.append(TP)
    fp_array.append(FP)
    tn_array.append(TN)
    fn_array.append(FN)    
    all_cms.append(k_cm)

  # Get averages across K fold cross validation
  final_tp = np.mean(np.asarray(tp_array), axis = 0)
  final_tn = np.mean(np.asarray(tn_array), axis = 0)
  final_fp = np.mean(np.asarray(fp_array), axis = 0)
  final_fn = np.mean(np.asarray(fn_array), axis = 0)
  cm_values = [final_tp, final_tn, final_fp, final_fn]
  accuracies = np.mean(np.asarray(all_accuracies),axis=0)
  average_accuracy = np.mean(accuracies)
  print('Average accuracy = {}'.format(average_accuracy))

  cm = np.mean(np.asarray(all_cms),axis=0)

  return cm, cm_labels, average_accuracy, accuracies, cm_values

def rf_classifier_aru(X, y, value):
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
  np.random.seed(0)
  chosen_data_pos = np.random.choice(data_pos, value)
  
  for pos in chosen_data_pos:
    new_X_noise.append(X_noise[pos])
    new_y_noise.append(y_noise[pos])

  X_normal = np.asarray(X_normal)
  new_X_noise = np.asarray(new_X_noise)
  y_normal = np.asarray(y_normal)
  new_y_noise = np.asarray(new_y_noise)
  
  X_train = np.concatenate((X_normal, new_X_noise), axis = 0)
  y_train = np.concatenate((y_normal, new_y_noise), axis = 0)
  
  species = ['noise', 'ratufa', 'dusky']
  train_res = {}
  for sp in species:
    train_res[sp] = 0
  
  for i in y_train:
    train_res[i] += 1
  print("Training set = {}".format(train_res))
  clf = RandomForestClassifier(random_state=0, n_estimators=200)#, max_features = 100)
  clf.fit(X_train, y_train)
  return clf