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
  sss = StratifiedShuffleSplit(n_splits=k_fold, test_size=0.3, random_state=0)

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
    mofa_train = 0
    fini_train = 0
    cuce_train = 0
    phma_train = 0
    gaso_train = 0
    poho_train = 0
    hyga_train = 0
    noise_train = 0
    for i in y_train:
      if i == 'MOFA':
        mofa_train += 1
      elif i == 'FINI':
        fini_train += 1
      elif i == 'CUCE':
        cuce_train += 1
      elif i == 'PHMA':
        phma_train += 1
      elif i == 'GASO':
        gaso_train += 1
      elif i == 'POHO':
        poho_train += 1
      elif i == 'NOISE':
        noise_train += 1
      elif i == 'HYGA':
        hyga_train += 1
    
    mofa_test = 0
    fini_test = 0
    cuce_test = 0
    phma_test = 0
    gaso_test = 0
    poho_test = 0
    hyga_test = 0
    noise_test = 0
    for i in y_test:
      if i == 'MOFA':
        mofa_test += 1
      elif i == 'FINI':
        fini_test += 1
      elif i == 'CUCE':
        cuce_test += 1
      elif i == 'PHMA':
        phma_test += 1
      elif i == 'GASO':
        gaso_test += 1
      elif i == 'POHO':
        poho_test += 1
      elif i == 'NOISE':
        noise_test += 1
      elif i == 'HYGA':
        hyga_test += 1

    print("mofa_train = {}, fini_train = {}, cuce_train = {}, phma_train = {}, poho_train = {}, noise_train = {}, hyga_train = {}, gaso_train = {}".format(mofa_train, fini_train, cuce_train, phma_train, poho_train, noise_train, hyga_train, gaso_train))
    print("mofa_test = {}, fini_test = {}, cuce_test = {}, phma_test = {}, poho_test = {}, noise_test = {}, hyga_test = {}, gaso_test{}".format(mofa_test, fini_test, cuce_test, phma_test, poho_test, noise_test, hyga_test, gaso_test))
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