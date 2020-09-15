
from analysis_libs_funambulus_with_noise import multi_class_classification, multi_class_classification_a, random_forest_regressor
from plot_libs_funambulus import plot_multi_class_recalls
import matplotlib.pyplot as plt
import matplotlib
import argparse
import numpy as np
import os
import pickle

'''
Multiclass classification problems using eco-acoustic features
'''

matplotlib.rcParams.update({'font.size': 24})

feats = ['raw_audioset_feats_960ms']

value = 5000
# How many training test splits - recommend 5
k_folds = 10
threshold = 0.5
# Figure setup
n_subplots_x = 1
n_subplots_y = 1
subplt_idx = 1

fig = plt.figure(figsize=(18,10))

ax = plt.gca()
#Project_path = input('Project path: ')
Project_path = '/content/drive/My Drive/Sciurid Lab/CNN/VGGish_Squirrels/'
for f in feats:
  # Load data from pickle files
  path_here = os.path.join(Project_path, 'Data/rnd.pickle')
  with open(path_here, 'rb') as savef:
    squirrels = pickle.load(savef)
  squirrels = np.transpose(np.array(squirrels))
  audio_feats_data, species, num_vecs = squirrels
  SQUIRRELS_LIST = []
  for i in range(audio_feats_data.shape[0]):
    toto = np.array(audio_feats_data[i], dtype = ('O')).astype(np.float)
    SQUIRRELS_LIST.append(toto)
  SQUIRRELS = np.array(SQUIRRELS_LIST)
  #cm, cm_labs, average_acc, accuracies, cm_values = multi_class_classification_a(SQUIRRELS, species, value, k_folds)
  cm, cm_labs, average_acc, accuracies, cm_values = random_forest_regressor(SQUIRRELS, species, threshold, k_folds)
  
  plot_multi_class_recalls(accuracies, cm_labs, average_acc, cm_values, 'species', f)
  ax.set_title('Species classification')
  ax.set_xlabel("Squirrel species")
  ax.set_ylabel("F1 score")

png_name = 'rnd 200 trees classification ' + str(value) + '.png'
save_path = os.path.join(Project_path, 'Figures', png_name)   
#fig.savefig(save_path)
plt.show()