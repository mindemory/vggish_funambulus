from analysis_libs_funambulus_with_noise import multi_class_classification
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

# How many training test splits - recommend 5
k_folds = 10

# Figure setup
n_subplots_x = 1
n_subplots_y = 1
subplt_idx = 1

fig = plt.figure(figsize=(18,10))

ax = plt.gca()
Project_path = input('Project path: ')
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
  cm, cm_labs, average_acc, accuracies, cm_values = multi_class_classification(SQUIRRELS, species, k_fold=k_folds)
  
  plot_multi_class_recalls(accuracies, cm_labs, average_acc, cm_values, 'species', f)
  ax.set_title('Species classification')
  ax.set_xlabel("Squirrel species")
  ax.set_ylabel("F1 score")

png_name = 'rnd classification 24000.png'
save_path = os.path.join(Project_path, 'Figures', png_name)   
fig.savefig(save_path)
plt.show()

import smtplib

server = smtplib.SMTP('smtp.gmail.com', 587)
server.starttls()
server.login("mrugankdake@gmail.com", "MRUGank19@")
message = """From: From Person <from@fromdomain.com>
To: To Person <to@todomain.com>
Subject: SMTP e-mail test

This is a test e-mail message.
"""
msg = """ From: Mrugank Colab <mrugank@gmail.com>
To: Mrugank Colab <mrugank@gmail.com>
Subject: SMTP e-mail test

Hi Mrugank, Mrugank here. The 24000 random forest has been trained!"""

server.sendmail("mrugankdake@gmail.com", "mrugankdake@gmail.com", msg)
server.quit()


fig1 = plt.figure(figsize=(18,10))

ax1 = plt.gca()
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
  cm, cm_labs, average_acc, accuracies, cm_values = multi_class_classification_a(SQUIRRELS, species, k_fold=k_folds)
  
  plot_multi_class_recalls(accuracies, cm_labs, average_acc, cm_values, 'species', f)
  ax1.set_title('Species classification')
  ax1.set_xlabel("Squirrel species")
  ax1.set_ylabel("F1 score")

png_name = 'rnd classification 15000.png'
save_path = os.path.join(Project_path, 'Figures', png_name)   
fig1.savefig(save_path)
plt.show()

import smtplib

server = smtplib.SMTP('smtp.gmail.com', 587)
server.starttls()
server.login("mrugankdake@gmail.com", "MRUGank19@")
message = """From: From Person <from@fromdomain.com>
To: To Person <to@todomain.com>
Subject: SMTP e-mail test

This is a test e-mail message.
"""
msg = """ From: Mrugank Colab <mrugank@gmail.com>
To: Mrugank Colab <mrugank@gmail.com>
Subject: SMTP e-mail test

Hi Mrugank, Mrugank here. The 15000 random forest has been trained!"""

server.sendmail("mrugankdake@gmail.com", "mrugankdake@gmail.com", msg)
server.quit()

fig2 = plt.figure(figsize=(18,10))

ax2 = plt.gca()
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
  cm, cm_labs, average_acc, accuracies, cm_values = multi_class_classification_b(SQUIRRELS, species, k_fold=k_folds)
  
  plot_multi_class_recalls(accuracies, cm_labs, average_acc, cm_values, 'species', f)
  ax2.set_title('Species classification')
  ax2.set_xlabel("Squirrel species")
  ax2.set_ylabel("F1 score")

png_name = 'rnd classification 15000.png'
save_path = os.path.join(Project_path, 'Figures', png_name)   
fig2.savefig(save_path)
plt.show()

import smtplib

server = smtplib.SMTP('smtp.gmail.com', 587)
server.starttls()
server.login("mrugankdake@gmail.com", "MRUGank19@")
message = """From: From Person <from@fromdomain.com>
To: To Person <to@todomain.com>
Subject: SMTP e-mail test

This is a test e-mail message.
"""
msg = """ From: Mrugank Colab <mrugank@gmail.com>
To: Mrugank Colab <mrugank@gmail.com>
Subject: SMTP e-mail test

Hi Mrugank, Mrugank here. The 7000 random forest has been trained!"""

server.sendmail("mrugankdake@gmail.com", "mrugankdake@gmail.com", msg)
server.quit()

fig3 = plt.figure(figsize=(18,10))

ax3 = plt.gca()
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
  cm, cm_labs, average_acc, accuracies, cm_values = multi_class_classification_c(SQUIRRELS, species, k_fold=k_folds)
  
  plot_multi_class_recalls(accuracies, cm_labs, average_acc, cm_values, 'species', f)
  ax3.set_title('Species classification')
  ax3.set_xlabel("Squirrel species")
  ax3.set_ylabel("F1 score")

png_name = 'rnd classification 15000.png'
save_path = os.path.join(Project_path, 'Figures', png_name)   
fig3.savefig(save_path)
plt.show()

import smtplib

server = smtplib.SMTP('smtp.gmail.com', 587)
server.starttls()
server.login("mrugankdake@gmail.com", "MRUGank19@")
message = """From: From Person <from@fromdomain.com>
To: To Person <to@todomain.com>
Subject: SMTP e-mail test

This is a test e-mail message.
"""
msg = """ From: Mrugank Colab <mrugank@gmail.com>
To: Mrugank Colab <mrugank@gmail.com>
Subject: SMTP e-mail test

Hi Mrugank, Mrugank here. The 3000 random forest has been trained!"""

server.sendmail("mrugankdake@gmail.com", "mrugankdake@gmail.com", msg)
server.quit()