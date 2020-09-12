import pickle
import numpy as np
import os
import random

#species = ['CUCE', 'FINI', 'GASO', 'HYGA', 'MOFA', 'NOISE', 'PHMA', 'POHO']
species = ['dusky', 'ratufa', 'noise']
concatenated_pickle = []
Project_path = input('Project path: ')
for spp in species:
  folder_name = Project_path + 'squirrel_note_embeddings/' + spp + '/'
  file_names = os.listdir(folder_name)
  print(['We are in species ' + spp]) 
  #print(file_names)
  #if spp == 'Noise':
  #  file_names = random.sample(file_names, 100)
  for FILE in file_names:
    with open(os.path.join(folder_name, FILE), 'rb') as savef:
      wtf = pickle.load(savef)
      audio_feats_data, sp_name = wtf['raw_audioset_feats_960ms'], wtf['species']
    
    num_vecs = audio_feats_data.shape[0] # Taking time information from file
    for row in range(audio_feats_data.shape[0]):
      concatenated_pickle.append([audio_feats_data[row], sp_name, num_vecs])
    
save_folder = Project_path + '/Data/'    
save_file_name = save_folder + 'rnd.pickle'
with open(save_file_name, 'wb') as opo:
  pickle.dump(concatenated_pickle, opo)
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

Hi Mrugank, Mrugank here. Concatenating files is done, finally!"""

server.sendmail("mrugankdake@gmail.com", "mrugankdake@gmail.com", msg)
server.quit()