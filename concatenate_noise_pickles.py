
import pickle
import numpy as np
import os
import random

species = ['dusky', 'ratufa', 'noise']
concatenated_pickle = []
#Project_path = input('Project path: ')
Project_path = '/content/drive/My Drive/Sciurid Lab/CNN/VGGish_Squirrels/'      
for spp in species:
  folder_name = Project_path + 'squirrel_note_embeddings_with_labels/' + spp + '/'
  file_names = os.listdir(folder_name)
  print(['We are in species ' + spp]) 
  #print(file_names)
  #if spp == 'Noise':
  #  file_names = random.sample(file_names, 100)
  for FILE in file_names:
    with open(os.path.join(folder_name, FILE), 'rb') as savef:
      wtf = pickle.load(savef)
      audio_feats_data, sp_name, file_name = wtf['raw_audioset_feats_960ms'], wtf['species'], wtf['file']
    
    num_vecs = audio_feats_data.shape[0] # Taking time information from file
    for row in range(audio_feats_data.shape[0]):
      concatenated_pickle.append([audio_feats_data[row], sp_name, file_name, num_vecs])
    
save_folder = Project_path + '/Data/'    
save_file_name = save_folder + 'rnd_with_labels.pickle'
with open(save_file_name, 'wb') as opo:
  pickle.dump(concatenated_pickle, opo)