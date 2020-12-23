
import time     
start_time = time.time()
import pickle
import numpy as np
import os
import random
import wave
import contextlib
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import Binarizer
from imblearn.over_sampling import SMOTE 
from save_text import make_annotation_file, make_day_annotation_file
from imblearn.under_sampling import RandomUnderSampler
from analysis_libs_funambulus_with_noise import rf_classifier_aru, rf_classifier_aru_simple  

NOISE = [3000, 5000, 10000]        
TREES = [100, 300, 500, 1000]   
DEPTH = [50, 80, 100, 120]      
#SAMPLESPLIT = 2#, 5, 10, 15, 20, 50, 100]       
#max_depth = 50  
#samplesleaf = 1 
#bootstrap = True        
randomstate = 0 
classweight = ['balanced', 'balanced_subsample']
for classw in classweight:     
  for depth in DEPTH: 
    for tree_count in TREES:      
      for noise_value in NOISE:   
        print('We are in samples depth = {}, tree_count = {}, noise_value = {}'.format(depth, tree_count, noise_value))     
        days = ['ratufa_whole_day', 'dusky']
        Project_path = '/content/drive/My Drive/Sciurid Lab/CNN/VGGish_Squirrels/'
        #/input('Project path: ')
        threshold = 0.5
        # Load training data from pickle files
        path_here = os.path.join(Project_path, 'Data/rnd19_with_labels.pickle')
        with open(path_here, 'rb') as savef:
          audio_feats_data_training, species_training, file_name, num_vecs = np.transpose(np.array(pickle.load(savef)))
        SQUIRRELS_LIST = []
        for i in range(audio_feats_data_training.shape[0]):
          toto = np.array(audio_feats_data_training[i], dtype = ('O')).astype(np.float)
          SQUIRRELS_LIST.append(toto)
        SQUIRRELS = np.array(SQUIRRELS_LIST)
        #clf = rf_classifier_aru(SQUIRRELS, species_training, noise_value, tree_count, max_depth, samplesplit, samplesleaf, bootstrap, randomstate, classweight)
        clf = rf_classifier_aru(SQUIRRELS, species_training, noise_value, tree_count, depth, randomstate, classw)   

        # Load sound files for annotations
        for day in days:
          folder_name = Project_path + 'ARU_embeddings/' + day + '/'
          file_names = sorted(os.listdir(folder_name))
          #print(file_names[0:3])
          #print(['We are on day ' + day])

          save_folder = Project_path+'Detections19/'+str(noise_value)+'noise_'+str(tree_count)+'trees'+str(depth)+'classweight'+classw+'_classifier/' 
          if not os.path.exists(save_folder):     
            os.mkdir(save_folder) 
          day_fold = save_folder + day +'/'
          if not os.path.exists(day_fold):
            os.mkdir(day_fold)

          duration_files = [0]
          num_preds = 0
          species_prediction_day = []
          num_preds_file = []
          duration = 0

          # Make annotations
          for FILE in file_names:
            species_prediction = []
            pickle_file_path = os.path.join(folder_name, FILE)
            with open(pickle_file_path, 'rb') as savef:
              wtf = pickle.load(savef)
            audio_feats_data, time_stamp = wtf['raw_audioset_feats_960ms'], wtf['file_name']
            predictions = clf.predict(audio_feats_data)
            predictions = np.asarray(predictions)
            species_prediction.append(predictions)
            species_prediction = np.transpose(np.asarray(species_prediction))
            species_prediction_day.append(np.asarray(species_prediction))
            #species_prediction[species_prediction == 'AAA'] = 'NOISE'
            num_preds += species_prediction.shape[0]
            num_preds_file.append(num_preds)

            save_path = save_folder + day + '/' + time_stamp +  '.txt'
            make_annotation_file(save_path, species_prediction)
            wav_file_path = os.path.join(Project_path, 'ARU_Test', day, FILE[:-7] + '.wav')	
            with contextlib.closing(wave.open(wav_file_path,'r')) as f:	
              frames = f.getnframes()	
              rate = f.getframerate()	
            duration += frames / float(rate)	
            duration_files.append(duration)	
          
          species_prediction_day = np.asarray(species_prediction_day)	
          #print(species_prediction_day.shape)
          save_path_day = save_folder + day + '.txt'	
          make_day_annotation_file(save_path_day, species_prediction_day, num_preds_file, duration_files)
print(time.time() - start_time)