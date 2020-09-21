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
from analysis_libs_funambulus_with_noise import rf_classifier_aru
 
#days = ['02', '04']
days = ['ratufa_whole_day', 'dusky']
Project_path = '/content/drive/My Drive/Sciurid Lab/CNN/VGGish_Squirrels/'
#/input('Project path: ')
threshold = 0.5
noise_value = 5000
# Load training data from pickle files
path_here = os.path.join(Project_path, 'Data/rnd.pickle')
with open(path_here, 'rb') as savef:
  audio_feats_data_training, species_training, num_vecs = np.transpose(np.array(pickle.load(savef)))
SQUIRRELS_LIST = []
for i in range(audio_feats_data_training.shape[0]):
  toto = np.array(audio_feats_data_training[i], dtype = ('O')).astype(np.float)
  SQUIRRELS_LIST.append(toto)
SQUIRRELS = np.array(SQUIRRELS_LIST)
print(np.unique(species_training))

clf = rf_classifier_aru(SQUIRRELS, species_training, noise_value)
#sm = SMOTE(random_state = 2)
#X_train, y_train = sm.fit_sample(BIRDS, species_training)

#rus = RandomUnderSampler(random_state=0)
#X_train, y_train = rus.fit_resample(BIRDS, species_training)

# Train regressor
#species_training[species_training == 'NOISE'] = 'AAA'
#enc = OneHotEncoder(categories = 'auto', sparse = False, handle_unknown = 'error')
#y_train = enc.fit_transform(species_training.reshape(species_training.shape[0], 1))
  
#clf = RandomForestRegressor(random_state=0, n_estimators=100)
#clf.fit(BIRDS, y_train)

# Load sound files for annotations
for day in days:
  folder_name = Project_path + 'ARU_embeddings/' + day + '/'
  file_names = sorted(os.listdir(folder_name))
  print(file_names[0:3])
  print(['We are on day ' + day])

  save_folder = Project_path + 'Annotations/rnd_classifier_5000_100/'
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
    wav_file_path = os.path.join(Project_path, 'ARU_Test', day, FILE[:-7] + '.wav')
    with open(pickle_file_path, 'rb') as savef:
      wtf = pickle.load(savef)
    audio_feats_data, time_stamp = wtf['raw_audioset_feats_960ms'], wtf['file_name']
    predictions = clf.predict(audio_feats_data)
    #predictions = Binarizer(threshold = threshold).fit_transform(predictions)
    #predictions_cat = enc.inverse_transform(predictions)
    #predictions_cat[predictions_cat == 'AAA'] = 'NOISE'
    #predictions_cat = predictions_cat.flatten()
    #species_prediction.append(predictions_cat)
    species_prediction.append(predictions)
    
    species_prediction = np.transpose(np.asarray(species_prediction))
    species_prediction_day.append(np.asarray(species_prediction))
    #species_prediction[species_prediction == 'AAA'] = 'NOISE'
    num_preds += species_prediction.shape[0]
    num_preds_file.append(num_preds)
    save_path = save_folder + day + '/' + time_stamp +  '.txt'
    make_annotation_file(save_path, species_prediction)
    with contextlib.closing(wave.open(wav_file_path,'r')) as f:
      frames = f.getnframes()
      rate = f.getframerate()
    duration += frames / float(rate)
    duration_files.append(duration)
  species_prediction_day = np.asarray(species_prediction_day)
  #species_prediction_day[species_prediction_day == 'AAA'] = 'NOISE'
  #print(species_prediction_day[0, 0, :])
  save_path_day = save_folder + day + '.txt'
  make_day_annotation_file(save_path_day, species_prediction_day, num_preds_file, duration_files)