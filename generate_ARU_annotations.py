import pickle
import numpy as np
import os
import random
import wave
import contextlib
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE 
from save_text import make_annotation_file, make_day_annotation_file
from imblearn.under_sampling import RandomUnderSampler
 
days = ['02', '04']
Project_path = '/content/drive/My Drive/Sciurid Lab/CNN/VGGish_Birds/'
#/input('Project path: ')

# Load training data from pickle files
path_here = os.path.join(Project_path, 'Data/birds_with_noise_single_notes_new.pickle')
with open(path_here, 'rb') as savef:
  audio_feats_data_training, species_training, num_vecs = np.transpose(np.array(pickle.load(savef)))
BIRDS_LIST = []
for i in range(audio_feats_data_training.shape[0]):
  toto = np.array(audio_feats_data_training[i], dtype = ('O')).astype(np.float)
  BIRDS_LIST.append(toto)
BIRDS = np.array(BIRDS_LIST)
sm = SMOTE(random_state = 2)
#X_train, y_train = sm.fit_sample(BIRDS, species_training)

rus = RandomUnderSampler(random_state=0)
X_train, y_train = rus.fit_resample(BIRDS, species_training)


fini_count = 0
cuce_count = 0
mofa_count = 0
poho_count = 0
phma_count = 0
gaso_count = 0
hyga_count = 0
noise_count = 0
for ioio in range(y_train.shape[0]):
  if y_train[ioio] == 'FINI':
    fini_count += 1
  elif y_train[ioio] == 'CUCE':
    cuce_count += 1
  elif y_train[ioio] == 'MOFA':
    mofa_count += 1
  elif y_train[ioio] == 'POHO':
    poho_count += 1
  elif y_train[ioio] == 'PHMA':
    phma_count += 1
  elif y_train[ioio] == 'GASO':
    gaso_count += 1
  elif y_train[ioio] == 'HYGA':
    hyga_count += 1
  elif y_train[ioio] == 'NOISE':
    noise_count += 1

print('FINI = {}\n CUCE = {}\n MOFA = {}\n POHO = {}\n PHMA = {}\n GASO = {}\n HYGA = {}\n Noise = {}'.format(fini_count, cuce_count, mofa_count, poho_count, phma_count, gaso_count, hyga_count, noise_count))

# Train classifier
clf = RandomForestClassifier(random_state=0, n_estimators=100)
clf.fit(X_train, y_train)

# Load sound files for annotations
for day in days:
  folder_name = Project_path + 'ARU_embeddings_no_overlap/' + day + '/'
  file_names = sorted(os.listdir(folder_name))
  print(['We are on day ' + day])

  save_folder = Project_path + 'ARU_annotations_no_overlap_undersample_notes/'
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
    day_label, audio_feats_data, time_stamp = wtf['day'], wtf['raw_audioset_feats_960ms'], wtf['time_stamp']
    
    species_prediction.append(clf.predict(audio_feats_data))
    species_prediction_day.append(np.asarray(species_prediction))
    species_prediction = np.transpose(np.asarray(species_prediction))
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
  #print(species_prediction_day[0, 0, :])
  save_path_day = save_folder + day + '.txt'
  make_day_annotation_file(save_path_day, species_prediction_day, num_preds_file, duration_files)