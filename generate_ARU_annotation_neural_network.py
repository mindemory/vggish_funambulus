
import time     
start_time = time.time()
import pickle
import numpy as np
import os
import random
import wave
import pandas as pd
import contextlib
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import Binarizer
from imblearn.over_sampling import SMOTE 
from save_text import make_annotation_file, make_day_annotation_file
from imblearn.under_sampling import RandomUnderSampler
from neural_network_classifier import train_model, get_model, noise_reducer
from keras.utils import to_categorical

import tensorflow as tf
import os

use_tpu = True

if use_tpu:
  assert 'COLAB_TPU_ADDR' in os.environ, 'Missing TPU; did yu request a TPU in Notebook Settings?'

if 'COLAB_TPU_ADDR' in os.environ:
  TF_MASTER = 'grpc://{}'.format(os.environ['COLAB_TPU_ADDR'])
else:
  TF_MASTER = ''

tpu_address = TF_MASTER
epochs = 50
steps_per_epoch = 5

resolver = tf.distribute.cluster_resolver.TPUClusterResolver(TF_MASTER)
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
strategy = tf.distribute.experimental.TPUStrategy(resolver)

NEURONS0 = [65]#[100, 128, 150, 200]#[3]#, 5]#, 10, 50, 65, 100, 128, 150]
NEURONS1 = [5000]
ACTIVATION_FUNCTIONS0 = ['relu', 'selu', 'elu', 'tanh', 'sigmoid', 'exponential']
ACTIVATION_FUNCTIONS1 = ['relu', 'selu', 'elu', 'tanh', 'sigmoid', 'exponential']
OPTIMIZERS = ['adam', 'RMSprop', 'sgd']
NOISE = [15000, 20000]#[3000, 5000, 10000]
#activation_function = 'relu'
neurons2 = 3
for neurons0 in NEURONS0:
  for neurons1 in NEURONS1:
    for activation_function0 in ACTIVATION_FUNCTIONS0:
      for activation_function1 in ACTIVATION_FUNCTIONS1:
        for optimizer in OPTIMIZERS:
          for noise in NOISE:
            days = ['ratufa_whole_day', 'dusky']
            Project_path = '/content/drive/My Drive/Sciurid Lab/CNN/VGGish_Squirrels/'
            threshold = 0.5

            # Load training data from pickle files
            path_here = os.path.join(Project_path, 'Data/rnd19_with_labels.pickle')
            with open(path_here, 'rb') as savef:
              audio_feats_data_training, species_training, file_name, num_vecs = np.transpose(np.array(pickle.load(savef)))
            
            SQUIRRELS_LIST = []
            for i in range(audio_feats_data_training.shape[0]):
              toto = np.array(audio_feats_data_training[i], dtype = ('O')).astype(np.float)
              SQUIRRELS_LIST.append(toto)
            
            SQUIRRELS_LIST, species_training = noise_reducer(SQUIRRELS_LIST, species_training, noise)

            SQUIRRELS = pd.DataFrame(SQUIRRELS_LIST)
            SQUIRRELS['species'] = species_training
            species_unique = np.unique(species_training)
            
            SQUIRRELS_dummies = pd.get_dummies(SQUIRRELS, columns = ['species'])
            
            predictors = SQUIRRELS_dummies.drop(columns = ['species_dusky_training_19+SNP',
                                                                        'species_noise', 'species_ratufa'])
            targets = SQUIRRELS_dummies[['species_dusky_training_19+SNP',
                                                                        'species_noise', 'species_ratufa']]
            
            input_shape = predictors.shape[1]
            with strategy.scope():
              model = get_model(neurons0, neurons1, neurons2, activation_function0, activation_function1, input_shape)
              model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])
            print(model.summary())
            model.fit(predictors, targets, epochs = epochs, steps_per_epoch = steps_per_epoch, verbose = 2)
            #model = train_model(neurons0, neurons1, activation_function, predictors, targets, strategy)
            
            # Load sound files for annotations
            for day in days:
              folder_name = Project_path + 'ARU_embeddings/' + day + '/'
              file_names = sorted(os.listdir(folder_name))
              save_folder = Project_path + 'Detections_neural_network_classifier/layer_1/'+str(neurons0)+'_neurons0_'+activation_function+'_'+optimizer+'_' +str(noise)+'_noise_classifier/'
              #save_folder = Project_path+'Detections19/'+str(noise_value)+'_noise_'+str(tree_count)+'trees_'+str(depth)+'depth_classweight_'+classw+'_bootstrap_' + str(bootstrap) + '_classifier/' 
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

                predictions_dummies = model.predict(audio_feats_data, batch_size = None)
                predictions_dummies_df = pd.DataFrame(predictions_dummies, columns = ['species_dusky_training_19+SNP',
                                                                        'species_noise', 'species_ratufa'])
                #predictions_dummies = np.argmax(predictions_dummies, axis = 1)
                #predictions = 
                predictions_dummies_df = predictions_dummies_df.idxmax(axis = 1)
                predictions_dummies_df = predictions_dummies_df.str.lstrip("species_")
                predictions = predictions_dummies_df.to_numpy()
                #print(predictions)
                
                species_prediction.append(predictions)
                species_prediction = np.transpose(np.asarray(species_prediction))
                species_prediction_day.append(np.asarray(species_prediction))
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
              save_path_day = save_folder + day + '.txt'	
              make_day_annotation_file(save_path_day, species_prediction_day, num_preds_file, duration_files)
print(time.time() - start_time)