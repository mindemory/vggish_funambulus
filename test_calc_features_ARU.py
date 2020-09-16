'''
# See https://github.com/sarabsethi/audioset_soundscape_feats_sethi2019/tree/master/calc_audioset_feats for installation instructions
'''

from AudiosetAnalysisFunambulus_ARU import AudiosetAnalysis
import os
import pickle

# Get all mp3 or wav files in our audio directory
#species = input('Species name: ')
Project_path = '/content/drive/My Drive/Sciurid Lab/CNN/VGGish_Squirrels/' #input('Project path: ')
Folder_name = input('dusky or ratufa or noise: ')
audio_dir = Project_path + 'ARU_Test/' + Folder_name +'/'
spec_dir = os.path.join(Project_path, 'ARU_embeddings', Folder_name)
if not os.path.exists(spec_dir):
  os.mkdir(spec_dir)
all_fs = os.listdir(audio_dir)
audio_fs = [f for f in all_fs if '.wav' in f.lower() or '.mp3' in f.lower()]

# Setup the audioset analysis
an = AudiosetAnalysis()
an.setup()

# Analyse each audio file in turn, and print the shape of the results
for f in audio_fs:
    path = os.path.join(audio_dir, f)
    results = an.analyse_audio(path)
    results['file_name'] = f
    file_name_f = spec_dir + '/' + f[:-4] + '.pickle'
    with open(file_name_f, 'wb') as opo:
        pickle.dump(results, opo)