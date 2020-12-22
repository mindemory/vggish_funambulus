
'''
# See https://github.com/sarabsethi/audioset_soundscape_feats_sethi2019/tree/master/calc_audioset_feats for installation instructions
'''

from AudiosetAnalysisFunambulus import AudiosetAnalysis
import os
import pickle
import wave
import contextlib


# Get all mp3 or wav files in our audio directory
species = input('Species name: ')
Project_path = '/content/drive/My Drive/Sciurid Lab/CNN/VGGish_Squirrels/'
#input('Project path: ')
audio_dir = Project_path + 'squirrel_notes/' + species + '/'
spec_dir = os.path.join(Project_path, 'squirrel_note_embeddings_with_labels', species)
if not os.path.exists(spec_dir):
  os.mkdir(spec_dir)
all_fs = os.listdir(audio_dir)
print('Listing audio files')
audio_fs_temp = [f for f in all_fs if '.wav' in f.lower() or '.mp3' in f.lower()]
print('2 check')

#audio_fs = []
#for temp_file in audio_fs_temp:
#  temp_path = os.path.join(audio_dir, temp_file)
#  with contextlib.closing(wave.open(temp_path,'r')) as fofo:
#    frames = fofo.getnframes()
#    rate = fofo.getframerate()
#  duration = frames / float(rate)
  #print(duration)
#  if duration > 0.98:
#    audio_fs.append(temp_file)
audio_fs = audio_fs_temp
print('3 check')
# Setup the audioset analysis
an = AudiosetAnalysis()
an.setup()
print('4 check')
# Analyse each audio file in turn, and print the shape of the results
for f in audio_fs:
    path = os.path.join(audio_dir, f)
    results = an.analyse_audio(path)
    results['species'] = species
    results['file'] = f
    file_name_f = spec_dir + '/' + f[:-4] + '.pickle'
    with open(file_name_f, 'wb') as opo:
        pickle.dump(results, opo)