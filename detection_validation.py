
import os
import csv
import numpy as np
from sklearn.metrics import confusion_matrix


project_path = '/content/drive/My Drive/Sciurid Lab/CNN/VGGish_Squirrels/'
detections_path = os.path.join(project_path, 'Detections_neural_network_classifier/layer_1')

dusky_time_path = '/content/drive/My Drive/Sciurid Lab/CNN/VGGish_Squirrels/Annotations/dusky_time.txt'
ratufa_time_path = '/content/drive/My Drive/Sciurid Lab/CNN/VGGish_Squirrels/Annotations/ratufa_whole_day_time.txt'
with open(dusky_time_path, 'r') as dtp:
  time_array_dusky = np.asarray([row_dtp[3] for row_dtp in csv.reader(dtp, delimiter = '\t')])[1:]
time_array_dusky = np.asarray([float(a) for a in time_array_dusky])
with open(ratufa_time_path, 'r') as rtp:
  time_array_ratufa = np.asarray([row_rtp[3] for row_rtp in csv.reader(rtp, delimiter = '\t')])[1:]
time_array_ratufa = np.asarray([float(b) for b in time_array_ratufa])


val_file_name = project_path + 'detector_validation/detector_validation_neural_network_layer_1.txt'
text_file = open(val_file_name, 'w')
text_file.write("Model\t Folder\t ratufa Total detections\t ratufa Total annotations\t ratufa TP\t ratufa FP\t ratufa FN\t ratufa TN\t ratufa Precison\t ratufa Recall\t ratufa F1\t dusky Total detections\t dusky Total annotations\t dusky TP\t dusky FP\t dusky FN\t dusky TN\t dusky Precison\t dusky Recall\t dusky F1\n")
del_t = 0.1
CLASSIFIERS = os.listdir(detections_path)
for classifier in CLASSIFIERS:
  print('We are in classifier = {}'.format(classifier))
  days = ['ratufa_whole_day', 'dusky']
  spec_dict = {'dusky':'dusky', ' dusky':'dusky', ' ratufa':'ratufa', 'ratufa':'ratufa', 'dusky_training_19+SNP':'dusky', ' dusky_training_19+SNP':'dusky'}
  tp = {}
  tn = {}
  fp = {}
  fn = {}
  precision = {}
  recall = {}
  F1 = {}

  OMG_bored = ['ratufa', 'dusky']
  species_column_dict = {}
  for i in range(len(OMG_bored)):
    species_column_dict[OMG_bored[i]] = i + 1

  for day in days:
    for OMG in OMG_bored:
      tp[OMG] = 0
      tn[OMG] = 0
      fp[OMG] = 0
      fn[OMG] = 0
      precision[OMG] = 0
      recall[OMG] = 0
      F1[OMG] = 0

    #*******************************************************************************#
    detection_folder = os.path.join(project_path, 'Detections_neural_network_classifier/layer_1', classifier)
    annotation_folder = os.path.join(project_path, 'Annotations')
    detection_file_path = os.path.join(detection_folder, day + '.txt')
    
    with open(detection_file_path, 'r') as dn:
      begin_time_dn = np.asarray([row_dn0[3] for row_dn0 in csv.reader(dn, delimiter = '\t')])[1:]
    begin_time_dn = np.asarray([float(tm0) for tm0 in begin_time_dn])
    with open(detection_file_path, 'r') as dn:
      end_time_dn = np.asarray([row_dn1[4] for row_dn1 in csv.reader(dn, delimiter = '\t')])[1:]
    end_time_dn = np.asarray([float(tm1) for tm1 in end_time_dn])
    with open(detection_file_path, 'r') as dn:
      species_dn = np.asarray([row_dn2[7] for row_dn2 in csv.reader(dn, delimiter = '\t')])[1:]
    species_dn = np.asarray([spec_dict[sp] for sp in species_dn])
    if day == 'dusky':
      annotation_file = 'dusky.txt'
      time_array = time_array_dusky
    else:
      annotation_file = 'ratufa.txt'
      time_array = time_array_ratufa

    detections = np.zeros((len(time_array), len(OMG_bored) + 1))
    annotations = np.zeros((len(time_array), len(OMG_bored)+ 1))
    for i in range(len(time_array)):
      detections[i, 0] = time_array[i]
      annotations[i, 0] = time_array[i]
    for i in range(begin_time_dn.shape[0]):
      btdn = round(begin_time_dn[i], 2)
      detection_row = np.where(abs(detections[:, 0] - btdn) <= 0.01)[0][0]
      detections[detection_row, species_column_dict[species_dn[i]]] = 1

    annotation_file_path = os.path.join(annotation_folder, annotation_file)
    with open(annotation_file_path, 'r') as an:
      begin_time_an = np.asarray([row_an0[3] for row_an0 in csv.reader(an, delimiter = '\t')])[1:]
    begin_time_an = np.asarray([float(tm0p) for tm0p in begin_time_an])
    with open(annotation_file_path, 'r') as an:
      end_time_an = np.asarray([row_an1[4] for row_an1 in csv.reader(an, delimiter = '\t')])[1:]
    end_time_an = np.asarray([float(tm1p) for tm1p in end_time_an])
    if day == 'dusky':
      species_an = np.array([day]*np.shape(begin_time_an)[0])
    else:
      species_an = np.array(['ratufa']*np.shape(begin_time_an)[0])
    
    for t in time_array:
      for j in range(begin_time_an.shape[0]):
        btan = begin_time_an[j]
        etan = end_time_an[j]
        if ((t+del_t < btan < t+0.96-del_t) or (t+del_t < etan < t+0.96-del_t) or (btan <= t and etan >= t+0.96)):
          annotation_row = np.where(annotations[:, 0] == t)[0][0]
          annotations[annotation_row, species_column_dict[species_an[j]]] = 1
      
    for spp in OMG_bored:
      y_true = annotations[:, species_column_dict[spp]]
      y_pred = detections[:, species_column_dict[spp]]
      
      TN, FP, FN, TP = confusion_matrix(y_true, y_pred, labels = [0, 1]).ravel()
      tn[spp] += TN
      fp[spp] += FP
      fn[spp] += FN
      tp[spp] += TP

    for omg in OMG_bored:
      prec_den = tp[omg] + fp[omg]
      rec_den = tp[omg] + fn[omg]
      if prec_den == 0:
        precision[omg] = 100
      else:
        precision[omg] = round(tp[omg]/(tp[omg] + fp[omg]), 4)
      if rec_den == 0:
        recall[omg] = 100
      else:
        recall[omg] = round(tp[omg]/(tp[omg] + fn[omg]), 4)
      if precision[omg] + recall[omg] > 0:
        F1[omg] = round(2*precision[omg]*recall[omg] / (precision[omg] + recall[omg]), 4)
      else:
        F1[omg] = 100
    print(tp, fp, fn, tn)
  
    text_file.write(classifier+'\t'+day+'\t'+ str(tp[OMG_bored[0]] + fp[OMG_bored[0]])+'\t'+str(tp[OMG_bored[0]] +fn[OMG_bored[0]])+'\t'+str(tp[OMG_bored[0]])+'\t'+str(fp[OMG_bored[0]])+'\t'+str(fn[OMG_bored[0]])+'\t'+str(tn[OMG_bored[0]])+'\t'+str(precision[OMG_bored[0]])+'\t'+str(recall[OMG_bored[0]])+'\t'+str(F1[OMG_bored[0]])+'\t'+ str(tp[OMG_bored[1]] + fp[OMG_bored[1]])+'\t'+str(tp[OMG_bored[1]] +fn[OMG_bored[1]])+'\t'+str(tp[OMG_bored[1]])+'\t'+str(fp[OMG_bored[1]])+'\t'+str(fn[OMG_bored[1]])+'\t'+str(tn[OMG_bored[1]])+'\t'+str(precision[OMG_bored[1]])+'\t'+str(recall[OMG_bored[1]])+'\t'+str(F1[OMG_bored[1]])+'\n')
text_file.close()