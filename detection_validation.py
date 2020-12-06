
import os
import csv
import numpy as np
from sklearn.metrics import confusion_matrix

del_t = 0.1
NOISE = [3000, 5000, 10000]
TREES = [100, 300, 500, 1000]   
DEPTH = [50, 80, 100, 120]
for depth in DEPTH: 
  for tree_count in TREES:      
    for noise_value in NOISE:
      days = ['ratufa_whole_day', 'dusky']
      project_path = '/content/drive/My Drive/Sciurid Lab/CNN/VGGish_Squirrels/'#/Detecions/'
      classifier_type = str(noise_value)+'noise_'+str(tree_count)+'trees'+str(depth)+'_classifier'
      spec_dict = {'dusky':'dusky', ' dusky':'dusky', ' ratufa':'ratufa', 'ratufa':'ratufa'}
      tp = {}
      tn = {}
      fp = {}
      fn = {}
      precision = {}
      recall = {}
      F1 = {}

      OMG_bored = ['dusky', 'ratufa']
      #time_array_file = [0,0.96,1.92,2.88,3.84,4.8,5.76,6.72,7.68,8.64,9.6,10.56,11.52,12.48,13.44,14.4,15.36,16.32,17.28,18.24,19.2,20.16,21.12,22.08,23.04,24,24.96,25.92,26.88,27.84,28.8,29.76,30.72,31.68,32.64,33.6,34.56,35.52,36.48,37.44,38.4,39.36,40.32,41.28,42.24,43.2,44.16,45.12,46.08,47.04,48,48.96,49.92,50.88,51.84,52.8,53.76,54.72,55.68,56.64,57.6,58.56,59.52,60.48,61.44,62.4,63.36,64.32,65.28,66.24,67.2,68.16,69.12,70.08,71.04,72,72.96,73.92,74.88,75.84,76.8,77.76,78.72,79.68,80.64,81.6,82.56,83.52,84.48,85.44,86.4,87.36,88.32,89.28,90.24,91.2,92.16,93.12,94.08,95.04,96,96.96,97.92,98.88,99.84,100.8,101.76,102.72,103.68,104.64,105.6,106.56,107.52,108.48,109.44,110.4,111.36,112.32,113.28,114.24,115.2,116.16,117.12,118.08,119.04,120,120.96,121.92,122.88,123.84,124.8,125.76,126.72,127.68,128.64,129.6,130.56,131.52,132.48,133.44,134.4,135.36,136.32,137.28,138.24,139.2,140.16,141.12,142.08,143.04,144,144.96,145.92,146.88,147.84,148.8,149.76,150.72,151.68,152.64,153.6,154.56,155.52,156.48,157.44,158.4,159.36,160.32,161.28,162.24,163.2,164.16,165.12,166.08,167.04,168,168.96,169.92,170.88,171.84,172.8,173.76,174.72,175.68,176.64,177.6,178.56,179.52,180.48,181.44,182.4,183.36,184.32,185.28,186.24,187.2,188.16,189.12,190.08,191.04,192,192.96,193.92,194.88,195.84,196.8,197.76,198.72,199.68,200.64,201.6,202.56,203.52,204.48,205.44,206.4,207.36,208.32,209.28,210.24,211.2,212.16,213.12,214.08,215.04,216,216.96,217.92,218.88,219.84,220.8,221.76,222.72,223.68,224.64,225.6,226.56,227.52,228.48,229.44,230.4,231.36,232.32,233.28,234.24,235.2,236.16,237.12,238.08,239.04,240,240.96,241.92,242.88,243.84,244.8,245.76,246.72,247.68,248.64,249.6,250.56,251.52,252.48,253.44,254.4,255.36,256.32,257.28,258.24,259.2,260.16,261.12,262.08,263.04,264,264.96,265.92,266.88,267.84,268.8,269.76,270.72,271.68,272.64,273.6,274.56,275.52,276.48,277.44,278.4,279.36,280.32,281.28,282.24,283.2,284.16,285.12,286.08,287.04,288,288.96,289.92,290.88,291.84,292.8,293.76,294.72,295.68,296.64,297.6,298.56]
      #time_array = [0, 0.096, 0.192, 0.288, 0.384, 0.48, 0.576, 0.672, 0.768, 0.864, 0.96, 1.056, 1.152, 1.248, 1.344, 1.44, 1.536, 1.632, 1.728, 1.824]
      species_column_dict = {}
      for i in range(len(OMG_bored)):
        species_column_dict[OMG_bored[i]] = i + 1
      
      for day in days:
        print(day)
        for OMG in OMG_bored:
          tp[OMG] = 0
          tn[OMG] = 0
          fp[OMG] = 0
          fn[OMG] = 0
          precision[OMG] = 0
          recall[OMG] = 0
          F1[OMG] = 0

        #*******************************************************************************#
        detection_folder = os.path.join(project_path, 'Detections', classifier_type)
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
        #time_array = []
        if day == 'dusky':
          annotation_file = 'dusky_training_19+SNR.txt'
          times = np.arange(0, 10080, 0.96)
          time_array = [round(a, 2) for a in times]
        else:
          annotation_file = 'yp-sm4-05_10_11_For mrugank.txt'
          times = np.arange(0, 37379, 0.96)
          time_array = [round(a, 2) for a in times]

        detections = np.zeros((len(time_array), len(OMG_bored) + 1))
        annotations = np.zeros((len(time_array), len(OMG_bored)+ 1))
        for i in range(len(time_array)):
          detections[i, 0] = time_array[i]
          annotations[i, 0] = time_array[i]
        for i in range(begin_time_dn.shape[0]):
          btdn = begin_time_dn[i]
          detection_row = np.where(detections[:, 0] == btdn)[0]
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
              annotation_row = np.where(annotations[:, 0] == t)[0]#[0]
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

        #val_file_name = project_path + 'detector_validation/' + classifier_type + '.txt'
        #text_file = open(val_file_name, 'w')
        #text_file.write("Species\t Total detections\t Total annotations\t TP\t FP\t FN\t TN\t Precison\t Recall\t F1\n")
        #for popo in OMG_bored:
        #  text_file.write(popo+'\t'+str(tp[popo] + fp[popo])+'\t'+str(tp[popo] +fn[popo])+'\t'+str(tp[popo])+'\t'+str(fp[popo])+'\t'+str(fn[popo])+'\t'+str(tn[popo])+'\t'+str(precision[popo])+'\t'+str(recall[popo])+'\t'+str(F1[popo])+'\n')
        #text_file.close()