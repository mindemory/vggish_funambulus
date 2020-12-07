import numpy as np

def make_annotation_file(save_path, species):
  low_freq_dict = {'dusky': 5000, 'ratufa': 800}
  high_freq_dict = {'dusky': 11000, 'ratufa': 4000}
  max_annotation_dict = {'dusky': 0, 'ratufa': 0}
  row_count = species.shape[0]
  text_file = open(save_path, 'w+')
  text_file.write("Selection\t View\t Channel\t Begin Time (S)\t End Time (S)\t Low Freq (Hz)\t High Freq (Hz)\t Species\n")
  annotation_count = 0
  row = 0
  while row < row_count:
    species_name = species[row][0]
    skip = True
    skip_annotation_count = 1
    while skip == True:
      if row + skip_annotation_count < row_count:
        if species[row] ==  species[row + skip_annotation_count] and skip_annotation_count <= max_annotation_count:
            skip_annotation_count += 1
        else:
          skip = False
      else:
        skip = False
      
    if species_name != 'noise':
      annotation_count += 1
      begin_time = round(0.960 * row, 5)
      end_time = round(begin_time + skip_annotation_count * 0.960, 5)
      low_freq = low_freq_dict[species_name]
      high_freq = high_freq_dict[species_name]
      row_input = "{}\t Spectrogram 1\t 1\t {}\t {}\t {}\t {}\t {}\n".format(annotation_count, begin_time, end_time, low_freq, high_freq, species_name)
      text_file.write(row_input)
    row += skip_annotation_count
  return text_file

def make_day_annotation_file(save_path, species, num_preds_file, duration_files):
  low_freq_dict = {'dusky': 5000, 'ratufa': 800}
  high_freq_dict = {'dusky': 11000, 'ratufa': 4000}
  max_annotation_dict = {'dusky': 0, 'ratufa': 0}
  text_file = open(save_path, 'w+')
  text_file.write("Selection\t View\t Channel\t Begin Time (S)\t End Time (S)\t Low Freq (Hz)\t High Freq (Hz)\t Species\n")
  annotation_count = 0
  for file_count in range(len(duration_files) - 1):
    file_length = duration_files[file_count]
    species_in_file = species[file_count].flatten()
    column_count = species_in_file.shape[0]
    column = 0
    while column < column_count:
      species_name = species_in_file[column]
      skip = True
      skip_annotation_count = 1
      while skip == True:
        if column + skip_annotation_count < column_count and skip_annotation_count <= max_annotation_count:
          if species_in_file[column] ==  species_in_file[column + skip_annotation_count]:
              skip_annotation_count += 1
          else:
            skip = False
        else:
          skip = False
      
      if species_name != 'noise':
        annotation_count += 1
        begin_time = round(0.960 * column + file_length, 5)
        end_time = round(begin_time + skip_annotation_count * 0.960, 5)
        low_freq = low_freq_dict[species_name]
        high_freq = high_freq_dict[species_name]
        row_input = "{}\t Spectrogram 1\t 1\t {}\t {}\t {}\t {}\t {}\n".format(annotation_count, begin_time, end_time, low_freq, high_freq, species_name)
        text_file.write(row_input)

      column += skip_annotation_count
  return text_file