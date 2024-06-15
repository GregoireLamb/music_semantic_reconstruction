import os
import random
import numpy as np

xml_dir=r'./data/DoReMi_v1/Parsed_by_page_omr_xml/'
output_path = './data/DoReMi_v1/'
train_ratio=0.7
val_ratio=0.15

list_score = os.listdir(xml_dir)
random.seed(123)
np.random.seed(123)

np.random.shuffle(list_score)

total_files = len(list_score)
train_size = int(total_files * train_ratio)
val_size = int(total_files * val_ratio)
test_size = total_files - train_size - val_size

train_files = list_score[:train_size]
val_files = list_score[train_size:train_size + val_size]
test_files = list_score[train_size + val_size:]

with open(output_path + 'train.ids', 'w') as train_file:
    for filename in train_files:
        train_file.write(f"{filename}\n")

with open(output_path + 'validation.ids', 'w') as val_file:
    for filename in val_files:
        val_file.write(f"{filename}\n")

with open(output_path + 'test.ids', 'w') as test_file:
    for filename in test_files:
        test_file.write(f"{filename}\n")