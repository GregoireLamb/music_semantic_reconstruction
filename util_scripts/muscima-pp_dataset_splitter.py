import os

import random

"""
Script used to generate a train, validation and test split for the Muscima dataset.
"""

file_path = f'./data/muscima-pp/v2.1/specifications/testset-independent.txt'
with open(file_path, 'r') as f:
    lines = f.readlines()
    test_scores_name = [line.rstrip('\n') + '.xml' for line in lines]

all_files_names = os.listdir("./data/muscima-pp/v2.1/data/annotations/")
train_scores_name = [all_files_names[i] for i in range(len(all_files_names)) if
                     all_files_names[i] not in test_scores_name]
validation_scores_names = []

random.seed(123)
for i in range(20):
    rnd = random.randint(0, len(train_scores_name) - 1)
    validation_scores_names.append(train_scores_name.pop(rnd))

# save the files
with open(f'./data/muscima-pp/v2.1/specifications/test.ids', 'w') as f:
    for item in test_scores_name:
        f.write("%s\n" % item)

with open(f'./data/muscima-pp/v2.1/specifications/train.ids', 'w') as f:
    for item in train_scores_name:
        f.write("%s\n" % item)

with open(f'./data/muscima-pp/v2.1/specifications/validation.ids', 'w') as f:
    for item in validation_scores_names:
        f.write("%s\n" % item)
