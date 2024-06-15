import os

DATASET = "doremi"

if DATASET == "muscima-pp":
    full_page_folder_path = './data/muscima-pp/v2.1/specifications/'
    save_folder_path = './data/muscima-pp/measure_cut/specifications/'
    measure_cut_file = os.listdir("./data/muscima-pp/measure_cut/data/annotations/")

elif DATASET == "doremi":
    full_page_folder_path = './data/DoReMi_v1/'
    save_folder_path = './data/DoReMi_v1/measure_cut/'
    measure_cut_file = os.listdir("./data/DoReMi_v1/measure_cut/Parsed_by_measure_omr_xml/")

fataset_full_page_train_list = []
fataset_full_page_validation_list = []
fataset_full_page_test_list = []

for file_name in ["train.ids", "validation.ids", "test.ids"]:
    with open(full_page_folder_path + file_name, 'r') as f:
        lines = f.readlines()
        if file_name == "train.ids":
            fataset_full_page_train_list = [line[:-5] for line in lines]
        elif file_name == "validation.ids":
            fataset_full_page_validation_list = [line[:-5] for line in lines]
        elif file_name == "test.ids":
            fataset_full_page_test_list = [line[:-5] for line in lines]

measure_cut_train_file = []
measure_cut_validation_file = []
measure_cut_test_file = []

for file in measure_cut_file:
    file_pref = file.split('_measure_')[0]
    if file_pref in fataset_full_page_train_list:
        measure_cut_train_file.append(file)
    elif file_pref in fataset_full_page_validation_list:
        measure_cut_validation_file.append(file)
    elif file_pref in fataset_full_page_test_list:
        measure_cut_test_file.append(file)

# save the files
with open(f'{save_folder_path}/test.ids', 'w') as f:
    for item in measure_cut_test_file:
        f.write("%s\n" % item)

with open(f'{save_folder_path}/train.ids', 'w') as f:
    for item in measure_cut_train_file:
        f.write("%s\n" % item)

with open(f'{save_folder_path}/validation.ids', 'w') as f:
    for item in measure_cut_validation_file:
        f.write("%s\n" % item)
