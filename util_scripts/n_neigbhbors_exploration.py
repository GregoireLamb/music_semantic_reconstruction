import os
import sys

# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from tqdm import tqdm
import torch

from src.utils import compute_ged_mer_for_batch
from src.datasetHandler import DatasetHandler
from src.config import Config
from src.dataLoader import Loader

####
#
#   Estimate link coverage of KNN graphs based on the number of neighbors used during the KNN graph generation
#
###

output_path = './util_scripts/results/'
dataset_names = ['musigraph'] #, 'musigraph']  # ["musigraph", "muscima-pp"]
label_to_use = '8_labels'
datasetHandler_list = []
config = Config()
n_values = [13]  # msucima-pp: 14, 18, 22, // musigraph 14

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config['labels_to_use'] = label_to_use
for dataset_name in dataset_names:
    config['dataset'] = dataset_name
    for n_value in n_values:
        config['n_neighbors_KNN'] = n_value
        datasetHandler = DatasetHandler(config)
        loader = Loader(config, device)
        loader.load(datasetHandler)
        loader.set_split([], [i for i in range(len(datasetHandler))], [])
        dataLoader = loader.get_dataLoader(split='test')

        edit_distances = []
        music_error_rate = []
        count_edges_in_knn = 0
        count_covered_edges = 0
        count_objects = 0
        # link_type_count = {}
        # label_encoder = datasetHandler.label_encoder

        for graph in (pbar := tqdm(dataLoader)):
            pbar.set_description(f"Testing {n_value} for {dataset_name}")
            graph = graph.to(device)
            truth = np.round(graph.truth.tolist())
            predictions = truth
            count_covered_edges += sum(truth)
            count_edges_in_knn += len(truth)
            count_objects += int(graph.x.size()[0])

            ged, mer = compute_ged_mer_for_batch(graph, predictions)
            edit_distances.extend(ged)
            music_error_rate.extend(mer)

            # for i in range(len(truth)):
            #     if truth[i] == 1:
            #         id = graph.edge_index[0][i].item()
            #         index = graph.x[id][:-4].argmax()
            #         label= label_encoder.inverse_transform([index.item()])[0]
            #         key = (f'{label} - '
            #                f'{label_encoder.inverse_transform([graph.x[graph.edge_index[1][i].item()][:-4].argmax().item()])[0]}')
            #         if key not in link_type_count:
            #             link_type_count[key] = 0
            #         link_type_count[key] += 1




        print("Average graph edit distance: ", np.mean(edit_distances))
        print("Average graph music error rate: ", np.mean(music_error_rate))

        # create file n_neigbhors_exploration if not exists and append edit dist
        if os.path.isfile(output_path + 'n_neigbhors_exploration.csv'):
            with open(output_path + 'n_neigbhors_exploration.csv', "a") as file:
                file.write(f'\n{dataset_name},{n_value},{np.mean(edit_distances)},{np.mean(music_error_rate)},'
                           f'{count_covered_edges},{count_edges_in_knn},{count_objects},{label_to_use}')
        else:
            with open(output_path + 'n_neigbhors_exploration.csv', "w") as file:
                file.write(f'dataset_name,n_value,graph_edit_distances,music_error_rate,count_covered_edges,count_edges_in_knn,count_objects,granularity\n')
                file.write(f'{dataset_name},{n_value},{np.mean(edit_distances)},{np.mean(music_error_rate)},{count_covered_edges},{count_edges_in_knn},{count_objects},{label_to_use}')
