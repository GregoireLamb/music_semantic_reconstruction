import os
import sys

# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

from src.datasetHandler import DatasetHandler
from src.config import Config
from src.dataLoader import Loader

"""
Script used to explore the influence of k in the KNN graph.
"""

output_path = '../util_scripts/results/'
dataset_names = ['muscima_measure_cut']
label_to_use = 'full_muscima_labels'
datasetHandler_list = []
config = Config()
k_values = [13, 14, 15, 20]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config['labels_to_use'] = label_to_use
config['undirected_edges'] = False
config['prefilter_KNN'] = False
for dataset_name in dataset_names:
    config['dataset'] = dataset_name
    for n_value in k_values:
        config['n_neighbors_KNN'] = n_value
        datasetHandler = DatasetHandler(config)
        loader = Loader(config, device)
        loader.load(datasetHandler)
        loader.set_split([], [i for i in range(len(datasetHandler))], [])
        dataLoader = loader.get_dataLoader(split='test')

        count_edges_in_knn = 0
        count_covered_edges = 0
        count_objects = 0
        count_node_ged_dict = {}

        for graph in (pbar := tqdm(dataLoader)):
            pbar.set_description(f"Testing {n_value} for {dataset_name}")
            graph = graph.to(device)
            truth = np.round(graph.truth.tolist())
            predictions = truth
            count_covered_edges += sum(truth)
            count_edges_in_knn += len(truth)
            count_objects += int(graph.x.size()[0])

            a, b = 0, 0
            for i, g in enumerate(graph.to_data_list()):
                b += g.edge_index.size()[1]
                if g.x.size()[0] not in count_node_ged_dict:
                    count_node_ged_dict[g.x.size()[0]] = []
                count_node_ged_dict[g.x.size()[0]].append(np.sum(truth[a:b])/g.original_edges_in.size()[0] if np.sum(truth[a:b]) > 0 else 1)
                a = b

        count_node_ged_dict = {k: np.mean(v) for k, v in count_node_ged_dict.items()}
        count_node_ged_df = pd.DataFrame(list(count_node_ged_dict.items()), columns=['n', 'avg_GED'])
        count_node_ged_df = count_node_ged_df.sort_values(by='n', ignore_index=True)

        # create file n_neigbhors_exploration if not exists and append edit dist
        if os.path.isfile(output_path + 'n_neigbhors_exploration.csv'):
            with open(output_path + 'n_neigbhors_exploration.csv', "a") as file:
                file.write(f'\n{dataset_name},{n_value},'
                           f'{count_covered_edges},{count_edges_in_knn},{count_objects},{label_to_use})')
                for i, row in count_node_ged_df.iterrows():
                    print(str(row))
                    file.write(str(row))
                    break
        else:
            with open(output_path + 'n_neigbhors_exploration.csv', "w") as file:
                file.write(
                    f'dataset_name,n_value,count_covered_edges,count_edges_in_knn,count_objects,granularity, count_ged_per_node\n')
                file.write(
                    f'{dataset_name},{n_value},{count_covered_edges},{count_edges_in_knn},{count_objects},{label_to_use}, {count_node_ged_df}')
