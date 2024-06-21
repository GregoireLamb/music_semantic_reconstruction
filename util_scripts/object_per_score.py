import os
import sys

# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tqdm import tqdm
import torch

import pandas as pd
from src.datasetHandler import DatasetHandler
from src.config import Config
from src.dataLoader import Loader

output_path = './util_scripts/results/'
dataset_names = ['musigraph']
label_to_use = '10_labels'
datasetHandler_list = []
config = Config()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config['labels_to_use'] = label_to_use
for dataset_name in dataset_names:
    config['dataset'] = dataset_name

    datasetHandler = DatasetHandler(config)
    loader = Loader(config, device)
    loader.load(datasetHandler)
    loader.set_split([], [i for i in range(len(datasetHandler))], [])
    dataLoader = loader.get_dataLoader(split='test')

    node_count_dict = {}
    link_count_dict = {}

    for graph in (pbar := tqdm(dataLoader)):
        pbar.set_description(f"Counting score size {dataset_name}")
        for g in graph.to_data_list():
            node_count = g.x.size()[0]
            link_count = len(g.original_edges_in)
            if node_count in node_count_dict:
                node_count_dict[node_count] += 1
            else:
                node_count_dict[node_count] = 1
            if link_count in link_count_dict:
                link_count_dict[link_count] += 1
            else:
                link_count_dict[link_count] = 1

link_count_df = pd.DataFrame(link_count_dict.items(), columns=['size', 'count'])
link_count_df = link_count_df.sort_values(by=['size'])
link_count_df.to_csv(output_path + f'{dataset_name}_link_size_count.csv', index=False)

node_count_df = pd.DataFrame(node_count_dict.items(), columns=['size', 'count'])
node_count_df = node_count_df.sort_values(by=['size'])
node_count_df.to_csv(output_path + f'{dataset_name}_score_size_count.csv', index=False)
