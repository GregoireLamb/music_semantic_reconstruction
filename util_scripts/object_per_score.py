import os
import sys

from matplotlib import pyplot as plt

# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tqdm import tqdm
import torch
import seaborn as sns
import pandas as pd

from src.datasetHandler import DatasetHandler
from src.config import Config
from src.dataLoader import Loader

output_path = './util_scripts/results/'
dataset_names = ['musigraph']
label_to_use = 'full_musigraph'
datasetHandler_list = []
config = Config()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config['labels_to_use'] = label_to_use
config['undirected_edges'] = False
config['prefilter_KNN'] = False
for dataset_name in dataset_names:
    config['dataset'] = dataset_name

    datasetHandler = DatasetHandler(config)
    loader = Loader(config, device)
    loader.load(datasetHandler)
    loader.set_split([], [i for i in range(len(datasetHandler))], [])
    dataLoader = loader.get_dataLoader(split='test')

    node_link_count_dict = {}
    max_node, max_link = 0, 0

    for graph in (pbar := tqdm(dataLoader)):
        pbar.set_description(f"Counting score size {dataset_name}")
        for g in graph.to_data_list():
            node = g.x.size()[0]
            link = len(g.original_edges_in)

            max_node = max(max_node, node)
            max_link = max(max_link, link)

            key = (node, link)
            if key not in node_link_count_dict:
                node_link_count_dict[key] = 0
            node_link_count_dict[key] += 1

    data_list = [(k[0], k[1], v) for k, v in node_link_count_dict.items()]
    df = pd.DataFrame(data_list, columns=['number of nodes', 'number of links', 'counts'])

    # Expand the DataFrame based on counts
    expanded_df = df.loc[df.index.repeat(df['counts'])].reset_index(drop=True)

    sns.set_theme(style="ticks")
    plot = sns.jointplot(data=expanded_df, x='number of nodes', y='number of links', kind='hist',
                         color="#4CB391", marginal_ticks=True)

    # save
    node_link_count_df = pd.DataFrame(node_link_count_dict.items(), columns=['(node, link)', 'count'])
    node_link_count_df.to_csv(output_path + f'{dataset_name}_node_link_size_count.csv', index=False)

    # save sns plot
    plt.savefig(output_path + f'{dataset_name}_node_link_plot.png')
