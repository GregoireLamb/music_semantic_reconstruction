import os
import sys
from datetime import datetime

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.utils.tensorboard import SummaryWriter

from src.config import Config
from src.utils import seed_everything_
from src.datasetHandler import DatasetHandler
from src.dataLoader import Loader
from src.model import Model
from src.main import test


def save_test(test_result):
    # try is ./util_scripts/results/test_results.csv exists
    if not os.path.isfile('./util_scripts/results/test_results.csv'):
        # if it does, open it in append mode
        with open('./util_scripts/results/test_results.csv', 'w') as f:
            # write the test_result dictionary to the file
            header = ''
            for key in test_result.keys():
                header += str(key) + ';'
            # header = ';'.join(test_result.keys())
            f.write(header + '\n')
    with open('./util_scripts/results/test_results.csv', 'a') as f:
        line = ''
        for key, val in test_result.items():
            line += str(key) + ';' + str(val) + ';'
        # header = ';'.join(test_result.keys())
        f.write(line + '\n')


def test_model(path_to_checkpoint: str, dataset_name="musigraph", save_results=False):
    checkpoint = torch.load(path_to_checkpoint)
    name = path_to_checkpoint.split("/")[-1].split(".")[0]
    writer = SummaryWriter(log_dir=f"./util_scripts/results/runs/{name}")

    # path_to_config = './processed/' + path_to_checkpoint.split("/")[-1].split(".")[0][:-17] + '.yml'
    config = Config()
    if 'config' in checkpoint:
        config_param = checkpoint['config']
        # check if config_param is a dictionary
        if isinstance(config_param, dict):
            for key, value in config_param.items():
                config[key] = value

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Use GPU if available
    loader = Loader(config=config, device=device)
    dataset = DatasetHandler(config=config)
    loader.load(dataset)

    init_n_label = len(loader.datasetHandler.label_list) + 4  # +4 is for the position
    model = Model(init_n_label, config).to(device)
    model.load_state_dict(checkpoint['best_model'])
    accuracy, edit_distance, music_error_rate, link_analyse_dict = test(model=model, device=device, loader=loader,
                                                                        writer=writer, config=config)

    test_result = link_analyse_dict
    test_result["accuracy"] = accuracy
    test_result["edit_distance"] = edit_distance
    test_result["music_error_rate"] = music_error_rate
    test_result["model"] = path_to_checkpoint.split("/")[-1].split(".")[0]
    model_type = path_to_checkpoint.split("/")[-1].split(".")[0][:-17]
    test_result["test_type"] = model_type

    if save_results:
        save_test(test_result)

    writer.flush()
    writer.close()


def visu_results(file_path='./util_scripts/results/test_results.csv', output_path='./util_scripts/results/test_plot/'):
    # get the data as a df
    data = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            l = line.split(';')
            line_dict = {l[i]: l[i + 1] for i in range(0, len(l), 2) if i + 1 < len(l)}
            data[line_dict['model']] = line_dict

    df = pd.DataFrame(data).T

    # filter df if edit_distance is above 50
    df = df[df['edit_distance'].astype(float) < 50]

    # Visualize the results global metrics
    for metric in ['accuracy', 'edit_distance', 'music_error_rate']:
        df[metric] = df[metric].astype(float)
        order = df.groupby('test_type')[metric].max().sort_values(ascending=False).index


        # Create a box plot with the ordered test types
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='test_type', y=metric, data=df, order=order)
        plt.xticks(rotation=45)
        plt.title(f'{metric} by experiment')
        plt.xlabel('Experiment')
        plt.ylabel(metric)
        plt.tight_layout()
        plt.savefig(output_path + f'{metric}.png')
        plt.close()

    # filter df by columns stating with notehead
    df_notehead = df[df.columns[pd.Series(df.columns).str.startswith('notehead')]]
    # print the max of each column with its index
    print(df_notehead)


seed_everything_(42)
models = [m for m in os.listdir('./models/')]

# filter the list
models = [models[-1]]

for model in models:
    test_model(f'./models/{model}', save_results=True)
# visu_results()
