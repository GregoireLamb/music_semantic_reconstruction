import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.utils.tensorboard import SummaryWriter

from src.config import Config
from src.utils import seed_everything_, generate_visualizations, compute_ged_mer_for_batch, \
    perform_predictions_analysis, save_confusion_matrix
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
            f.write(header + '\n')
    with open('./util_scripts/results/test_results.csv', 'a') as f:
        line = ''
        for key, val in test_result.items():
            line += str(key) + ';' + str(val) + ';'
        f.write(line + '\n')


def test_model(path_to_checkpoint: str, save_results=False):
    checkpoint = torch.load(path_to_checkpoint)
    name = path_to_checkpoint.split("/")[-1].split(".")[0]
    writer = SummaryWriter(log_dir=f"./util_scripts/results/runs/{name}")

    config = Config()
    if 'config' in checkpoint:
        config_param = checkpoint['config']
        config_param = yaml.safe_load(config_param)

        for key, value in config_param.items():
            config[key] = value

    print("Warning: manual setting in test_model")
    config["n_neighbors_KNN"] = 13

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


# def visu_results(file_path='./util_scripts/results/test_results.csv', output_path='./util_scripts/results/test_plot/'):
#     # get the data as a df
#     data = {}
#     with open(file_path, 'r') as f:
#         lines = f.readlines()
#         for line in lines[1:]:
#             l = line.split(';')
#             line_dict = {l[i]: l[i + 1] for i in range(0, len(l), 2) if i + 1 < len(l)}
#             data[line_dict['model']] = line_dict
#
#     df = pd.DataFrame(data).T
#
#     # filter df if edit_distance is above 50
#     df = df[df['edit_distance'].astype(float) < 50]
#
#     # Visualize the results global metrics
#     for metric in ['accuracy', 'edit_distance', 'music_error_rate']:
#         df[metric] = df[metric].astype(float)
#         order = df.groupby('test_type')[metric].max().sort_values(ascending=False).index
#
#         # Create a box plot with the ordered test types
#         plt.figure(figsize=(10, 6))
#         sns.boxplot(x='test_type', y=metric, data=df, order=order)
#         plt.xticks(rotation=45)
#         plt.title(f'{metric} by experiment')
#         plt.xlabel('Experiment')
#         plt.ylabel(metric)
#         plt.tight_layout()
#         plt.savefig(output_path + f'{metric}.png')
#         plt.close()
#
#     # filter df by columns stating with notehead
#     df_notehead = df[df.columns[pd.Series(df.columns).str.startswith('notehead')]]
#     # print the max of each column with its index
#     print(df_notehead)

def mix_multi_predictions(graph, all_predictions, dict_linkType_modelNUmber, label_encoder):
    predictions = all_predictions[0]

    for i in range(len(predictions)):
        primitive1 = graph.edge_index[0][i].item()
        primitive2 = graph.edge_index[1][i].item()

        graph = graph.to('cpu')

        # get primitive class
        id1 = np.where(graph.x[primitive1] == 1)[0][0]
        id2 = np.where(graph.x[primitive2] == 1)[0][0]
        class1 = label_encoder.inverse_transform([id1])[0]
        class2 = label_encoder.inverse_transform([id2])[0]

        key = f"{class1} - {class2}"

        if key not in dict_linkType_modelNUmber:
            continue

        model_to_use = dict_linkType_modelNUmber[key]
        predictions[i] = all_predictions[model_to_use][i]
    return predictions

def test_model_ensemble(dict_linkType_pathToModel: dict):
    dict_linkType_model = {}
    config = Config()
    writer = SummaryWriter(log_dir=f"./util_scripts/results/runs/model_ensemble_{str(datetime.now()).replace(' ','_')}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset_loaded = False
    print(datetime.now())

    for linkType, path_to_checkpoint in dict_linkType_pathToModel.items():
        checkpoint = torch.load(path_to_checkpoint)
        config_param = checkpoint['config']
        config_param = yaml.safe_load(config_param)

        for key, value in config_param.items():
            config[key] = value

        if not dataset_loaded:
            loader = Loader(config=config, device=device)
            dataset = DatasetHandler(config=config)
            loader.load(dataset)
            init_n_label = len(loader.datasetHandler.label_list) + 4  # +4 is for the position

        # set up the model
        model = Model(init_n_label, config).to(device)
        model.load_state_dict(checkpoint['best_model'])
        dict_linkType_model[linkType] = model

    writer.add_text('config', str(config))

    # Predictions

    test_loader = loader.get_dataLoader('test')
    link_analyse_dict = {}
    confusion_mat = [[0, 0], [0, 0]]
    accuracy, edit_distances, music_error_rate = [], [], []
    do_visualize_first_score = config['visualize_first_score']
    label_encoder = dataset.label_encoder

    for graph in (pbar := tqdm(test_loader)):
        pbar.set_description(f"Testing ")

        all_predictions = []
        dict_linkType_modelNUmber = {key: i for i, key in enumerate(dict_linkType_model.keys())}

        for linkType, model in dict_linkType_model.items():
            model.eval()
            predictions = model(x=graph.x, pos=graph.pos, edge_index=graph.edge_index)
            all_predictions.append(predictions)
        predictions = mix_multi_predictions(graph, all_predictions, dict_linkType_modelNUmber, label_encoder)

        predictions = predictions.view(-1)

        truth = np.round(graph.truth.tolist())
        predictions = np.round(predictions.tolist())

        ged, mer = compute_ged_mer_for_batch(graph, predictions, truth, config=config)
        edit_distances.extend(ged)
        music_error_rate.extend(mer)

        link_analyse_dict = perform_predictions_analysis(graph, predictions, truth,
                                                         loader.datasetHandler.label_encoder,
                                                         dict=link_analyse_dict)

        if do_visualize_first_score:
            first_score = graph.to_data_list()[0]
            score_image = loader.datasetHandler.get_score_image(first_score.index)
            n_edges = first_score.edge_index.shape[1]
            predictions_first_score = predictions[0:n_edges]
            truth_first_score = truth[0:n_edges]

            generate_visualizations(first_score, predictions_first_score, writer, score_image, truth_first_score,
                                    loader=loader)
            do_visualize_first_score = False

        conf = confusion_matrix(truth, predictions)

        if len(conf) == 1:  # case there is only positive or negative edges
            print("Unusual conf matrix")
            if len(conf[0]) != 1:
                print("Only positive or negative edges in the confusion matrix ", len(conf))
                if truth[0] == 1:
                    conf = np.array([[0, 0], conf[0]])
                else:
                    conf = np.array([conf[0], [0, 0]])

        confusion_mat += conf
        accuracy.append(accuracy_score(truth, predictions))

    save_confusion_matrix(confusion_mat=confusion_mat, writer=writer)
    writer.add_scalar(tag="Accuracy/test", scalar_value=np.mean(accuracy),
                      global_step=int(datetime.now().timestamp()))
    writer.add_scalar(tag="Mean_GED/test",
                      scalar_value=np.mean(edit_distances),
                      global_step=int(datetime.now().timestamp()))
    writer.add_scalar(tag="Mean_MER/test",
                      scalar_value=np.mean(music_error_rate),
                      global_step=int(datetime.now().timestamp()))

    writer.add_histogram(tag="GED_hist/test",
                         values=np.array(edit_distances))
    writer.add_histogram(tag="MER_hist/test",
                         values=np.array(music_error_rate))

    writer.flush()

    for key, value in link_analyse_dict.items():
        link_analyse_dict[key] = [np.round(value[1] / (value[0] + value[1]), decimals=5), value[0] + value[1]]
    link_analyse_dict = dict(sorted(link_analyse_dict.items(), key=lambda item: sum(item[1]), reverse=True))
    writer.add_text("Link Analysis", str(link_analyse_dict))
    writer.flush()
    writer.close()

    return np.mean(accuracy), np.mean(edit_distances), np.mean(music_error_rate), link_analyse_dict


seed_everything_(42)

dict_linkType_pathToModel_6_labels = {
    "noteheadBlack - stem":"./models/ds_continue_6_labels_123_08-04-2024_13-07.pth",
    "noteheadBlack - accidental":"./models/ds_continue_6_labels_123_08-04-2024_13-07.pth",
    "noteheadBlack - flag":"./models/ds_continue_6_labels_123_08-04-2024_13-07.pth",
    "noteheadBlack - beam":"./models/ds_continue_6_labels_12345_08-02-2024_02-17.pth",
    "noteheadWholeOrHalf - stem":"./models/ds_continue_6_labels_12345_08-02-2024_12-13.pth",
    "noteheadWholeOrHalf - accidental":"./models/ds_continue_6_labels_123_08-04-2024_13-07.pth",
    "noteheadWholeOrHalf - beam":"./models/ds_continue_6_labels_123_08-04-2024_13-07.pth",
    "noteheadWholeOrHalf - flag":"./models/ds_continue_6_labels_123_08-04-2024_13-07.pth",
}

# filter the list
# models = [""]
# test_model(f'./models/{model}', save_results=True)

# mmodel ensemble
test_model_ensemble(dict_linkType_pathToModel_6_labels)