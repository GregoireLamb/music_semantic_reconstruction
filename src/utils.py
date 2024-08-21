import copy
import os
from io import BytesIO

import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sn
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torch_geometric import seed_everything
from torch_geometric.data import DataLoader
from torch_geometric.utils import to_networkx
from torchvision import transforms

from src.config import Config
from src.model import Model


def seed_everything_(seed: int) -> None:
    """
    Set the seed for the different libraries (do not assert reproducibility)
    """
    # random.seed(seed)
    seed_everything(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_model(model_path: str, model: Model, best_model: Model, optimizer, config: Config):
    """
    Load the model from the checkpoint to either continue the training or restart from the best model.
    :return: model, best_model, optimizer, starting_epoch, best_validation_loss
    """
    print("Loading ", model_path)
    checkpoint = torch.load(model_path)
    best_epoch = checkpoint['best_epoch']
    best_validation_loss = checkpoint['best_validation_loss']
    model.load_state_dict(checkpoint['model'])
    best_model.load_state_dict(checkpoint['best_model'])
    starting_epoch = checkpoint['final_epoch']
    config['starting_epoch'] = starting_epoch
    optimizer.load_state_dict(checkpoint['optimizer'])
    if config['start_from_best_model']:
        model.load_state_dict(checkpoint['best_model'])
        starting_epoch = best_epoch + 1
        optimizer.load_state_dict(checkpoint['best_optimizer'])

    if 'fine_tunning' in config:
        config['fine_tunning'] = False
        best_validation_loss = np.inf

    return model, best_model, optimizer, starting_epoch, best_validation_loss


def compute_weight_imbalance(data_loader: DataLoader) -> float:
    """
    Compute the weight imbalance based on data_loader set
    """
    all_t = torch.cat([graph.truth for graph in data_loader.dataset])
    return len(all_t) / max(1, torch.sum(all_t).item())


def edit_distance(original_set, predicted_set, config):
    """
    Compute the edit distance between two set of edges, original_set is considered as the ground truth
    """
    set1 = {(original_set[0][i].item(), original_set[1][i].item())
            for i in range(len(original_set[0]))}
    set2 = {(predicted_set[0][i].item(), predicted_set[1][i].item())
            for i in range(len(predicted_set[0]))}

    if config['undirected_edges']:
        set2_tmp = set2.copy()
        for i in set2:
            if not (i[1], i[0]) in set2_tmp:
                set2_tmp.add(i)
        set2 = set2_tmp
        set_3 = set2 - set1.intersection(set2)
        for i in set_3:
            set2.remove(i)
            set2.add((i[1], i[0]))

    dist = len(set1.union(set2)) - len(set1.intersection(set2))
    mer = -1
    if len(set1) > 0:
        mer = dist / len(set1)
    return dist, mer


def compute_ged_mer_for_graph(graph, predictions, config):
    """
    Genrate the list of predicted edges and recover the list of original edges
    and compute the GED and MER
    """
    predicted_graph = copy.deepcopy(graph)
    predicted_graph.edge_index = graph.edge_index[:, predictions >= 1]

    original_edge_set = [predicted_graph.original_edges_in, predicted_graph.original_edges_out]

    return edit_distance(original_edge_set, predicted_graph.edge_index, config)


def compute_ged_mer_for_batch(graph, predictions, config):
    """
    Compute the GED and MER for a batch of graphs
    """
    i, j = 0, 0
    dist, mer = [], []
    for g in graph.to_data_list():
        n_edges = g.edge_index.shape[1]
        i = j
        j = i + n_edges
        d, m = compute_ged_mer_for_graph(g, predictions[i:j], config)
        dist.append(d)

        if m != -1:  # if the graph has no edges but there is a false positive
            mer.append(m)

    return dist, mer


def perform_predictions_analysis(graph, predictions, truth, label_encoder, dict={}):
    """
    Analyze the prediction of the model
    :return: a dictionary with the accuracy for each link type
    """
    graph = graph.to('cpu')
    primitive_classes = [label_encoder.inverse_transform([np.where(graph.x[i] == 1)[0][0]])[0] for i in
                         range(graph.x.shape[0])]

    for i in range(len(predictions)):
        primitive1 = graph.edge_index[0][i].item()
        primitive2 = graph.edge_index[1][i].item()

        class1 = primitive_classes[primitive1]
        if class1.startswith("notehea"):
            class2 = primitive_classes[primitive2]

            key = f"{class1} - {class2}"

            if key not in dict:
                dict[key] = [0, 0]

            if truth[i] == predictions[i]:
                dict[key][1] += 1
            else:
                dict[key][0] += 1
        else:
            pass
            # print("Warning: Error in the prediction analysis, ignore warning if using undirected edges")
    return dict


def generate_visualisations(graph, predictions, writer: SummaryWriter, score_img, truth):
    """
    Generate 3 image of the scores "True positive", "False Positive", "False Negative"
    the corresponding edges are displayed on the image
    """
    correctly_positively_labeled_mask = (truth == 1) & (predictions == 1)
    false_positive_mask = (truth == 0) & (predictions == 1)
    false_negative_mask = (truth == 1) & (predictions == 0)

    masks = [correctly_positively_labeled_mask, false_positive_mask, false_negative_mask]
    names = ["True positive", "False Positive", "False Negative"]
    colors = ["green", "red", "blue"]

    for mask, name, color in zip(masks, names, colors):
        visualise_one_graph(graph, mask, name, score_img, writer, color=color)


def visualise_one_graph(graph, mask, name, score_img, writer: SummaryWriter, color: str):
    """
    Visualise one graph with the edges selected by the mask and save it with the tensorboard writer
    """
    fig, ax = plt.subplots(figsize=(7.5, 7.5), dpi=175)
    ax.imshow(score_img)

    graph = copy.deepcopy(graph).cpu()
    graph.edge_index = graph.edge_index[:, mask]
    pos = graph.pos.numpy()
    pos_scale = np.array([sc.item() for sc in graph.scale])
    pos = pos * pos_scale[0:2] + pos_scale[2:4]

    graph = to_networkx(graph)

    plt.title(name + " (num_edges = " + str(len(graph.edges)) + ")")
    nx.draw(graph,
            pos,
            alpha=1,
            node_size=0.8,
            node_color='red',
            edge_color=color,
            arrowsize=10,
            width=1)

    writer.add_figure(f"Score/{name}", fig)
    writer.flush()
    plt.close('all')


def save_confusion_matrix(confusion_mat, writer: SummaryWriter):
    """
    Save the confusion matrix as an images in tensorboard

    :param confusion_mat: as a 2x2 matrix
    :param writer: tensorboard writer
    :return: None
    """
    img_conf_mat = pd.DataFrame(confusion_mat / np.sum(confusion_mat, axis=1)[:, None],
                                index=[f'Actual Negative \n(sum = {np.sum(confusion_mat, axis=1)[0]})',
                                       f'Actual Positive \n(sum = {np.sum(confusion_mat, axis=1)[1]})'],
                                columns=[f'Predicted Negative \n(sum = {confusion_mat[0][0] + confusion_mat[1][0]})',
                                         f'Predicted Positive \n(sum = {confusion_mat[0][1] + confusion_mat[1][1]})'])

    # Make the heatmap
    plt.figure(figsize=(5, 5))
    sn.heatmap(img_conf_mat, annot=True)
    plt.xlabel('Predicted Label')
    plt.ylabel('Actual Label')
    plt.yticks(rotation=90)
    plt.title(f'Confusion Matrix')

    # create a little space around the plot
    plt.tight_layout()

    # Convert the Matplotlib figure to a PIL images
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=200)
    buffer.seek(0)
    img = Image.open(buffer)

    # Convert the images to a PyTorch tensor
    img = transforms.ToTensor()(img)

    # Add the images to the writer
    writer.add_image('Confusion Matrix', img)
    writer.flush()
    plt.close('all')
