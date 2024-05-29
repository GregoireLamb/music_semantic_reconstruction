import os

import numpy as np
import torch
import torch_geometric.transforms as T
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data.data import Data

from src.config import Config


class KNNGraphContainer:
    """
    This class is used to store a score as its KNN graph, its original (true) graph and other
    parameters used to reconstruct the score
    """

    def __init__(self, raw_data: Data, index: int, label_encoder: LabelEncoder,
                 config: Config, device: torch.device, scale=(1., 1.), root="") -> None:
        """
        :param raw_data: Data object corresponding to the score in the dataset (nodes might be filtered already)
        :param index: Score id
        :param label_encoder: label encoder to used to encode the score in PyG format
        :param config: config object
        :param device: torch device where computation will be done
        :param scale: scale used during the PyG encoding of the score
        :param root: root folder to the project
        """
        self.raw_data = raw_data
        self.config = config
        self.label_encoder = label_encoder
        self.device = device
        self.index = index
        self.root = root
        self.scale = scale
        self.truth = None
        self.graph = None

        self.generate_graph_and_truth()

    def get_data(self):
        return self.graph

    def generate_graph_and_truth(self) -> None:
        """
        Instantiate the graph object with the truth and the original edges
        """
        n_neighbors = self.config.__getitem__('n_neighbors_KNN')
        labels_to_use = self.config.__getitem__('labels_to_use')
        normalize_positions = self.config.__getitem__('normalize_positions')
        prefilter_KNN = self.config.__getitem__('prefilter_KNN')
        position_as_bounding_box = self.config.__getitem__('position_as_bounding_box')
        # mono_filtering = self.config.__getitem__('mono_filtering') #TODO what to do with it ?

        processing_folder = (f'{self.root}KNNGraphContainers/'
                             f'{labels_to_use}/n_neighbors_{n_neighbors}/'
                             f'filtered_KNN_{prefilter_KNN}/normalise_positions_{normalize_positions}/'
                             f'using_bounding_boxe_{position_as_bounding_box}/')

        if os.path.isfile(
                f'{processing_folder}/KNNGraphContainer_{self.index}.pt'):  # KNNGraphContainer already exists
            self.graph = torch.load(f'{processing_folder}/KNNGraphContainer_{self.index}.pt')
        else:

            if len(self.raw_data.edge_index.shape) < 2:  # No edge in the graph
                true_edges_set = set()
            else:
                true_edges_set = {(self.raw_data.edge_index[0][i].item(), self.raw_data.edge_index[1][i].item()) for i
                                  in range(len(self.raw_data.edge_index[0]))}

            self.graph = T.KNNGraph(k=n_neighbors)(self.raw_data)

            if normalize_positions:
                if self.graph.x.shape[0] > 0:  # skip case where score doesn't contain any object
                    scale = torch.tensor(self.scale, dtype=torch.float)
                    scale_double = torch.tensor([self.scale[0], self.scale[1], self.scale[0], self.scale[1]],
                                                dtype=torch.float)
                    self.graph.pos = self.graph.pos / scale
                    self.graph.x[:, -4:] = self.graph.x[:, -4:] / scale_double

            if prefilter_KNN:
                self.graph = self.prefilter_KNNGraph(self.graph)

            # Generate truth, a tensor of 1 if the edge of the KNN graph is in the original graph, 0 otherwise
            knn_edges_set_list = [(self.graph.edge_index[0][i].item(), self.graph.edge_index[1][i].item()) for i in
                                  range(len(self.graph.edge_index[0]))]

            self.graph.truth = torch.tensor([1 if edge in true_edges_set else 0 for edge in knn_edges_set_list],
                                            dtype=torch.float)
            self.graph.index = self.index

            if len(self.raw_data.edge_index.shape) < 2:
                self.graph.original_edges_in = torch.tensor([])
                self.graph.original_edges_out = torch.tensor([])
            else:
                self.graph.original_edges_in = self.raw_data.edge_index[0].clone().detach()
                self.graph.original_edges_out = self.raw_data.edge_index[1].clone().detach()

            # make directory if not exist
            if not os.path.isdir(f'{processing_folder}/'):
                folders_list = processing_folder.split('/')
                for i in range(1, len(folders_list)):
                    processing_folder = '/'.join(folders_list[:i + 1])
                    if not os.path.isdir(processing_folder):
                        os.mkdir(f'{processing_folder}')

            # save the graph
            torch.save(self.graph, f'{processing_folder}/KNNGraphContainer_{self.index}.pt')

    #     if hetero_data:
    #         self.make_hetero_data()

    def prefilter_KNNGraph(self, graph):
        """"
        Cut impossible edges based on the node type
        """
        edges = {'notehead-full - stem', 'notehead-full - beam', 'notehead-full - 8th_flag', 'notehead-empty - stem',
                 'notehead-full - duration-dot', 'notehead-full - sharp',
                 'notehead-full - natural', 'notehead-full - tie', 'notehead-empty - duration-dot',
                 'notehead-full - 16th_flag', 'notehead-empty - tie',
                 'notehead-full - flat', 'notehead-empty - sharp', 'notehead-empty - natural'}

        label_encoder = self.label_encoder
        mask = np.ones(len(graph.edge_index[0]), dtype=bool)
        for i in range(len(graph.edge_index[0])):
            node1 = label_encoder.inverse_transform([np.where(graph.x[graph.edge_index[0][i]] == 1)[0][0]])[0]
            node2 = label_encoder.inverse_transform([np.where(graph.x[graph.edge_index[1][i]] == 1)[0][0]])[0]

            if node1 + " - " + node2 not in edges:
                mask[i] = False

        graph.edge_index = graph.edge_index[:, mask]
        return graph

    # def make_hetero_data(self):
    #     hetero_graph = HeteroData()
    #     corresp_id_hetero_id = {}  # PyG hetero W with id starting from 0 for each class
    #
    #     if self.graph.x.shape[0] == 0:
    #         self.graph = hetero_graph
    #         self.graph.validate()
    #         return
    #     label_id = {}
    #     new_x = self.graph.x[:, -4:]  # Keep only the position (bounding box)
    #     dict_x = {}
    #
    #     for i in range(len(self.graph.x)):
    #         label = self.label_encoder.inverse_transform([np.where(self.graph.x[i] == 1)[0][0]])[0]
    #         if label in dict_x:
    #             dict_x[label].append(new_x[i].tolist())
    #         else:
    #             dict_x[label] = [new_x[i].tolist()]
    #
    #         if label not in label_id:
    #             label_id[label] = 0
    #         corresp_id_hetero_id[i] = label_id[label]
    #         label_id[label] += 1
    #
    #     for k, v in dict_x.items():
    #         hetero_graph[k].x = torch.Tensor(v)
    #
    #     dict_edge = {}
    #
    #     for i in range(self.graph.edge_index.shape[1]):
    #         primitive1 = self.graph.edge_index[0][i].item()
    #         primitive2 = self.graph.edge_index[1][i].item()
    #         id1 = np.where(self.graph.x[primitive1] == 1)[0][0]
    #         id2 = np.where(self.graph.x[primitive2] == 1)[0][0]
    #         class1 = self.label_encoder.inverse_transform([id1])[0]
    #         class2 = self.label_encoder.inverse_transform([id2])[0]
    #
    #         if class1 + "." + class2 not in dict_edge:
    #             dict_edge[class1 + "." + class2] = ([], [])
    #
    #         dict_edge[class1 + "." + class2][0].append(corresp_id_hetero_id[self.graph.edge_index[0][i].item()])
    #         dict_edge[class1 + "." + class2][1].append(corresp_id_hetero_id[self.graph.edge_index[1][i].item()])
    #
    #     for k, v in dict_edge.items():
    #         k = k.split(".")
    #         hetero_graph[k[0], "into", k[1]].edge_index = torch.Tensor(v)
    #
    #     hetero_graph.pos = self.graph.pos
    #     hetero_graph.truth = self.graph.truth
    #     hetero_graph.index = self.graph.index
    #     hetero_graph.original_edges_in = self.graph.original_edges_in
    #     hetero_graph.original_edges_out = self.graph.original_edges_out
    #
    #     self.graph = hetero_graph
    #     self.graph.validate()
