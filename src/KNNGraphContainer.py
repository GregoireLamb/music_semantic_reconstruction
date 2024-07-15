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
                 config: Config, device: torch.device, scale=(1., 1., 0, 0), root="") -> None:
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
        n_neighbors = self.config['n_neighbors_KNN']
        labels_to_use = self.config['labels_to_use']
        normalize_positions = self.config['normalize_positions']
        prefilter_KNN = self.config['prefilter_KNN']
        position_as_bounding_box = self.config['position_as_bounding_box']
        # mono_filtering = self.config['mono_filtering'] #TODO what to do with it ?

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
                    top, bottom, left, right = 0,0,0,0
                    if position_as_bounding_box:
                        top = torch.min(self.graph.x[:, -4:])
                        left = torch.min(self.graph.x[:, -3:])
                        bottom = torch.max(self.graph.x[:, -2:])
                        right = torch.max(self.graph.x[:, -1:])

                        self.graph.x[:, -4:] = self.graph.x[:, -4:] - torch.tensor([top, left, top, left], dtype=torch.float)
                    else:
                        bb_top = - self.graph.x[:, -4:-3]/2 + self.graph.x[:, -2:]
                        bb_bottom = self.graph.x[:, -4:-3]/2 + self.graph.x[:, -2:]
                        bb_left = - self.graph.x[:, -3:-2]/2 + self.graph.x[:, -1:]
                        bb_right = self.graph.x[:, -3:-2]/2 + self.graph.x[:, -1:]

                        top = torch.min(bb_top)
                        left = torch.min(bb_bottom)
                        bottom = torch.max(bb_left)
                        right = torch.max(bb_right)

                        self.graph.x[:, -2:] = self.graph.x[:, -2:] - torch.tensor([top, left], dtype=torch.float)

                    self.scale = (right - left, bottom - top, top, left)
                    self.graph.scale = self.scale

                    scale_double = torch.tensor([self.scale[0], self.scale[1], self.scale[0], self.scale[1]],
                                                dtype=torch.float)
                    scale_simple = torch.tensor([self.scale[0], self.scale[1]], dtype=torch.float)
                    self.graph.pos = (self.graph.pos - torch.tensor([top, left], dtype=torch.float)) / scale_simple
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
                    folder2create = '/'.join(folders_list[:i + 1])
                    if not os.path.isdir(folder2create):
                        os.mkdir(f'{folder2create}')

            # save the graph
            # save_path = u"\\\\?\\" + os.path.abspath(f'{processing_folder}\KNNGraphContainer_{self.index}.pt') # windows
            save_path = os.path.abspath(f'{processing_folder}\KNNGraphContainer_{self.index}.pt') # LInux
            torch.save(self.graph, save_path)

    def prefilter_KNNGraph(self, graph):
        """"
        Cut impossible edges based on the node type
        """
        edges = self.load_authorised_edges()

        label_encoder = self.label_encoder
        mask = np.ones(len(graph.edge_index[0]), dtype=bool)
        for i in range(len(graph.edge_index[0])):
            node1 = label_encoder.inverse_transform([np.where(graph.x[graph.edge_index[0][i]] == 1)[0][0]])[0]
            node2 = label_encoder.inverse_transform([np.where(graph.x[graph.edge_index[1][i]] == 1)[0][0]])[0]

            if node1 + " - " + node2 not in edges:
                mask[i] = False

        graph.edge_index = graph.edge_index[:, mask]
        return graph

    def load_authorised_edges(self):
        """
        Load the edges that are allowed in the graph
        """
        edges = set()
        path = os.path.abspath("./data/labels_and_links/")
        with open(f"{path}/{self.config['labels_to_use']}_links.txt", "r") as f:
            for line in f:
                edges.add(line.strip())
        return edges
