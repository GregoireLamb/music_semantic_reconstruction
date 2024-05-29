import torch.nn.functional as F

from src.datasetHandler import *
from src.gnn_layer.Graph_conv import SpatialGraphConv


class GNN_geoGCN(torch.nn.Module):
    def __init__(self, num_node_features):
        super(GNN_geoGCN, self).__init__()

        self.conv1_config = {'coors': 2,
                             'in_channels': num_node_features,
                             'out_channels': 2048,
                             'hidden_size': 1,
                             'dropout': 0}
        self.conv2_config = {'coors': 2,
                             'in_channels': 2048,
                             'out_channels': 2048,
                             'hidden_size': 1,
                             'dropout': 0}
        self.conv3_config = {'coors': 2,
                             'in_channels': 2048,
                             'out_channels': 1024,
                             'hidden_size': 1,
                             'dropout': 0}

        self.conv1 = SpatialGraphConv(**self.conv1_config)
        self.conv2 = SpatialGraphConv(**self.conv2_config)
        self.conv3 = SpatialGraphConv(**self.conv3_config)

    def forward(self, x, pos, edge_index):
        x, pos, edge_index = x, pos, edge_index

        x = self.conv1(x, pos, edge_index)
        x = F.relu(x)
        x = self.conv2(x, pos, edge_index)
        x = F.relu(x)
        x = self.conv3(x, pos, edge_index)
        x = F.relu(x)
        x = F.normalize(x, dim=1)

        return x
