import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class GNN_graphSage(torch.nn.Module):
    def __init__(self, num_node_features, small=False, medium=False, large=False):
        super(GNN_graphSage, self).__init__()
        self.small = small
        self.medium = medium
        self.large = large

        self.conv1_config = {'in_channels': num_node_features,
                             'out_channels': 2048,
                             'aggr': 'mean'}
        self.conv2_config = {'in_channels': 2048,
                             'out_channels': 2048,
                             'aggr': 'mean'}
        self.conv3_config = {'in_channels': 2048,
                             'out_channels': 1024,
                             'aggr': 'mean'}
        if large:
            self.conv3_config = {'in_channels': 2048,
                                 'out_channels': 2048,
                                 'aggr': 'mean'}
            self.conv4_config = {'in_channels': 2048,
                                 'out_channels': 1024,
                                 'aggr': 'mean'}

        if small:
            self.conv1_config = {'in_channels': num_node_features,
                                 'out_channels': 1024,
                                 'aggr': 'mean'}

        if medium:
            self.conv1_config = {'in_channels': num_node_features,
                                 'out_channels': 2048,
                                 'aggr': 'mean'}
            self.conv2_config = {'in_channels': 2048,
                                 'out_channels': 1024,
                                 'aggr': 'mean'}

        self.conv1 = SAGEConv(**self.conv1_config)
        if not small:
            self.conv2 = SAGEConv(**self.conv2_config)
            if not medium:
                self.conv3 = SAGEConv(**self.conv3_config)
                if large:
                    self.conv4 = SAGEConv(**self.conv4_config)

    def forward(self, x, _, edge_index):

        x = self.conv1(x, edge_index)
        x = F.relu(x)

        if not self.small:
            x = self.conv2(x, edge_index)
            x = F.relu(x)

            if not self.medium:
                x = self.conv3(x, edge_index)
                x = F.relu(x)

                if self.large:
                    x = self.conv4(x, edge_index)
                    x = F.relu(x)

        x = F.normalize(x, dim=1)
        return x
