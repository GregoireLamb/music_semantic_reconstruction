import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GNN_GATConv(torch.nn.Module):
    def __init__(self, num_node_features):
        super(GNN_GATConv, self).__init__()

        self.conv1_config = {'in_channels': num_node_features,
                             'out_channels': 2048}
        self.conv2_config = {'in_channels': 2048,
                             'out_channels': 2048}
        self.conv3_config = {'in_channels': 2048,
                             'out_channels': 1024}

        self.conv1 = GATConv(**self.conv1_config)
        self.conv2 = GATConv(**self.conv2_config)
        self.conv3 = GATConv(**self.conv3_config)

    def forward(self, x, _, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = F.normalize(x, dim=1)

        return x
