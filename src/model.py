import torch
from torch import Tensor

from src.gnn_layer.Model_GATConv import GNN_GATConv
from src.gnn_layer.Model_GeoGCN import GNN_geoGCN
from src.gnn_layer.Model_GraphSAGE import GNN_graphSage


class Classifier(torch.nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

    def forward(self, node_embeddings: Tensor, edge_label_index: Tensor) -> Tensor:
        # Convert node embeddings to edge-level representations:

        # if not torch.is_tensor(edge_label_index): # Deals with the case where edge_label_index is a sparse tensor
        #     # TODO might do the conversion earlier in the pipeline for efficiency
        #     row, col , edge_attr = edge_label_index.t().coo()
        #     edge_label_index = torch.stack([row, col], dim=0)

        edge_feat_node_1 = node_embeddings[edge_label_index[0]]
        edge_feat_node_2 = node_embeddings[edge_label_index[1]]

        cross_p = (edge_feat_node_1 * edge_feat_node_2).sum(dim=-1)
        return cross_p


class Model(torch.nn.Module):

    def __init__(self, num_node_features, config):
        self.config = config
        self.classifier = Classifier()
        self.gnn = None
        layer_type = config.__getitem__("layer_type")

        if layer_type == "geoGCN":
            self.gnn = GNN_geoGCN(num_node_features)
        elif layer_type == "graphSAGE":
            self.gnn = GNN_graphSage(num_node_features)
        elif layer_type == "graphSAGE_small":
            self.gnn = GNN_graphSage(num_node_features, small=True)
        elif layer_type == "gatconv":
            self.gnn = GNN_GATConv(num_node_features)
        else:
            raise ValueError("Layer type not recognized")

        super().__init__()

    def forward(self, x, pos, edge_index) -> Tensor:
        x_dict = self.gnn(x, pos, edge_index)
        predictions = self.classifier(x_dict, edge_index)
        return predictions

    def __repr__(self):
        msg = ''
        # add the conv layer description taht are used in the GNN
        if hasattr(self.gnn, 'conv1_config'):
            msg += f', {self.gnn.conv1_config}'
        if hasattr(self.gnn, 'conv2_config'):
            msg += f', {self.gnn.conv2_config}'
        if hasattr(self.gnn, 'conv3_config'):
            msg += f', {self.gnn.conv3_config}'

        return msg
