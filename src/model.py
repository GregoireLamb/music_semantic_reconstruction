import torch
from torch import Tensor

from src.gnn_layer.Model_GATConv import GNN_GATConv
from src.gnn_layer.Model_GeoGCN import GNN_geoGCN
from src.gnn_layer.Model_GraphSAGE import GNN_graphSage


class Classifier(torch.nn.Module):
    """
    This class is used to combine node embeddings to predict the existence of an edge between them.
    """
    def __init__(self):
        super(Classifier, self).__init__()

    def forward(self, node_embeddings: Tensor, edge_index: Tensor) -> Tensor:
        edge_feat_node_1 = node_embeddings[edge_index[0]]
        edge_feat_node_2 = node_embeddings[edge_index[1]]

        cross_p = (edge_feat_node_1 * edge_feat_node_2).sum(dim=-1)
        return cross_p


class Model(torch.nn.Module):
    """
    This class is used to combine the GNN and the classifier.
    It handle the different type of GNN layers.
    """

    def __init__(self, num_node_features, config):
        super().__init__()
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
        elif layer_type == "graphSAGE_medium":
            self.gnn = GNN_graphSage(num_node_features, medium=True)
        elif layer_type == "graphSAGE_large":
            self.gnn = GNN_graphSage(num_node_features, large=True)
        elif layer_type == "gatconv":
            self.gnn = GNN_GATConv(num_node_features)
        else:
            raise ValueError("Layer type not recognized")

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
        if hasattr(self.gnn, 'conv4_config'):
            msg += f', {self.gnn.conv4_config}'

        return msg
