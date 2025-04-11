import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn

from trajlib.data.data import GraphData
from trajlib.model.embedding.embedding_trainer import EmbeddingTrainer


class GCNEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = gnn.GCNConv(in_channels, 2 * out_channels, cached=True)
        self.conv2 = gnn.GCNConv(2 * out_channels, out_channels, cached=True)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x


class GATEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GATEncoder, self).__init__()
        self.conv1 = gnn.GATConv(in_channels, out_channels // 8, heads=4)
        self.conv2 = gnn.GATConv(out_channels // 2, out_channels, heads=1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x


class GNNWithEmbedding(nn.Module):
    def __init__(self, embedding_name, num_nodes, embedding_dim):
        super(GNNWithEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_nodes, 32)
        gnn_config = {
            "in_channels": 32,
            "hidden_channels": 128,
            "num_layers": 2,
            "out_channels": embedding_dim,
        }
        if embedding_name == "gat":
            self.gnn = gnn.GAT(**gnn_config, heads=4)
        else:
            self.gnn = gnn.GCN(**gnn_config, cached=True)

    def forward(self, x, edge_index):
        x = self.embedding(x)
        x = self.gnn(x, edge_index)
        return x


class GAETrainer(EmbeddingTrainer):
    def __init__(self, embedding_config, graph_data: GraphData):
        name = embedding_config["emb_name"]
        extra_config = {
            "ckpt_path": f"./resource/model/embedding/{name}.pth",
            "embs_path": f"./resource/model/embedding/{name}.pkl",
            "num_epochs": 200,
            "patience": 15,
        }
        super().__init__(embedding_config, extra_config)

        self.data = graph_data.to_geo_data().to(self.device)
        self.model = gnn.GAE(GNNWithEmbedding(name, len(graph_data.nodes), embedding_config["emb_dim"])).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-2)

    def _train_one_epoch(self):
        self.optimizer.zero_grad()
        z = self.model.encode(self.data.x, self.data.edge_index)
        loss = self.model.recon_loss(z, self.data.edge_index)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _get_embs(self):
        return self.model.encode(self.data.x, self.data.edge_index)
