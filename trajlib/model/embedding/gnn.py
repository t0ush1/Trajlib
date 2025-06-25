import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from torch_geometric.data import Data

from trajlib.model.embedding.embedding_trainer import EmbeddingTrainer


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
        elif embedding_name == "gcn":
            self.gnn = gnn.GCN(**gnn_config, cached=True)

    def forward(self, x, edge_index):
        x = self.embedding(x)
        x = self.gnn(x, edge_index)
        return x


class GAETrainer(EmbeddingTrainer):
    def __init__(self, emb_name, emb_dim, embs_path, data: Data):
        super().__init__(emb_name, embs_path, num_epochs=200, patience=15)

        self.data = data.to(self.device)
        self.model = gnn.GAE(GNNWithEmbedding(emb_name, len(data.num_nodes), emb_dim)).to(self.device)
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
