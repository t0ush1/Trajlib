import torch
import torch_geometric.nn as gnn
from tqdm import tqdm

from trajlib.data.data import GraphData
from trajlib.model.embedding.embedding_trainer import EmbeddingTrainer


class Node2VecTrainer(EmbeddingTrainer):
    def __init__(self, embedding_config, graph_data: GraphData):
        extra_config = {
            "ckpt_path": "./resource/model/embedding/node2vec.pth",
            "embs_path": "./resource/model/embedding/node2vec.pkl",
            "num_epochs": 8,
            "patience": 8,
        }
        super().__init__(embedding_config, extra_config)

        self.data = graph_data.to_geo_data().to(self.device)
        self.model = gnn.Node2Vec(
            self.data.edge_index,
            embedding_dim=embedding_config["emb_dim"],
            walk_length=50,
            context_size=10,
            walks_per_node=10,
            num_negative_samples=10,
            p=1,
            q=1,
            sparse=True,
        ).to(self.device)
        self.loader = self.model.loader(batch_size=16, shuffle=True, num_workers=0)
        self.optimizer = torch.optim.SparseAdam(self.model.parameters(), lr=0.001)

    def _train_one_epoch(self):
        total_loss = 0
        for pos_rw, neg_rw in tqdm(self.loader):
            self.optimizer.zero_grad()
            pos_rw, neg_rw = pos_rw.to(self.device), neg_rw.to(self.device)
            loss = self.model.loss(pos_rw, neg_rw)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(self.loader)

    def _get_embs(self):
        return self.model()
