import os
import pickle
import torch
import torch.nn as nn
from torch_geometric.data import Data as GeoData

from trajlib.model.transformer import Transformer
from trajlib.model.positional_encoding import PositionalEncoding
from trajlib.model.embedding.gnn import GNNWithEmbedding


class TrajTransformer(nn.Module):
    def __init__(self, encoder_config, embedding, task_head):
        super(TrajTransformer, self).__init__()
        self.embedding = embedding
        self.positional_encoding = PositionalEncoding(encoder_config["d_model"])
        self.encoder = Transformer(encoder_config)
        self.task_head = task_head

    def forward(self, x, mask=None, geo_data: GeoData = None):
        if geo_data is not None:
            embedding = self.embedding(geo_data.x, geo_data.edge_index)
            x = embedding[x]
        else:
            x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.encoder(x, mask)
        x = x.mean(dim=1)
        x = self.task_head(x)
        return x


def create_embedding(config):
    data_config = config["data_config"]
    embedding_config = config["embedding_config"]
    vocab_size = data_config["vocab_size"]
    emb_dim = embedding_config["emb_dim"]

    match (data_config, embedding_config):
        case {"data_form": "gps"}, _:
            return nn.Linear(2, emb_dim)
        case {"data_form": "grid"}, {"pre-trained": True, "embs_path": embs_path}:
            with open(embs_path, "rb") as f:
                embeddings = pickle.load(f)
            return nn.Embedding.from_pretrained(embeddings=embeddings, freeze=True)
        case {"data_form": "grid"}, {"emb_name": "normal"}:
            return nn.Embedding(vocab_size, emb_dim)
        case {"data_form": "grid"}, {"emb_name": "gat" | "gcn"}:
            return GNNWithEmbedding(vocab_size, emb_dim)
        case _:
            raise ValueError()


def create_task_head(config):
    task_config = config["task_config"]
    data_config = config["data_config"]
    embedding_config = config["embedding_config"]
    emb_dim = embedding_config["emb_dim"]

    match (task_config, data_config):
        case ({"task_name": "prediction"}, {"data_form": "gps"}):
            return nn.Linear(emb_dim, 2)
        case ({"task_name": "prediction"}, {"data_form": "grid", "vocab_size": vocab_size}):
            return nn.Linear(emb_dim, vocab_size)
        case ({"task_name": "similarity"}, _):
            return nn.Identity()
        case _:
            raise ValueError()


def create_model(config):
    embedding = create_embedding(config)
    task_head = create_task_head(config)
    return TrajTransformer(config["encoder_config"], embedding, task_head)
