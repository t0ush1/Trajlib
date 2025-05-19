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

    if data_config["data_form"] == "gps":
        return nn.Linear(2, emb_dim)

    elif data_config["data_form"] == "grid":
        if embedding_config["pre-trained"]:
            with open(embedding_config["embs_path"], "rb") as f:
                embeddings = pickle.load(f)
            return nn.Embedding.from_pretrained(embeddings=embeddings, freeze=True)

        if embedding_config["emb_name"] == "normal":
            return nn.Embedding(vocab_size, emb_dim)
        elif embedding_config["emb_name"] in ["gat", "gcn"]:
            return GNNWithEmbedding(vocab_size, emb_dim)


def create_task_head(config):
    task_config = config["task_config"]
    data_config = config["data_config"]
    embedding_config = config["embedding_config"]

    if task_config["task_name"] == "prediction":
        if data_config["data_form"] == "gps":
            return nn.Linear(embedding_config["emb_dim"], 2)
        elif data_config["data_form"] == "grid":
            return nn.Linear(embedding_config["emb_dim"], data_config["vocab_size"])

    elif task_config["task_name"] == "similarity":
        return nn.Identity()


def create_model(config):
    embedding = create_embedding(config)
    task_head = create_task_head(config)
    return TrajTransformer(config["encoder_config"], embedding, task_head)
