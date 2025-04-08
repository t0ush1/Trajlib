import os
import pickle
import torch
import torch.nn as nn
from torch_geometric.data import Data as GeoData

from trajlib.model.transformer import Transformer
from trajlib.model.positional_encoding import PositionalEncoding
from trajlib.model.embedding.gat import GATWithEmbedding


class TrajTransformer(nn.Module):
    def __init__(self, encoder_config, embedding, task_head):
        super(TrajTransformer, self).__init__()
        self.embedding = embedding
        self.positional_encoding = PositionalEncoding(encoder_config["d_model"], 10)
        self.encoder = Transformer(encoder_config)
        self.task_head = task_head

    def forward(self, x, geo_data: GeoData = None):
        if geo_data is not None:
            embedding = self.embedding(geo_data.x, geo_data.edge_index)
            x = embedding[x]
        else:
            x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        x = self.task_head(x)
        return x.unsqueeze(1)


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
<<<<<<< HEAD
        elif model_config["embedding"] == "gat":
            pass  # TODO 独立训练
        elif model_config["embedding"] == "gcn":
            pass  #
        
    elif data_config["data_form"] == "roadnet":
        if model_config["embedding"] == "node2vec":
            with open(model_config["embs_path"], "rb") as f:
                embeddings = pickle.load(f)
            return nn.Embedding.from_pretrained(embeddings=embeddings, freeze=True)
=======

        if embedding_config["emb_name"] == "normal":
            return nn.Embedding(vocab_size, emb_dim)
        elif embedding_config["emb_name"] == "gat":
            return GATWithEmbedding(vocab_size, emb_dim)
        elif embedding_config["emb_name"] == "gcn":
            pass
>>>>>>> d46ad5d9cc834b3e78370095a479189c6a674282


def create_task_head(config):
    data_config = config["data_config"]
    embedding_config = config["embedding_config"]

    if data_config["data_form"] == "gps":
        return nn.Linear(embedding_config["emb_dim"], 2)
    elif data_config["data_form"] == "grid":
<<<<<<< HEAD
        return nn.Linear(model_config["d_model"], model_config["vocab_size"])
    elif data_config["data_form"] == "roadnet":
        return nn.Linear(model_config["d_model"], model_config["vocab_size"])
=======
        return nn.Linear(embedding_config["emb_dim"], data_config["vocab_size"])
>>>>>>> d46ad5d9cc834b3e78370095a479189c6a674282


def create_model(config):
    embedding = create_embedding(config)
    task_head = create_task_head(config)
    return TrajTransformer(config["encoder_config"], embedding, task_head)
