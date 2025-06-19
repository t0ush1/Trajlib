import os
import pickle
import torch
import torch.nn as nn
from torch_geometric.data import Data as GeoData

from trajlib.data.data import SPECIAL_TOKENS
from trajlib.model.transformer import Transformer
from trajlib.model.lstm import LSTM
from trajlib.model.cnn import CNN
from trajlib.model.mlp import MLP
from trajlib.model.positional_encoding import PositionalEncoding
from trajlib.model.embedding.gnn import GNNWithEmbedding


class TrajEmbedding(nn.Module):
    def __init__(self, embedding, emb_dim, tokenized=False):
        super(TrajEmbedding, self).__init__()
        self.embedding = embedding
        self.tokenized = tokenized
        if tokenized:
            self.pad_emb = nn.Parameter(torch.zeros(emb_dim), requires_grad=False)
            self.mask_emb = nn.Parameter(torch.randn(emb_dim))
            self.special_embs = {
                SPECIAL_TOKENS["pad"]: self.pad_emb,
                SPECIAL_TOKENS["mask"]: self.mask_emb,
            }

    def forward(self, x, geo_data):
        if not self.tokenized:
            return self.embedding(x)

        masks = {idx: x == idx for idx in self.special_embs}
        for mask in masks.values():
            x[mask] = 0

        x = x.long() 
        if geo_data is None:
            x = self.embedding(x)  # 当 geo_data 为 None 时，只传递 x
        else:
            x = self.embedding(geo_data.x, geo_data.edge_index)[x]

        for idx, mask in masks.items():
            x[mask] = self.special_embs[idx]  # TODO 广播？

        return x


class TrajBackbone(nn.Module):
    def __init__(self, d_model, embedding, encoder, pe, pooling, task_head):
        super(TrajBackbone, self).__init__()
        self.embedding = embedding
        self.pe = pe
        if self.pe:
            self.positional_encoding = PositionalEncoding(d_model)
        self.encoder = encoder
        self.pooling = pooling
        self.task_head = task_head

    def forward(self, x, mask=None, geo_data: GeoData = None):
        # if geo_data is None:
        #     x = self.embedding(x)  # 当 geo_data 为 None 时，只传递 x
        # else:
        #     x = self.embedding(geo_data.x, geo_data.edge_index)[x]

        x = self.embedding(x, geo_data)

        if self.pe:
            x = self.positional_encoding(x)
        x = self.encoder(x, mask)
        if self.pooling: # TODO 加CLS，取最后一个
            x = x.mean(dim=1)
            # x = x[:, -1, :]  # lstm?
        x = self.task_head(x)
        return x


def create_embedding(config):
    data_config = config["data_config"]
    embedding_config = config["embedding_config"]
    vocab_size = data_config["vocab_size"]
    emb_dim = embedding_config["emb_dim"]

    match (data_config, embedding_config):
        case {"data_form": "gps"}, _:
            embedding = nn.Linear(2, emb_dim)
        case {"data_form": "grid" | "roadnet"}, {"pre-trained": True, "embs_path": embs_path}:
            with open(embs_path, "rb") as f:
                embeddings = pickle.load(f)
            embedding = nn.Embedding.from_pretrained(embeddings=embeddings, freeze=True)
        case {"data_form": "grid" | "roadnet"}, {"emb_name": "normal"}:
            embedding = nn.Embedding(vocab_size, emb_dim)
        case {"data_form": "grid" | "roadnet"}, {"emb_name": "gat" | "gcn"}:
            embedding = GNNWithEmbedding(embedding_config["emb_name"], vocab_size, emb_dim)
        case _:
            raise ValueError()

    return TrajEmbedding(embedding, emb_dim, tokenized=data_config["data_form"] != "gps")


def create_encoder(config):
    encoder_config = config["encoder_config"]
    encoder_name = encoder_config["encoder_name"]

    match encoder_name:
        case "transformer":
            return True, Transformer(encoder_config)
        case "lstm":
            return False, LSTM(encoder_config)
        case "cnn":
            return True, CNN(encoder_config)
        case "mlp":
            return True, MLP(encoder_config)
        case _:
            raise ValueError()


def create_task_head(config):
    task_config = config["task_config"]
    data_config = config["data_config"]
    embedding_config = config["embedding_config"]
    emb_dim = embedding_config["emb_dim"]

    match (task_config, data_config):
        case ({"task_name": "prediction"}, {"data_form": "gps"}):
            return True, nn.Linear(emb_dim, 2)
        case ({"task_name": "prediction"}, {"data_form": "grid" | "roadnet", "vocab_size": vocab_size}):
            return True, nn.Linear(emb_dim, vocab_size)
        case ({"task_name": "similarity"}, _):
            return True, nn.Identity()
        case ({"task_name": "filling"}, {"data_form": "grid" | "roadnet", "vocab_size": vocab_size}):
            return False, nn.Linear(emb_dim, vocab_size)
        case _:
            raise ValueError()


def create_model(config):
    embedding = create_embedding(config)
    pooling, task_head = create_task_head(config)
    pe, encoder = create_encoder(config)
    d_model = config["encoder_config"]["d_model"]
    return TrajBackbone(d_model, embedding, encoder, pe, pooling, task_head)