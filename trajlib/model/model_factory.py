import os
import pickle
import torch
import torch.nn as nn
from torch_geometric.data import Data as GeoData

from trajlib.data.data import SpecialToken
from trajlib.model.embedding.embedding_trainer import EmbeddingTrainer
from trajlib.model.embedding.gnn import GAETrainer
from trajlib.model.embedding.node2vec import Node2VecTrainer
from trajlib.model.transformer import Transformer
from trajlib.model.positional_encoding import PositionalEncoding
from trajlib.model.embedding.gnn import GNNWithEmbedding


class GPSEmbedding(nn.Module):
    def __init__(self, embedding):
        super(GPSEmbedding, self).__init__()
        self.embedding = embedding

    def forward(self, coordinates, **kwargs):
        return self.embedding(coordinates)


class GridEmbedding(nn.Module):
    def __init__(self, embedding, is_gnn, emb_dim):
        super(GridEmbedding, self).__init__()
        self.embedding = embedding
        self.is_gnn = is_gnn
        self.unk_emb = nn.Parameter(torch.randn(emb_dim))

    def forward(self, grid_ids, grid_geo_data: GeoData, special_pos, **kwargs):
        unk_pos = grid_ids == SpecialToken.UNKNOWN
        grid_ids[special_pos | unk_pos] = 0

        if not self.is_gnn:
            x = self.embedding(grid_ids)
        else:
            x = self.embedding(grid_geo_data.x, grid_geo_data.edge_index)[grid_ids]

        x[unk_pos] = self.unk_emb
        return x


class RoadnetEmbedding(nn.Module):
    def __init__(self, embedding, is_gnn, emb_dim):
        super(RoadnetEmbedding, self).__init__()
        self.embedding = embedding
        self.is_gnn = is_gnn
        self.unk_emb = nn.Parameter(torch.randn(emb_dim))

    def forward(self, road_ids, road_geo_data: GeoData, special_pos, **kwargs):
        unk_pos = road_ids == SpecialToken.UNKNOWN
        road_ids[special_pos | unk_pos] = 0

        if not self.is_gnn:
            x = self.embedding(road_ids)
        else:
            x = self.embedding(road_geo_data.x, road_geo_data.edge_index)[road_ids]

        x[unk_pos] = self.unk_emb
        return x


class TrajEmbedding(nn.Module):
    def __init__(self, embeddings, emb_dim):
        super(TrajEmbedding, self).__init__()
        self.embeddings = nn.ModuleDict(embeddings)
        self.pad_emb = nn.Parameter(torch.zeros(emb_dim), requires_grad=False)
        self.mask_emb = nn.Parameter(torch.randn(emb_dim))

    def forward(self, road_ids, **kwargs):
        pad_pos = road_ids == SpecialToken.PAD
        mask_pos = road_ids == SpecialToken.MASK

        x = 0
        for emb in self.embeddings.values():
            x = x + emb(special_pos=pad_pos | mask_pos, road_ids=road_ids, **kwargs)

        x[pad_pos] = self.pad_emb
        x[mask_pos] = self.mask_emb

        return x


class TrajTransformer(nn.Module):
    def __init__(self, encoder_config, embedding, pooling, task_head):
        super(TrajTransformer, self).__init__()
        self.embedding = embedding
        self.positional_encoding = PositionalEncoding(encoder_config["d_model"])
        self.encoder = Transformer(encoder_config)
        self.pooling = pooling
        self.task_head = task_head

    def forward(self, mask=None, **kwargs):
        x = self.embedding(**kwargs)
        x = self.positional_encoding(x)
        x = self.encoder(x, mask)
        if self.pooling:
            x = x.mean(dim=1)
            # x = x[:, -1, :]
        x = self.task_head(x)
        return x


def pretrain_embedding(config, grid_geo_data, road_geo_data, overwrite=False):
    embedding_config = config["embedding_config"]
    emb_dim = embedding_config["emb_dim"]

    mapper: dict[str, type[EmbeddingTrainer]] = {
        "node2vec": Node2VecTrainer,
        "gat": GAETrainer,
        "gcn": GAETrainer,
    }

    for token, data in zip(["grid", "roadnet"], [grid_geo_data, road_geo_data]):
        sub_config = embedding_config[token]
        emb_name = sub_config["emb_name"]
        embs_path = sub_config["embs_path"]
        if sub_config["pre-trained"] and (overwrite or not os.path.exists(embs_path)):
            print("Pre-train embedding for", token)
            trainer = mapper[emb_name](emb_name, emb_dim, embs_path, data)
            trainer.train()


def create_embedding(config):
    embedding_config = config["embedding_config"]
    emb_dim = embedding_config["emb_dim"]

    embeddings = {}
    for token in embedding_config["tokens"]:
        sub_config = embedding_config[token]
        match sub_config:
            case {"emb_name": "linear"}:
                embedding = nn.Linear(2, emb_dim)
                is_gnn = False
            case {"pre-trained": True, "embs_path": embs_path}:
                with open(embs_path, "rb") as f:
                    embs = pickle.load(f)
                embedding = nn.Embedding.from_pretrained(embeddings=embs, freeze=True)
                is_gnn = False
            case {"emb_name": "embedding", "vocab_size": vocab_size}:
                embedding = nn.Embedding(vocab_size, emb_dim)
                is_gnn = False
            case {"emb_name": "gat" | "gcn", "vocab_size": vocab_size}:
                embedding = GNNWithEmbedding(sub_config["emb_name"], vocab_size, emb_dim)
                is_gnn = True
            case _:
                raise ValueError()

        match token:
            case "gps":
                embedding = GPSEmbedding(embedding)
            case "grid":
                embedding = GridEmbedding(embedding, is_gnn, emb_dim)
            case "roadnet":
                embedding = RoadnetEmbedding(embedding, is_gnn, emb_dim)
            case _:
                raise ValueError()

        embeddings[token] = embedding
    return TrajEmbedding(embeddings, emb_dim)


def create_task_head(config):
    task_config = config["task_config"]
    embedding_config = config["embedding_config"]
    emb_dim = embedding_config["emb_dim"]

    pooling = task_config["task_name"] != "filling"

    match task_config:
        case {"task_name": "prediction" | "filling", "token": "gps"}:
            task_head = nn.Linear(emb_dim, 2)
        case {"task_name": "prediction" | "filling", "token": "grid" | "roadnet" as token}:
            task_head = nn.Linear(emb_dim, embedding_config[token]["vocab_size"])
        case {"task_name": "similarity"}:
            task_head = nn.Identity()
        case {"task_name": "classification", "num_classes": num_classes}:
            task_head = nn.Linear(emb_dim, num_classes)
        case _:
            raise ValueError()

    return pooling, task_head


def create_model(config):
    embedding = create_embedding(config)
    pooling, task_head = create_task_head(config)
    return TrajTransformer(config["encoder_config"], embedding, pooling, task_head)
