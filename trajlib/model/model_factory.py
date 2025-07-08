import os
import pickle
import torch
import torch.nn as nn
from torch_geometric.data import Data as GeoData

from trajlib.data.data import SpecialToken
from trajlib.model.embedding.embedding_trainer import EmbeddingTrainer
from trajlib.model.embedding.gnn import GAETrainer, GNNWithEmbedding
from trajlib.model.embedding.node2vec import Node2VecTrainer
from trajlib.model.transformer import Transformer
from trajlib.model.positional_encoding import PositionalEncoding


class GPSEmbedding(nn.Module):
    def __init__(self, emb_config, emb_dim):
        super(GPSEmbedding, self).__init__()
        self.embedding = self.__create_embedding__(emb_config, emb_dim)

    def __create_embedding__(self, emb_config, emb_dim):
        match emb_config:
            case {"emb_name": "linear"}:
                return nn.Linear(2, emb_dim)
            case {"emb_name": "non-linear"}:
                return nn.Sequential(nn.Linear(2, emb_dim), nn.ReLU(), nn.Linear(emb_dim, emb_dim))
            case _:
                raise ValueError()

    def forward(self, coordinates, **kwargs):
        return self.embedding(coordinates)


class GraphEmbedding(nn.Module):
    def __init__(self, emb_config, emb_dim):
        super(GraphEmbedding, self).__init__()
        self.is_gnn, self.embedding = self.__create_embedding__(emb_config, emb_dim)
        self.unk_emb = nn.Parameter(torch.randn(emb_dim))

    def __create_embedding__(self, emb_config, emb_dim):
        match emb_config:
            case {"pre-trained": True, "embs_path": embs_path}:
                with open(embs_path, "rb") as f:
                    embs = pickle.load(f)
                return False, nn.Embedding.from_pretrained(embeddings=embs, freeze=True)
            case {"emb_name": "embedding", "vocab_size": vocab_size}:
                return False, nn.Embedding(vocab_size, emb_dim)
            case {"emb_name": "gat" | "gcn" as emb_name, "vocab_size": vocab_size}:
                return True, GNNWithEmbedding(emb_name, vocab_size, emb_dim)
            case _:
                raise ValueError()

    def _embed(self, input_ids, geo_data: GeoData, special_pos, **kwargs):
        unk_pos = input_ids == SpecialToken.UNK
        x = torch.where(special_pos | unk_pos, 0, input_ids)
        x = self.embedding(geo_data.x, geo_data.edge_index)[x] if self.is_gnn else self.embedding(x)
        x[unk_pos] = self.unk_emb
        return x


class GridEmbedding(GraphEmbedding):
    def forward(self, grid_ids, grid_geo_data, **kwargs):
        return super()._embed(input_ids=grid_ids, geo_data=grid_geo_data, **kwargs)


class RoadnetEmbedding(GraphEmbedding):
    def forward(self, road_ids, road_geo_data, **kwargs):
        return super()._embed(input_ids=road_ids, geo_data=road_geo_data, **kwargs)


class TrajEmbedding(nn.Module):
    def __init__(self, embedding_config):
        super(TrajEmbedding, self).__init__()
        emb_dim = embedding_config["emb_dim"]
        self.tokens = embedding_config["tokens"]
        mapper = {
            "gps": GPSEmbedding,
            "grid": GridEmbedding,
            "roadnet": RoadnetEmbedding,
        }
        self.embeddings = nn.ModuleDict(
            {token: mapper[token](embedding_config[token], emb_dim) for token in self.tokens}
        )
        # self.embeddings = nn.ModuleDict(
        #     {
        #         "gps": GPSEmbedding(embedding_config["gps"], emb_dim),
        #         "grid": GridEmbedding(embedding_config["grid"], emb_dim),
        #         "roadnet": RoadnetEmbedding(embedding_config["roadnet"], emb_dim),
        #     }
        # )
        self.pad_emb = nn.Parameter(torch.zeros(emb_dim), requires_grad=False)
        self.mask_emb = nn.Parameter(torch.randn(emb_dim))
        # TODO 不同模态 norm 后再相加
        # self.norm = nn.LayerNorm(emb_dim)

    def forward(self, road_ids, **kwargs):
        pad_pos = road_ids == SpecialToken.PAD
        mask_pos = road_ids == SpecialToken.MASK

        x = 0
        for token in self.tokens:
            x = x + self.embeddings[token](special_pos=pad_pos | mask_pos, road_ids=road_ids, **kwargs)

        x[pad_pos] = self.pad_emb
        x[mask_pos] = self.mask_emb

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
            print(f"[Model Factory] Pre-train {emb_name} embedding for {token}")
            trainer = mapper[emb_name](emb_name, emb_dim, embs_path, data)
            trainer.train()


class MultiTokenTaskHead(nn.Module):
    def __init__(self, tokens, emb_dim, grid_vocab_size, road_vocab_size):
        super(MultiTokenTaskHead, self).__init__()
        self.tokens = tokens
        head_dims = {
            "gps": 2,
            "grid": grid_vocab_size,
            "roadnet": road_vocab_size,
        }
        self.task_heads = nn.ModuleDict({token: nn.Linear(emb_dim, head_dims[token]) for token in self.tokens})
        # self.task_heads = nn.ModuleDict(
        #     {
        #         "gps": nn.Linear(emb_dim, 2),
        #         "grid": nn.Linear(emb_dim, grid_vocab_size),
        #         "roadnet": nn.Linear(emb_dim, road_vocab_size),
        #     }
        # )

    def forward(self, x):
        return {token: self.task_heads[token](x) for token in self.tokens}


def create_task_head(task_config, embedding_config):
    pooling = task_config["task_name"] != "filling"

    match task_config:
        case {"task_name": "prediction" | "filling", "tokens": tokens}:
            task_head = MultiTokenTaskHead(
                tokens,
                embedding_config["emb_dim"],
                embedding_config["grid"]["vocab_size"],
                embedding_config["roadnet"]["vocab_size"],
            )
        case {"task_name": "similarity"}:
            task_head = nn.Identity()
        case {"task_name": "classification", "num_classes": num_classes}:
            task_head = nn.Linear(embedding_config["emb_dim"], num_classes)
        case _:
            raise ValueError()

    return pooling, task_head


class TrajModel(nn.Module):
    def __init__(self, embedding, pe, encoder, pooling, task_head):
        super(TrajModel, self).__init__()
        self.embedding = embedding
        self.pe = pe
        self.encoder = encoder
        self.pooling = pooling
        self.task_head = task_head

    def forward(self, mask=None, **kwargs):
        x = self.embedding(**kwargs)
        x = self.pe(x)
        x = self.encoder(x, mask)
        if self.pooling:
            x = x.mean(dim=1)
            # x = x[:, -1, :]
        x = self.task_head(x)
        return x


def create_model(config):
    task_config = config["task_config"]
    embedding_config = config["embedding_config"]
    encoder_config = config["encoder_config"]

    embedding = TrajEmbedding(embedding_config)
    pe = PositionalEncoding(encoder_config["d_model"])
    encoder = Transformer(encoder_config)
    pooling, task_head = create_task_head(task_config, embedding_config)

    return TrajModel(embedding, pe, encoder, pooling, task_head)
