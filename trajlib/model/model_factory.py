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

        x = self.embedding(x) if geo_data is None else self.embedding(geo_data.x, geo_data.edge_index)[x]

        for idx, mask in masks.items():
            x[mask] = self.special_embs[idx]  # TODO 广播？

        return x


class TrajTransformer(nn.Module):
    def __init__(self, encoder_config, embedding, pooling, task_head):
        super(TrajTransformer, self).__init__()
        self.embedding = embedding
        self.positional_encoding = PositionalEncoding(encoder_config["d_model"])
        self.encoder = Transformer(encoder_config)
        self.pooling = pooling
        self.task_head = task_head

    def forward(self, x, mask=None, geo_data: GeoData = None):
        x = self.embedding(x, geo_data)
        x = self.positional_encoding(x)
        x = self.encoder(x, mask)
        if self.pooling: # TODO 加CLS，取最后一个
            x = x.mean(dim=1)
        x = self.task_head(x)
        return x


class TrajLSTM(nn.Module):
    def __init__(self, encoder_config, embedding, task_head):
        super(TrajLSTM, self).__init__()
        self.embedding = embedding
        self.encoder = LSTM(encoder_config)
        self.task_head = task_head

        # 便于外部查询最终序列输出维度
        self.output_dim = encoder_config["hidden_size"] * (2 if encoder_config.get("bidirectional", False) else 1)

    def forward(self, x, mask=None, geo_data: GeoData = None):
        # -------- Embedding --------
        if geo_data is not None:
            # 图嵌入：embedding(节点特征, 边索引) → (N_nodes, d_model)
            embedding = self.embedding(geo_data.x, geo_data.edge_index)
            x = embedding[x]          # (B, T, d_model)
        else:
            # 普通 token/ID 或已有特征
            x = self.embedding(x)      # (B, T, d_model)

        x = self.encoder(x, mask)      # (B, T, hidden_size * directions)
        # -------- Pool & Head --------
        x = x[:, -1, :]
        # x = x.mean(dim=1)              # 简单时序平均池化 (B, hidden*)
        x = self.task_head(x)          # 输出维度由具体任务决定
        return x


class TrajCNN(nn.Module):
    def __init__(self, encoder_config, embedding, task_head):
        super(TrajCNN, self).__init__()
        self.embedding = embedding
        self.positional_encoding = PositionalEncoding(encoder_config["d_model"])
        self.encoder = CNN(encoder_config)
        self.task_head = task_head

    def forward(self, x, mask=None, geo_data: GeoData = None):
        # 如果有 GeoData，使用该数据来嵌入
        if geo_data is not None:
            embedding = self.embedding(geo_data.x, geo_data.edge_index)
            x = embedding[x]
        else:
            x = self.embedding(x)
        
        x = self.positional_encoding(x)  # 添加位置编码
        # 使用 CNNEncoder 提取时序特征
        x = self.encoder(x, mask)
        # 对所有时间步的特征进行平均（可以替换为其他聚合方式）
        x = x.mean(dim=1)
        # 通过任务头进行预测
        x = self.task_head(x)
        return x


class TrajMLP(nn.Module):
    def __init__(self, encoder_config, embedding, task_head):
        super(TrajMLP, self).__init__()
        self.embedding = embedding
        self.positional_encoding = PositionalEncoding(encoder_config["d_model"])
        self.encoder = MLP(encoder_config)
        self.task_head = task_head

    def forward(self, x, mask=None, geo_data: GeoData = None):
        # 如果有 GeoData，使用该数据来嵌入
        if geo_data is not None:
            embedding = self.embedding(geo_data.x, geo_data.edge_index)
            x = embedding[x]
        else:
            x = self.embedding(x)
        
        x = self.positional_encoding(x)  # 添加位置编码
        # 使用 MLPEncoder 提取时序特征
        x = self.encoder(x, mask)
        # 对所有时间步的特征进行平均（可以替换为其他聚合方式）
        x = x.mean(dim=1)
        # 通过任务头进行预测
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
            embedding = GNNWithEmbedding(vocab_size, emb_dim)
        case _:
            raise ValueError()

    return TrajEmbedding(embedding, emb_dim, tokenized=data_config["data_form"] != "gps")


def create_task_head(config):
    task_config = config["task_config"]
    data_config = config["data_config"]
    embedding_config = config["embedding_config"]
    emb_dim = embedding_config["emb_dim"]

    match (task_config, data_config):
        case ({"task_name": "prediction"}, {"data_form": "gps"}):
            return True, nn.Linear(emb_dim, 2)
        case ({"task_name": "prediction"}, {"data_form": "grid", "vocab_size": vocab_size}):
            return True, nn.Linear(emb_dim, vocab_size)
        case ({"task_name": "similarity"}, _):
            return True, nn.Identity()
        case ({"task_name": "filling"}, {"data_form": "grid", "vocab_size": vocab_size}):
            return False, nn.Linear(emb_dim, vocab_size)
        case _:
            raise ValueError()


def create_model(config):
    embedding = create_embedding(config)
    pooling, task_head = create_task_head(config)
    if config["encoder_config"]["encoder_name"] == "transformer":
        return TrajTransformer(config["encoder_config"], embedding, task_head)
    elif config["encoder_config"]["encoder_name"] == "lstm":
        return TrajLSTM(config["encoder_config"], embedding, task_head)
    elif config["encoder_config"]["encoder_name"] == "cnn":
        return TrajCNN(config["encoder_config"], embedding, task_head)
    elif config["encoder_config"]["encoder_name"] == "mlp":
        return TrajMLP(config["encoder_config"], embedding, task_head)
    # return TrajTransformer(config["encoder_config"], embedding, pooling, task_head)
