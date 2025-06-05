import os
import pickle
import torch
import torch.nn as nn
from torch_geometric.data import Data as GeoData

from trajlib.model.transformer import Transformer
from trajlib.model.lstm import LSTM
from trajlib.model.cnn import CNN
from trajlib.model.mlp import MLP
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
    if config["encoder_config"]["encoder_name"] == "transformer":
        return TrajTransformer(config["encoder_config"], embedding, task_head)
    elif config["encoder_config"]["encoder_name"] == "lstm":
        return TrajLSTM(config["encoder_config"], embedding, task_head)
    elif config["encoder_config"]["encoder_name"] == "cnn":
        return TrajCNN(config["encoder_config"], embedding, task_head)
    elif config["encoder_config"]["encoder_name"] == "mlp":
        return TrajMLP(config["encoder_config"], embedding, task_head)