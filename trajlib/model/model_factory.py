import os
import pickle
import torch
import torch.nn as nn

from trajlib.model.transformer import Transformer
from trajlib.model.positional_encoding import PositionalEncoding


class TrajTransformer(nn.Module):
    def __init__(self, model_config, embedding, task_head):
        super(TrajTransformer, self).__init__()
        self.embedding = embedding
        self.positional_encoding = PositionalEncoding(model_config["d_model"], 10)
        self.encoder = Transformer(model_config)
        self.task_head = task_head

    def forward(self, x):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.encoder(x)
        x = self.task_head(x[:, -1, :])
        return x.unsqueeze(1)


def create_embedding(config):
    data_config = config["data_config"]
    model_config = config["model_config"]

    if data_config["data_form"] == "gps":
        return nn.Linear(2, model_config["d_model"])

    elif data_config["data_form"] == "grid":
        if model_config["embedding"] == "normal":
            return nn.Embedding(model_config["vocab_size"], model_config["d_model"])
        elif model_config["embedding"] == "node2vec":
            with open(model_config["embs_path"], "rb") as f:
                embeddings = pickle.load(f)
            return nn.Embedding.from_pretrained(embeddings=embeddings, freeze=True)
        elif model_config["embedding"] == "gat":
            pass  # TODO 独立训练
        elif model_config["embedding"] == "gcn":
            pass  #


def create_task_head(config):
    data_config = config["data_config"]
    model_config = config["model_config"]

    if data_config["data_form"] == "gps":
        return nn.Linear(model_config["d_model"], 2)
    elif data_config["data_form"] == "grid":
        return nn.Linear(model_config["d_model"], model_config["vocab_size"])


def create_model(config):
    embedding = create_embedding(config)
    task_head = create_task_head(config)
    return TrajTransformer(config["model_config"], embedding, task_head)
