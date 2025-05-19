from torch.utils.data import random_split

from trajlib.data.data import TrajData
from trajlib.dataset.datasets import PredictionDataset, SimilarityCDDDataset, SimilarityKNNDataset


def create_dataset(config, data: TrajData):
    task_config = config["task_config"]

    if task_config["task_name"] == "prediction":
        dataset = PredictionDataset(data, task_config["input_len"], task_config["output_len"])

    elif task_config["task_name"] == "similarity":
        if task_config["sub-task"] == "CDD":
            dataset = SimilarityCDDDataset(data)
        elif task_config["sub-task"] == "kNN":
            dataset = SimilarityKNNDataset(data)

    val_size = int(task_config["dataset_prop"][1] * len(dataset))
    test_size = int(task_config["dataset_prop"][2] * len(dataset))
    train_size = len(dataset) - val_size - test_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    return train_dataset, val_dataset, test_dataset
