from torch.utils.data import random_split

from trajlib.data.data import TrajData
from trajlib.dataset.datasets import PredictionDataset


def create_dataset(config, data: TrajData):
    task_config = config["task_config"]
    dataset_config = config["dataset_config"]

    if task_config["task_name"] == "prediction":
        dataset = PredictionDataset(data, task_config["input_len"], task_config["output_len"])

    val_size = int(dataset_config["val_prop"] * len(dataset))
    test_size = int(dataset_config["test_prop"] * len(dataset))
    train_size = len(dataset) - val_size - test_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    return train_dataset, val_dataset, test_dataset
