from torch.utils.data import random_split

from trajlib.data.data import TrajData
from trajlib.dataset.datasets import (
    PredictionDataset,
    SimilarityCDDDataset,
    SimilarityKNNDataset,
    MLMDataset,
    AutoregressiveDataset,
)


def create_dataset(config, data: TrajData):
    task_config = config["task_config"]

    match task_config:
        case {"task_name": "prediction", "input_len": input_len, "output_len": output_len}:
            dataset = PredictionDataset(data, input_len, output_len)
        case {"task_name": "similarity", "sub-task": "CDD", "variant": variant}:
            dataset = SimilarityCDDDataset(data, variant)
        case {"task_name": "similarity", "sub-task": "kNN", "variant": variant}:
            dataset = SimilarityKNNDataset(data, variant)
        case {"task_name": "filling", "sub-task": "mlm"}:
            dataset = MLMDataset(data)
        case {"task_name": "filling", "sub-task": "autoregressive"}:
            dataset = AutoregressiveDataset(data)
        case _:
            raise ValueError()

    val_size = int(task_config["dataset_prop"][1] * len(dataset))
    test_size = int(task_config["dataset_prop"][2] * len(dataset))
    train_size = len(dataset) - val_size - test_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    return train_dataset, val_dataset, test_dataset
