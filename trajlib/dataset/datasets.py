import random
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import Counter

from trajlib.data.data import TrajData, SPECIAL_TOKENS


class PredictionDataset(Dataset):
    def __init__(self, traj_data: TrajData, input_len, output_len):
        self.trajectories = traj_data.original
        self.input_len = input_len
        self.output_len = output_len
        self.__read_data__()

    def __read_data__(self):
        x_loc, x_ts = [], []
        y_loc, y_ts = [], []
        for traj in self.trajectories:
            if len(traj) < self.input_len + self.output_len:
                continue
            for i in range(len(traj) - self.input_len - self.output_len + 1):
                x_loc.append(traj.locations[i : i + self.input_len])
                x_ts.append(traj.timestamps[i : i + self.input_len])
                y_loc.append(traj.locations[i + self.input_len : i + self.input_len + self.output_len])
                y_ts.append(traj.timestamps[i + self.input_len : i + self.input_len + self.output_len])
        self.x_loc = torch.tensor(x_loc)
        self.x_ts = torch.tensor(x_ts)
        self.y_loc = torch.tensor(y_loc)
        self.y_ts = torch.tensor(y_ts)

    def __len__(self):
        return len(self.x_loc)

    def __getitem__(self, index):
        return self.x_loc[index], self.x_ts[index], self.y_loc[index], self.y_ts[index]


class SimilarityDataset(Dataset):
    def __init__(self, traj_data: TrajData, variant, limit=10000):
        self.original = traj_data.original
        self.variant = traj_data.original if variant == "original" else traj_data.variants[variant]
        self.limit = limit
        self.data_list = self.__read_data__()

    def __read_data__(self):
        raise NotImplementedError()

    def __len__(self):
        return len(self.data_list[0])

    def __getitem__(self, index):
        return [torch.tensor(x[index]) for x in self.data_list]


class SimilarityMSSDataset(SimilarityDataset):
    def __read_data__(self):
        x_even_loc, x_odd_loc = [], []
        for i in range(len(self.variant)):
            if len(x_even_loc) >= self.limit:
                break
            x_even_loc.append(self.variant[i].locations[0::2])
            x_odd_loc.append(self.variant[i].locations[1::2])
        return x_even_loc, x_odd_loc


class SimilarityCDDDataset(SimilarityDataset):
    def __read_data__(self):
        x_ori_loc, y_ori_loc = [], []
        x_var_loc, y_var_loc = [], []
        for i in range(len(self.original)):
            if len(x_ori_loc) >= self.limit:
                break
            for j in range(i, len(self.original)):
                if len(x_ori_loc) >= self.limit:
                    break
                x_ori_loc.append(self.original[i].locations)
                y_ori_loc.append(self.original[j].locations)
                x_var_loc.append(self.variant[i].locations)
                y_var_loc.append(self.variant[j].locations)
        return x_ori_loc, y_ori_loc, x_var_loc, y_var_loc


class SimilarityKNNDataset(SimilarityDataset):
    def __read_data__(self):
        x_ori_loc = []
        x_var_loc = []
        for i in range(len(self.original)):
            if len(x_ori_loc) >= self.limit:
                break
            x_ori_loc.append(self.original[i].locations)
            x_var_loc.append(self.variant[i].locations)
        return x_ori_loc, x_var_loc


class FillingDataset(Dataset):
    def __init__(self, traj_data: TrajData):
        self.trajectories = traj_data.original
        self.inputs, self.labels = self.__read_data__()

    def __read_data__(self):
        raise NotImplementedError()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return torch.tensor(self.inputs[index]), torch.tensor(self.labels[index])


class MLMDataset(FillingDataset):
    def __init__(self, traj_data: TrajData, ratio=0.15, num_var=5):
        self.ratio = ratio
        self.num_var = num_var
        super().__init__(traj_data)

    def __read_data__(self):
        inputs, labels = [], []
        for traj in self.trajectories:
            for i in range(self.num_var):
                input = traj.locations.copy()
                for j in range(len(input)):
                    if random.random() < self.ratio:
                        input[j] = SPECIAL_TOKENS["mask"]
                inputs.append(input)
                labels.append(traj.locations)
        return inputs, labels


class AutoregressiveDataset(FillingDataset):
    def __read_data__(self):
        inputs, labels = [], []
        for traj in self.trajectories:
            inputs.append(traj.locations[:-1])
            labels.append(traj.locations[1:])
        return inputs, labels


class ClassificationDataset(Dataset):
    def __init__(self, traj_data: TrajData, class_attr):
        self.trajectories = traj_data.original
        self.class_attr = class_attr
        self.__read_data__()

    def __read_data__(self):
        class_values = [traj.attributes[self.class_attr] for traj in self.trajectories]
        counter = Counter(class_values)
        print(counter)
        sorted_values = sorted(counter.keys(), key=lambda c: -counter[c])
        self.class_mapping = {cls: idx for idx, cls in enumerate(sorted_values)}

        self.inputs, self.labels = [], []
        for traj, val in zip(self.trajectories, class_values):
            self.inputs.append(traj.locations)
            self.labels.append(self.class_mapping[val])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return torch.tensor(self.inputs[index]), torch.tensor(self.labels[index])
