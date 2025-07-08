import random
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import Counter

from trajlib.data.data import Trajectory, TrajData


class PredictionDataset(Dataset):
    def __init__(self, traj_data: TrajData, input_len, output_len):
        self.trajectories = traj_data.original
        self.input_len = input_len
        self.output_len = output_len
        self.x_trajs: list[Trajectory] = []
        self.y_trajs: list[Trajectory] = []

        for traj in self.trajectories:
            if len(traj) < self.input_len + self.output_len:
                continue
            for i in range(len(traj) - self.input_len - self.output_len + 1):
                self.x_trajs.append(traj[i : i + self.input_len])
                self.y_trajs.append(traj[i + self.input_len : i + self.input_len + self.output_len])

    def __len__(self):
        return len(self.x_trajs)

    def __getitem__(self, index):
        return self.x_trajs[index].to_tensor(), self.y_trajs[index].to_tensor()


class SimilarityDataset(Dataset):
    def __init__(self, traj_data: TrajData, variant, limit=10000):
        self.original = traj_data.original
        self.variant = traj_data.variants[variant]
        self.limit = limit
        self.traj_lists: tuple[list[Trajectory]] = self.__read_data__()

    def __read_data__(self):
        raise NotImplementedError()

    def __len__(self):
        return len(self.traj_lists[0])

    def __getitem__(self, index):
        return [x[index].to_tensor() for x in self.traj_lists]


class SimilarityMSSDataset(SimilarityDataset):
    def __read_data__(self):
        x_even, x_odd = [], []
        for i in range(len(self.variant)):
            if len(x_even) >= self.limit:
                break
            x_even.append(self.variant[i][0::2])
            x_odd.append(self.variant[i][1::2])
        return x_even, x_odd


class SimilarityCDDDataset(SimilarityDataset):
    def __read_data__(self):
        x_ori, y_ori = [], []
        x_var, y_var = [], []
        for i in range(len(self.original)):
            if len(x_ori) >= self.limit:
                break
            for j in range(i, len(self.original)):
                if len(x_ori) >= self.limit:
                    break
                x_ori.append(self.original[i])
                y_ori.append(self.original[j])
                x_var.append(self.variant[i])
                y_var.append(self.variant[j])
        return x_ori, y_ori, x_var, y_var


class SimilarityKNNDataset(SimilarityDataset):
    def __read_data__(self):
        x_ori = []
        x_var = []
        for i in range(len(self.original)):
            if len(x_ori) >= self.limit:
                break
            x_ori.append(self.original[i])
            x_var.append(self.variant[i])
        return x_ori, x_var


class FillingDataset(Dataset):
    def __init__(self, traj_data: TrajData):
        self.trajectories = traj_data.original

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, index):
        return self.trajectories[index].to_tensor()


class ClassificationDataset(Dataset):
    def __init__(self, traj_data: TrajData, class_attr):
        self.trajectories = traj_data.original
        self.class_attr = class_attr
        self.__read_data__()

    def __read_data__(self):
        class_values = [traj.attributes[self.class_attr] for traj in self.trajectories]
        counter = Counter(class_values)
        sorted_values = sorted(counter.keys(), key=lambda c: -counter[c])
        self.class_mapping = {cls: idx for idx, cls in enumerate(sorted_values)}
        self.labels = [self.class_mapping[val] for val in class_values]
        print("[Dataset] classification counter:", counter)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.trajectories[index].to_tensor(), torch.tensor(self.labels[index])
