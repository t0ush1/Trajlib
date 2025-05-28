import random
import numpy as np
import torch
from torch.utils.data import Dataset

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


# TODO 加上时间戳
class SimilarityDataset(Dataset):
    def __init__(self, traj_data: TrajData, variant, limit=10000):
        self.original = traj_data.original
        if variant == "cropp":
            self.variant = traj_data.cropped
        elif variant == "distort":
            self.variant = traj_data.distorted
        self.limit = limit
        self.data_list = self.__read_data__()

    def __read_data__(self):
        raise NotImplementedError()

    def __len__(self):
        return len(self.data_list[0])

    def __getitem__(self, index):
        return [torch.tensor(x[index]) for x in self.data_list]


class SimilarityCDDDataset(SimilarityDataset):
    def __read_data__(self):
        x_ori_loc, y_ori_loc = [], []
        x_var_loc, y_var_loc = [], []
        for i in range(len(self.original)):
            for j in range(i, len(self.original)):
                if len(x_ori_loc) >= self.limit:
                    break
                x_ori_loc.append(self.original[i].locations)
                y_ori_loc.append(self.original[j].locations)
                x_var_loc.append(self.variant[i].locations)
                y_var_loc.append(self.variant[j].locations)
            if len(x_ori_loc) >= self.limit:
                break
        return x_ori_loc, y_ori_loc, x_var_loc, y_var_loc


class SimilarityKNNDataset(SimilarityDataset):
    def __read_data__(self):
        x_ori_loc = []
        x_var_loc = []
        for i in range(len(self.original)):
            if len(x_ori_loc) >= self.limit:
                return
            x_ori_loc.append(self.original[i].locations)
            x_var_loc.append(self.variant[i].locations)
        return x_ori_loc, x_var_loc


# TODO 时间戳怎么掩码？
class MLMDataset(Dataset):
    def __init__(self, traj_data: TrajData, ratio=0.1, num_var=5):
        self.trajectories = traj_data.original
        self.ratio = ratio
        self.num_var = num_var
        self.__read_data__()

    def __read_data__(self):
        self.x_loc = []
        self.y_loc = []
        for traj in self.trajectories:
            for i in range(self.num_var):
                locs = traj.locations.copy()
                for j in range(len(locs)):
                    if random.random() < self.ratio:
                        locs[j] = SPECIAL_TOKENS["mask"]
                self.x_loc.append(locs)
                self.y_loc.append(traj.locations)

    def __len__(self):
        return len(self.x_loc)

    def __getitem__(self, index):
        return torch.tensor(self.x_loc[index]), torch.tensor(self.y_loc[index])
