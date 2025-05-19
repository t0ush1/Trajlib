import numpy as np
import torch
from torch.utils.data import Dataset

from trajlib.data.data import TrajData


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


class SimilarityCDDDataset(Dataset):
    def __init__(self, traj_data: TrajData, limit=10000):
        self.original = traj_data.original
        self.variant = traj_data.cropped
        self.limit = limit
        self.__read_data__()

    def __read_data__(self):
        self.x_ori_loc, self.y_ori_loc = [], []
        self.x_var_loc, self.y_var_loc = [], []
        for i in range(len(self.original)):
            for j in range(i, len(self.original)):
                if len(self.x_ori_loc) >= self.limit:
                    return
                self.x_ori_loc.append(self.original[i].locations)
                self.y_ori_loc.append(self.original[j].locations)
                self.x_var_loc.append(self.variant[i].locations)
                self.y_var_loc.append(self.variant[j].locations)

    def __len__(self):
        return len(self.x_ori_loc)

    def __getitem__(self, index):
        return (
            torch.tensor(self.x_ori_loc[index]),
            torch.tensor(self.y_ori_loc[index]),
            torch.tensor(self.x_var_loc[index]),
            torch.tensor(self.y_var_loc[index]),
        )


class SimilarityKNNDataset(Dataset):
    def __init__(self, traj_data: TrajData, limit=10000):
        self.original = traj_data.original
        self.variant = traj_data.cropped
        self.limit = limit
        self.__read_data__()

    def __read_data__(self):
        self.x_ori_loc = []
        self.x_var_loc = []
        for i in range(len(self.original)):
            if len(self.x_ori_loc) >= self.limit:
                return
            self.x_ori_loc.append(self.original[i].locations)
            self.x_var_loc.append(self.variant[i].locations)

    def __len__(self):
        return len(self.x_ori_loc)

    def __getitem__(self, index):
        return (
            torch.tensor(self.x_ori_loc[index]),
            torch.tensor(self.x_var_loc[index]),
        )
