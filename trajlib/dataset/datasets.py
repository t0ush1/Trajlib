import numpy as np
import torch
from torch.utils.data import Dataset

from trajlib.data.utils.data_definition import TrajectoryData


class TrajectoryDataset(Dataset):
    def __init__(self, trajectory_data: TrajectoryData, input_len, output_len):
        self.trajectory_data = trajectory_data
        self.trajectories = self.trajectory_data.cal_all_trajs(attrs=["point_id", "timestamp", "lon", "lat"])
        self.traj_ids = list(self.trajectories.keys())
        self.input_len = input_len
        self.output_len = output_len
        self.__read_data__()

    def __read_data__(self):
        x_data = []
        x_mark_data = []
        y_data = []
        y_mark_data = []
        for index in self.traj_ids:
            traj_id = self.traj_ids[index]
            traj = self.trajectories[traj_id]
            points = traj["point_id"]
            timestamps = traj["timestamp"]
            lons = traj["lon"]
            lats = traj["lat"]

            # 初始化存储输入和输出序列的列表
            x_list = []
            x_mark_list = []
            y_list = []
            y_mark_list = []

            # 滑动窗口生成输入输出序列
            for i in range(len(points) - self.input_len - self.output_len + 1):
                x = []
                x_mark = []
                y = []
                y_mark = []
                # 生成输入序列 x 及其时间戳 x_mark
                for j in range(self.input_len):
                    x.append([lons[i + j], lats[i + j]])
                    x_mark.append(timestamps[i + j])
                # 生成目标序列 y 及其时间戳 y_mark
                for k in range(self.input_len, self.input_len + self.output_len):
                    y.append([lons[i + k], lats[i + k]])
                    y_mark.append(timestamps[i + k])
                x_list.append(x)
                x_mark_list.append(x_mark)
                y_list.append(y)
                y_mark_list.append(y_mark)

            # 将列表转换为张量
            x_tensor = torch.tensor(x_list, dtype=torch.float32)
            x_mark_tensor = torch.tensor(x_mark_list, dtype=torch.float32)
            y_tensor = torch.tensor(y_list, dtype=torch.float32)
            y_mark_tensor = torch.tensor(y_mark_list, dtype=torch.float32)

            x_data.append(x_tensor)
            x_mark_data.append(x_mark_tensor)
            y_data.append(y_tensor)
            y_mark_data.append(y_mark_tensor)

        self.x_data = torch.concat(x_data)
        self.x_mark_data = torch.concat(x_mark_data)
        self.y_data = torch.concat(y_data)
        self.y_mark_data = torch.concat(y_mark_data)

    def __len__(self):
        """
        返回轨迹数据集中轨迹的数量
        """
        return len(self.x_data)

    def __getitem__(self, index):
        """
        根据索引获取轨迹数据
        :param index: 轨迹的索引
        :return: 对应索引的轨迹数据，包括输入序列 x 和 x_mark 以及目标序列 y 和 y_mark
        """

        return (
            self.x_data[index],
            self.x_mark_data[index],
            self.y_data[index],
            self.y_mark_data[index],
        )


from trajdl.tokenizers import T2VECTokenizer


class GridDataset(Dataset):
    def __init__(self, trajs, tokenizer: T2VECTokenizer, input_len, output_len):
        self.trajs = trajs
        self.tokenizer = tokenizer
        self.input_len = input_len
        self.output_len = output_len
        self.__to_grid__()
        self.__read_data__()

    def __to_grid__(self):
        self.grid_vectors = []
        # for traj in self.trajs:
        #     grid_vector = self.tokenizer.tokenize_traj(traj)  # TODO return as numpy
        #     if len(grid_vector) < self.input_len + self.output_len:
        #         continue
        #     self.grid_vectors.append(grid_vector)
        for traj in self.trajs:
            grid_vector = []
            for i in range(len(traj)):
                grid_vector.append(self.tokenizer.tokenize_traj(traj[i : i + 1])[0])
            self.grid_vectors.append(grid_vector)

    def __read_data__(self):
        x_data = []
        y_data = []
        for grid_vector in self.grid_vectors:
            x_list = []
            y_list = []
            for i in range(len(grid_vector) - self.input_len - self.output_len + 1):
                x = []
                y = []
                for j in range(self.input_len):
                    x.append(grid_vector[i + j])
                for k in range(self.input_len, self.input_len + self.output_len):
                    y.append(grid_vector[i + k])
                x_list.append(x)
                y_list.append(y)

            x_tensor = torch.tensor(x_list, dtype=torch.int)
            y_tensor = torch.tensor(y_list, dtype=torch.int)

            x_data.append(x_tensor)
            y_data.append(y_tensor)

        self.x_data = torch.concat(x_data)
        self.y_data = torch.concat(y_data)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]


from trajlib.data.data import TrajData


class PredictionDataset(Dataset):
    def __init__(self, traj_data: TrajData, input_len, output_len):
        self.trajectories = traj_data.trajectories
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
        self.x_locations = torch.tensor(x_loc)
        self.x_timestamps = torch.tensor(x_ts)
        self.y_locations = torch.tensor(y_loc)
        self.y_timestamps = torch.tensor(y_ts)

    def __len__(self):
        return len(self.x_locations)

    def __getitem__(self, index):
        return self.x_locations[index], self.x_timestamps[index], self.y_locations[index], self.y_timestamps[index]
