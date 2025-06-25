import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch_geometric.data import Data as GeoData
from accelerate import Accelerator
from torch.nn.utils.rnn import pad_sequence

from trajlib.data.data import SpecialToken
from trajlib.runner.utils.early_stopping import EarlyStopping


def pad_trajectory(traj_batch):
    coords_batch, grids_batch, roads_batch, times_batch = zip(*traj_batch)
    max_len = max(len(c) for c in coords_batch)
    coords = torch.zeros(len(traj_batch), max_len, 2)
    for i, c in enumerate(coords_batch):
        coords[i, : len(c)] = c
    grids = pad_sequence(grids_batch, batch_first=True, padding_value=SpecialToken.PAD)
    roads = pad_sequence(roads_batch, batch_first=True, padding_value=SpecialToken.PAD)
    times = pad_sequence(times_batch, batch_first=True, padding_value=0)
    return coords, grids, roads, times


class BaseTrainer:
    def __init__(
        self,
        trainer_config,
        accelerator: Accelerator,
        model: nn.Module,
        dataset,
        grid_geo_data: GeoData,
        road_geo_data: GeoData,
        collate_fn=None,
    ):
        self.trainer_config = trainer_config
        self.accelerator = accelerator
        self.model = model
        self.grid_geo_data = grid_geo_data
        self.road_geo_data = road_geo_data

        train_dataset, val_dataset, test_dataset = dataset
        batch_size = trainer_config["batch_size"]
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        if trainer_config["optimizer"] == "adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=trainer_config["learning_rate"])

        if trainer_config["lr_scheduler"] == "step_lr":
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)

        if trainer_config["loss_function"] == "cross_entropy":
            self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        elif trainer_config["loss_function"] == "mse":
            self.criterion = nn.MSELoss()

        self.early_stopping = EarlyStopping(patience=7, delta=0)

        (
            self.model,
            self.train_loader,
            self.val_loader,
            self.test_loader,
            self.optimizer,
            self.scheduler,
        ) = self.accelerator.prepare(
            self.model,
            self.train_loader,
            self.val_loader,
            self.test_loader,
            self.optimizer,
            self.scheduler,
        )

        self.road_geo_data = self.road_geo_data.to(self.accelerator.device)
        self.grid_geo_data = self.grid_geo_data.to(self.accelerator.device)

    def _call_model(self, traj, **kwargs):
        coords, grids, roads, times = traj
        device = self.accelerator.device

        coords = coords.to(device)
        grids = grids.to(device)
        roads = roads.to(device)
        times = times.to(device)
        kwargs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in kwargs.items()}

        output = self.model(
            coordinates=coords,
            grid_ids=grids,
            road_ids=roads,
            timestamps=times,
            grid_geo_data=self.grid_geo_data,
            road_geo_data=self.road_geo_data,
            **kwargs
        )
        return output

    def _get_tokens(self, traj, token):
        coords, grids, roads, times = traj
        tokens = {
            "gps": coords,
            "grid": grids,
            "roadnet": roads,
            "timestamp": times,
        }[token]
        return tokens.to(self.accelerator.device)

    def train(self, epoch):
        raise NotImplementedError()

    def validate(self, epoch):
        raise NotImplementedError()

    def test(self, epoch):
        raise NotImplementedError()
