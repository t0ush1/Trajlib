import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.data import Data as GeoData
from accelerate import Accelerator
from tqdm import tqdm

from trajlib.runner.utils.early_stopping import EarlyStopping


class BaseTrainer:
    def __init__(self, trainer_config, accelerator: Accelerator, model: nn.Module, dataset, geo_data: GeoData):
        self.trainer_config = trainer_config
        self.accelerator = accelerator
        self.model = model
        self.geo_data = geo_data

        train_dataset, val_dataset, test_dataset = dataset
        self.train_loader = DataLoader(train_dataset, batch_size=trainer_config["batch_size"], shuffle=False)
        self.val_loader = DataLoader(val_dataset, batch_size=trainer_config["batch_size"], shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=trainer_config["batch_size"], shuffle=False)

        if trainer_config["optimizer"] == "adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=trainer_config["learning_rate"])

        if trainer_config["lr_scheduler"] == "step_lr":
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)

        if trainer_config["loss_function"] == "cross_entropy":
            self.criterion = nn.CrossEntropyLoss()

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

    def train(self, epoch):
        raise NotImplementedError()

    def validate(self):
        raise NotImplementedError()

    def test(self):
        raise NotImplementedError()
