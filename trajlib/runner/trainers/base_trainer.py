import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm import tqdm

from trajlib.runner.utils.early_stopping import EarlyStopping


class BaseTrainer:
    def __init__(self, trainer_config, accelerator: Accelerator, model: nn.Module, dataset):
        self.trainer_config = trainer_config
        self.accelerator = accelerator
        self.model = model

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
        self.model.train()
        train_loss = 0
        for x_loc, x_ts, y_loc, y_ts in tqdm(
            self.train_loader, disable=not self.accelerator.is_local_main_process, desc=f"Epoch: {epoch+1}"
        ):
            x_loc = x_loc.to(self.accelerator.device)
            y_loc = y_loc.to(self.accelerator.device)
            output = self.model(x_loc)

            loss = self.criterion(output.squeeze(1), y_loc.squeeze(1).long())
            self.optimizer.zero_grad()
            self.accelerator.backward(loss)
            self.optimizer.step()

            train_loss += loss.item()
        train_loss /= len(self.train_loader)
        return train_loss

    def validate(self):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_loc, x_ts, y_loc, y_ts in self.val_loader:
                x_loc = x_loc.to(self.accelerator.device)
                y_loc = y_loc.to(self.accelerator.device)
                output = self.model(x_loc)

                preds, trues = self.accelerator.gather_for_metrics([output, y_loc])
                loss = self.criterion(preds.squeeze(1), trues.squeeze(1).long())

                val_loss += loss.item()
            val_loss /= len(self.val_loader)
        return val_loss

    def test(self):
        self.model.eval()
        all_preds, all_trues = [], []
        with torch.no_grad():
            for x_loc, x_ts, y_loc, y_ts in self.test_loader:
                x_loc = x_loc.to(self.accelerator.device)
                y_loc = y_loc.to(self.accelerator.device)
                output = self.model(x_loc)

                preds, trues = self.accelerator.gather_for_metrics([output, y_loc])
                all_preds.append(preds)
                all_trues.append(trues)
        all_preds = torch.concat(all_preds, dim=0)
        all_trues = torch.concat(all_trues, dim=0)

        test_loss = F.cross_entropy(all_preds.squeeze(1), all_trues.squeeze(1).long()).item()
        top_k_preds = torch.topk(all_preds, 3, dim=-1).indices
        test_acc = (top_k_preds == all_trues.unsqueeze(-1)).any(dim=-1).float().mean().item()
        return test_loss, test_acc
