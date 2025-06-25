import torch
import torch.nn.functional as F
from tqdm import tqdm

from trajlib.runner.trainers.base_trainer import BaseTrainer


class PredictionTrainer(BaseTrainer):
    def __init__(self, *args, token):
        super().__init__(*args)
        self.token = token

    def train(self, epoch):
        self.model.train()
        train_loss = 0
        for x_traj, y_traj in tqdm(
            self.train_loader, disable=not self.accelerator.is_local_main_process, desc=f"Epoch {epoch+1} Train"
        ):
            output = self._call_model(x_traj)
            y_locs = self._get_tokens(y_traj, self.token).squeeze(1)

            loss = self.criterion(output, y_locs)
            self.optimizer.zero_grad()
            self.accelerator.backward(loss)
            self.optimizer.step()

            train_loss += loss.item()
        train_loss /= len(self.train_loader)
        return train_loss

    def validate(self, epoch):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_traj, y_traj in tqdm(
                self.val_loader, disable=not self.accelerator.is_local_main_process, desc=f"Epoch {epoch+1} Valid"
            ):
                output = self._call_model(x_traj)
                y_locs = self._get_tokens(y_traj, self.token).squeeze(1)

                preds = self.accelerator.gather_for_metrics(output)
                trues = self.accelerator.gather_for_metrics(y_locs)
                loss = self.criterion(preds, trues)

                val_loss += loss.item()
            val_loss /= len(self.val_loader)
        return val_loss

    def test(self, epoch):
        self.model.eval()
        all_preds, all_trues = [], []
        with torch.no_grad():
            for x_traj, y_traj in tqdm(
                self.test_loader,
                disable=not self.accelerator.is_local_main_process,
                desc=f"Epoch {epoch+1}  Test" if epoch >= 0 else "Final Test",
            ):
                output = self._call_model(x_traj)
                y_locs = self._get_tokens(y_traj, self.token).squeeze(1)

                preds = self.accelerator.gather_for_metrics(output)
                trues = self.accelerator.gather_for_metrics(y_locs)
                all_preds.append(preds)
                all_trues.append(trues)
        all_preds = torch.cat(all_preds, dim=0)
        all_trues = torch.cat(all_trues, dim=0)

        if self.token == "gps":
            test_loss = F.mse_loss(all_preds, all_trues).item()
            return {"Loss": test_loss}
        else:
            test_loss = F.cross_entropy(all_preds, all_trues).item()
            k = 3
            top_k_preds = torch.topk(all_preds, k, dim=-1).indices
            test_acc = (top_k_preds == all_trues.unsqueeze(1)).any(dim=-1).float().mean().item()
            return {"Loss": test_loss, f"Top-{k} Accuracy": test_acc}
