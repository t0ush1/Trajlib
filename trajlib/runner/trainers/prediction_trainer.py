import torch
import torch.nn.functional as F
from tqdm import tqdm

from trajlib.runner.trainers.base_trainer import BaseTrainer


class PredictionTrainer(BaseTrainer):
    def train(self, epoch):
        self.model.train()
        train_loss = 0
        for x_loc, x_ts, y_loc, y_ts in tqdm(
            self.train_loader, disable=not self.accelerator.is_local_main_process, desc=f"Epoch {epoch+1} Train"
        ):
            x_loc = x_loc.to(self.accelerator.device)
            y_loc = y_loc.to(self.accelerator.device)
            output = self.model(x_loc, self.geo_data)

            loss = self.criterion(output, y_loc.squeeze(1).long())
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
            for x_loc, x_ts, y_loc, y_ts in tqdm(
                self.val_loader, disable=not self.accelerator.is_local_main_process, desc=f"Epoch {epoch+1} Valid"
            ):
                x_loc = x_loc.to(self.accelerator.device)
                y_loc = y_loc.to(self.accelerator.device)
                output = self.model(x_loc)

                preds, trues = self.accelerator.gather_for_metrics([output, y_loc])
                loss = self.criterion(preds, trues.squeeze(1).long())

                val_loss += loss.item()
            val_loss /= len(self.val_loader)
        return val_loss

    def test(self, epoch):
        self.model.eval()
        all_preds, all_trues = [], []
        with torch.no_grad():
            for x_loc, x_ts, y_loc, y_ts in tqdm(
                self.val_loader,
                disable=not self.accelerator.is_local_main_process,
                desc=f"Epoch {epoch+1}  Test" if epoch >= 0 else "Final Test",
            ):
                x_loc = x_loc.to(self.accelerator.device)
                y_loc = y_loc.to(self.accelerator.device)
                output = self.model(x_loc)

                preds, trues = self.accelerator.gather_for_metrics([output, y_loc])
                all_preds.append(preds)
                all_trues.append(trues)
        all_preds = torch.cat(all_preds, dim=0)
        all_trues = torch.cat(all_trues, dim=0)

        test_loss = F.cross_entropy(all_preds, all_trues.squeeze(1).long()).item()
        k = 3
        top_k_preds = torch.topk(all_preds, k, dim=-1).indices
        test_acc = (top_k_preds == all_trues).any(dim=-1).float().mean().item()
        return {
            "Loss": test_loss,
            f"Top-{k} Accuracy": test_acc,
        }
