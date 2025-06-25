import torch
import torch.nn.functional as F
from tqdm import tqdm

from trajlib.data.data import SpecialToken
from trajlib.runner.trainers.base_trainer import BaseTrainer, pad_trajectory


def collate_fn(batch):
    traj_batch, label_batch = zip(*batch)
    traj = pad_trajectory(traj_batch)
    label = torch.tensor(label_batch)

    pad_mask = traj[1] != SpecialToken.PAD
    mask = pad_mask.unsqueeze(1) & pad_mask.unsqueeze(2)
    return traj, label, mask.int()


class ClassificationTrainer(BaseTrainer):
    def __init__(self, *args):
        super().__init__(*args, collate_fn=collate_fn)

    def train(self, epoch):
        self.model.train()
        train_loss = 0
        for traj, label, mask in tqdm(
            self.train_loader, disable=not self.accelerator.is_local_main_process, desc=f"Epoch {epoch+1} Train"
        ):
            output = self._call_model(traj, mask=mask)
            label = label.to(self.accelerator.device)

            loss = self.criterion(output, label)
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
            for traj, label, mask in tqdm(
                self.val_loader, disable=not self.accelerator.is_local_main_process, desc=f"Epoch {epoch+1} Valid"
            ):
                output = self._call_model(traj, mask=mask)
                label = label.to(self.accelerator.device)

                preds = self.accelerator.gather_for_metrics(output)
                trues = self.accelerator.gather_for_metrics(label)
                loss = self.criterion(preds, trues)

                val_loss += loss.item()
            val_loss /= len(self.val_loader)
        return val_loss

    def test(self, epoch):
        self.model.eval()
        all_preds, all_trues = [], []
        with torch.no_grad():
            for traj, label, mask in tqdm(
                self.test_loader,
                disable=not self.accelerator.is_local_main_process,
                desc=f"Epoch {epoch+1}  Test" if epoch >= 0 else "Final Test",
            ):
                output = self._call_model(traj, mask=mask)
                label = label.to(self.accelerator.device)

                preds = self.accelerator.gather_for_metrics(output)
                trues = self.accelerator.gather_for_metrics(label)
                all_preds.append(preds)
                all_trues.append(trues)
        all_preds = torch.cat(all_preds, dim=0)
        all_trues = torch.cat(all_trues, dim=0)

        test_loss = F.cross_entropy(all_preds, all_trues).item()
        pred_labels = torch.argmax(all_preds, dim=1)
        test_acc = (pred_labels == all_trues).sum().item() / all_trues.size(0)

        return {
            "Loss": test_loss,
            "Accuracy": test_acc,
        }
