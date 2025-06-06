import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from trajlib.data.data import SPECIAL_TOKENS
from trajlib.runner.trainers.base_trainer import BaseTrainer


def collate_fn(batch):
    input_batch, label_batch = zip(*batch)
    input = pad_sequence(input_batch, batch_first=True, padding_value=SPECIAL_TOKENS["pad"])
    label = torch.tensor(label_batch)
    pad_mask = input != SPECIAL_TOKENS["pad"]
    mask = pad_mask.unsqueeze(1) & pad_mask.unsqueeze(2)
    return input, label, mask.int()


class ClassificationTrainer(BaseTrainer):
    def __init__(self, trainer_config, accelerator, model, dataset, geo_data):
        super().__init__(trainer_config, accelerator, model, dataset, geo_data, collate_fn)

    def train(self, epoch):
        self.model.train()
        train_loss = 0
        for input, label, mask in tqdm(
            self.train_loader, disable=not self.accelerator.is_local_main_process, desc=f"Epoch {epoch+1} Train"
        ):
            input = input.to(self.accelerator.device)
            label = label.to(self.accelerator.device)
            mask = mask.to(self.accelerator.device)
            output = self.model(input, mask=mask, geo_data=self.geo_data)

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
            for input, label, mask in tqdm(
                self.val_loader, disable=not self.accelerator.is_local_main_process, desc=f"Epoch {epoch+1} Valid"
            ):
                input = input.to(self.accelerator.device)
                label = label.to(self.accelerator.device)
                mask = mask.to(self.accelerator.device)
                output = self.model(input, mask=mask)

                preds, trues = self.accelerator.gather_for_metrics([output, label])
                loss = self.criterion(preds, trues)

                val_loss += loss.item()
            val_loss /= len(self.val_loader)
        return val_loss

    def test(self, epoch):
        self.model.eval()
        all_preds, all_trues = [], []
        with torch.no_grad():
            for input, label, mask in tqdm(
                self.test_loader,
                disable=not self.accelerator.is_local_main_process,
                desc=f"Epoch {epoch+1}  Test" if epoch >= 0 else "Final Test",
            ):
                input = input.to(self.accelerator.device)
                label = label.to(self.accelerator.device)
                mask = mask.to(self.accelerator.device)
                output = self.model(input, mask=mask)

                preds, trues = self.accelerator.gather_for_metrics([output, label])
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
