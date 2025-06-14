import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from trajlib.data.data import SPECIAL_TOKENS
from trajlib.runner.trainers.base_trainer import BaseTrainer


def mlm_collate_fn(batch):
    input_batch, label_batch = zip(*batch)
    input = pad_sequence(input_batch, batch_first=True, padding_value=SPECIAL_TOKENS["pad"])
    label = pad_sequence(label_batch, batch_first=True, padding_value=SPECIAL_TOKENS["pad"])
    label[input != SPECIAL_TOKENS["mask"]] = SPECIAL_TOKENS["ignore"]
    mask = input != SPECIAL_TOKENS["pad"]
    mask = mask.unsqueeze(1) & mask.unsqueeze(2)
    return input, label, mask.int()


def autoregressive_collate_fn(batch):
    input_batch, label_batch = zip(*batch)
    input = pad_sequence(input_batch, batch_first=True, padding_value=SPECIAL_TOKENS["pad"])
    label = pad_sequence(label_batch, batch_first=True, padding_value=SPECIAL_TOKENS["pad"])
    label[input == SPECIAL_TOKENS["pad"]] = SPECIAL_TOKENS["ignore"]
    pad_mask = input != SPECIAL_TOKENS["pad"]
    causal_mask = torch.tril(torch.ones(input.size(1), input.size(1))).bool()
    mask = pad_mask.unsqueeze(1) & pad_mask.unsqueeze(2) & causal_mask.unsqueeze(0)
    return input, label, mask.int()


# TODO 改名为填补任务，masked/padding/mask在dataset还是collate_fn中处理？写两个版本
class FillingTrainer(BaseTrainer):
    def __init__(self, trainer_config, accelerator, model, dataset, geo_data, sub_task):
        super().__init__(
            trainer_config,
            accelerator,
            model,
            dataset,
            geo_data,
            mlm_collate_fn if sub_task == "mlm" else autoregressive_collate_fn,
        )

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

            output = output.view(-1, output.size(-1))
            label = label.view(-1)
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

                outputs, labels = self.accelerator.gather_for_metrics([output, label])
                outputs = outputs.view(-1, outputs.size(-1))
                labels = labels.view(-1)
                loss = self.criterion(outputs, labels)

                val_loss += loss.item()
            val_loss /= len(self.val_loader)
        return val_loss

    def test(self, epoch):
        return {}
