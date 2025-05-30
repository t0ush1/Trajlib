import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from trajlib.data.data import SPECIAL_TOKENS
from trajlib.runner.trainers.base_trainer import BaseTrainer


def collate_fn(batch):
    input_batch, label_batch = zip(*batch)
    input = pad_sequence(input_batch, batch_first=True, padding_value=SPECIAL_TOKENS["pad"])
    label = pad_sequence(label_batch, batch_first=True, padding_value=SPECIAL_TOKENS["pad"])
    label[input != SPECIAL_TOKENS["mask"]] = -100
    mask = input != SPECIAL_TOKENS["pad"]
    mask = mask.unsqueeze(1) & mask.unsqueeze(2)
    return input, label, mask.int()


class AutoregressiveTrainer(BaseTrainer):
    def __init__(self, trainer_config, accelerator, model, dataset, geo_data):
        super().__init__(trainer_config, accelerator, model, dataset, geo_data, collate_fn)

    def train(self, epoch):
        self.model.train()
        train_loss = 0
        for input, label, mask in tqdm(
            self.train_loader, disable=not self.accelerator.is_local_main_process, desc=f"Epoch: {epoch+1}"
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

    def validate(self):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for input, label, mask in self.val_loader:
                input = input.to(self.accelerator.device)
                label = label.to(self.accelerator.device)
                mask = mask.to(self.accelerator.device)
                output = self.model(input, mask=mask, geo_data=self.geo_data)

                outputs, labels = self.accelerator.gather_for_metrics([output, label])
                outputs = outputs.view(-1, outputs.size(-1))
                labels = labels.view(-1)
                loss = self.criterion(outputs, labels)

                val_loss += loss.item()
            val_loss /= len(self.val_loader)
        return val_loss

    def test(self):
        return {}
