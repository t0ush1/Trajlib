import torch
from tqdm import tqdm
import copy
from functools import partial

from trajlib.data.data import SpecialToken
from trajlib.runner.trainers.base_trainer import BaseTrainer, pad_trajectory


def mlm_collate_fn(batch, grid_vocab_size, road_vocab_size):
    input_batch = batch
    label_batch = copy.deepcopy(input_batch)
    input = pad_trajectory(input_batch)
    label = pad_trajectory(label_batch)

    pad_pos = input[1] == SpecialToken.PAD
    prob_mask = torch.rand(input[1].shape)
    mask_pos = ~pad_pos & (prob_mask < 0.15)
    prob_token = torch.rand(input[1].shape)
    mask_token_pos = mask_pos & (prob_token < 0.8)
    rand_token_pos = mask_pos & (prob_token > 0.9)

    input[1][mask_token_pos] = SpecialToken.MASK
    input[2][mask_token_pos] = SpecialToken.MASK
    input[1][rand_token_pos] = torch.randint(0, grid_vocab_size, input[1][rand_token_pos].shape)
    input[2][rand_token_pos] = torch.randint(0, road_vocab_size, input[2][rand_token_pos].shape)

    pad_mask = ~pad_pos
    mask = pad_mask.unsqueeze(1) & pad_mask.unsqueeze(2)
    return input, label, mask.int(), mask_pos & ~pad_pos


def autoregressive_collate_fn(batch):
    input_batch = [[seq[:-1] for seq in traj] for traj in batch]
    label_batch = [[seq[1:] for seq in traj] for traj in batch]
    input = pad_trajectory(input_batch)
    label = pad_trajectory(label_batch)

    pad_pos = input[1] == SpecialToken.PAD
    pad_mask = ~pad_pos
    causal_mask = torch.tril(torch.ones(input[1].size(1), input[1].size(1))).bool()
    mask = pad_mask.unsqueeze(1) & pad_mask.unsqueeze(2) & causal_mask.unsqueeze(0)
    return input, label, mask.int(), ~pad_pos


class FillingTrainer(BaseTrainer):
    def __init__(self, *args, sub_task, tokens, grid_vocab_size, road_vocab_size):
        collate_fns = {
            "mlm": partial(mlm_collate_fn, grid_vocab_size=grid_vocab_size, road_vocab_size=road_vocab_size),
            "autoregressive": autoregressive_collate_fn,
        }
        super().__init__(*args, collate_fn=collate_fns[sub_task])
        self.tokens = tokens

    def train(self, epoch):
        self.model.train()
        train_loss = 0
        for input, label, mask, valid_pos in tqdm(
            self.train_loader, disable=not self.accelerator.is_local_main_process, desc=f"Epoch {epoch+1} Train"
        ):
            output = self._call_model(input, mask=mask)
            label = self._get_seqs(label, self.tokens)

            loss = self.criterion(output, label, self.tokens, valid_pos=valid_pos)
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
            for input, label, mask, valid_pos in tqdm(
                self.val_loader, disable=not self.accelerator.is_local_main_process, desc=f"Epoch {epoch+1} Valid"
            ):
                output = self._call_model(input, mask=mask)
                label = self._get_seqs(label, self.tokens)

                output = self.accelerator.pad_across_processes(output, dim=1)
                label = self.accelerator.pad_across_processes(label, dim=1)
                valid_pos = self.accelerator.pad_across_processes(valid_pos.int(), dim=1).bool()

                output = self.accelerator.gather_for_metrics(output)
                label = self.accelerator.gather_for_metrics(label)
                valid_pos = self.accelerator.gather_for_metrics(valid_pos)

                loss = self.criterion(output, label, self.tokens, valid_pos=valid_pos)

                val_loss += loss.item()
            val_loss /= len(self.val_loader)
        return val_loss

    def test(self, epoch):
        return {}
