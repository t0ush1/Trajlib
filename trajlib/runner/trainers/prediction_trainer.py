import torch
import torch.nn.functional as F
from tqdm import tqdm

from trajlib.runner.trainers.base_trainer import BaseTrainer


class PredictionTrainer(BaseTrainer):
    def __init__(self, *args, tokens):
        super().__init__(*args)
        self.tokens = tokens

    def train(self, epoch):
        self.model.train()
        train_loss = 0
        for x_traj, y_traj in tqdm(
            self.train_loader, disable=not self.accelerator.is_local_main_process, desc=f"Epoch {epoch+1} Train"
        ):
            output = self._call_model(x_traj)
            y_traj = self._get_seqs(y_traj, self.tokens)
            y_traj = {token: y_traj[token].squeeze(1) for token in self.tokens}

            loss = self.criterion(output, y_traj, self.tokens)
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
                y_traj = self._get_seqs(y_traj, self.tokens)
                y_traj = {token: y_traj[token].squeeze(1) for token in self.tokens}

                output = self.accelerator.gather_for_metrics(output)
                y_traj = self.accelerator.gather_for_metrics(y_traj)
                loss = self.criterion(output, y_traj, self.tokens)

                val_loss += loss.item()
            val_loss /= len(self.val_loader)
        return val_loss

    def test(self, epoch):
        self.model.eval()
        all_output = {token: [] for token in self.tokens}
        all_y_traj = {token: [] for token in self.tokens}
        with torch.no_grad():
            for x_traj, y_traj in tqdm(
                self.test_loader,
                disable=not self.accelerator.is_local_main_process,
                desc=f"Epoch {epoch+1}  Test" if epoch >= 0 else "Final Test",
            ):
                output = self._call_model(x_traj)
                y_traj = self._get_seqs(y_traj, self.tokens)
                y_traj = {token: y_traj[token].squeeze(1) for token in self.tokens}

                output = self.accelerator.gather_for_metrics(output)
                y_traj = self.accelerator.gather_for_metrics(y_traj)

                for token in self.tokens:
                    all_output[token].append(output[token])
                    all_y_traj[token].append(y_traj[token])

        for token in self.tokens:
            all_output[token] = torch.cat(all_output[token], dim=0)
            all_y_traj[token] = torch.cat(all_y_traj[token], dim=0)

        results = {"Loss": 0}
        for token in self.tokens:
            token: str
            output: torch.Tensor = all_output[token]
            y_traj: torch.Tensor = all_y_traj[token]

            if token == "gps":
                test_loss = F.mse_loss(output, y_traj).item()
                results[f"Loss of GPS"] = test_loss
            else:
                test_loss = F.cross_entropy(output, y_traj).item()
                k = 3
                top_k_preds = torch.topk(output, k, dim=-1).indices
                test_acc = (top_k_preds == y_traj.unsqueeze(1)).any(dim=-1).float().mean().item()
                results[f"Loss of {token.capitalize()}"] = test_loss
                results[f"Top-{k} Accuracy of {token.capitalize()}"] = test_acc

            results["Loss"] += test_loss
        return results
