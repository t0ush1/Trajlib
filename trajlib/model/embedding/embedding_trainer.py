import time
import pickle
import torch


class EmbeddingTrainer:
    def __init__(self, embedding_config, extra_config):
        self.config = embedding_config | extra_config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    def _train_one_epoch(self):
        raise NotImplementedError()

    def train(self):
        epoch_total = self.config["num_epochs"]
        epoch_train_loss_best = 10000000.0
        epoch_patience = self.config["patience"]
        epoch_worse_count = 0

        emb_name = self.config["emb_name"]
        print(f"[{emb_name}] start.")
        time_training = time.time()
        for epoch in range(epoch_total):
            time_ep = time.time()
            self.model.train()
            loss = self._train_one_epoch()
            print(f"[{emb_name}] i_ep={epoch}, loss={loss:.4f} @={time.time() - time_ep:.1f}")

            if loss < epoch_train_loss_best:
                epoch_train_loss_best = loss
                epoch_worse_count = 0
            else:
                epoch_worse_count += 1
                if epoch_worse_count >= epoch_patience:
                    break

        self.save_checkpoint()
        self.save_embeddings()
        print(f"[{emb_name}] final_ep={epoch}, final_loss={loss}, @={time.time() - time_training:.1f}")

    @torch.no_grad()
    def save_checkpoint(self):
        torch.save({"model_state_dict": self.model.state_dict()}, self.config["ckpt_path"])
        print("[save checkpoint] done.")

    @torch.no_grad()
    def load_checkpoint(self):
        checkpoint = torch.load(self.config["ckpt_path"])
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        print("[load checkpoint] done.")

    def _get_embs(self):
        raise NotImplementedError

    @torch.no_grad()
    def save_embeddings(self):
        embs = self._get_embs()
        with open(self.config["embs_path"], "wb") as fh:
            pickle.dump(embs, fh, protocol=pickle.HIGHEST_PROTOCOL)
        print("[save embedding] done.")
