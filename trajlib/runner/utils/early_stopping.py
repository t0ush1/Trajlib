import numpy as np


class EarlyStopping:
    def __init__(self, patience=7, delta=0):
        self.delta = delta
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.val_loss_min = np.Inf

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is not None and score < self.best_score + self.delta:
            self.counter += 1
            is_best = False
            if self.counter >= self.patience:
                is_stop = True
            else:
                is_stop = False
        else:
            self.best_score = score
            self.counter = 0
            is_stop = False
            is_best = True
            self.val_loss_min = val_loss

        return {
            "is_stop": is_stop,
            "is_best": is_best,
            "vali_loss_min": self.val_loss_min,
            "patience": self.patience,
            "counter": self.counter,
        }

    def reset(self):
        self.counter = 0
        self.best_score = None
        self.val_loss_min = np.Inf
