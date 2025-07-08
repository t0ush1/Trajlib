import torch.nn as nn


class MultiTokenLoss(nn.Module):
    def __init__(self):
        super(MultiTokenLoss, self).__init__()
        mse_loss = nn.MSELoss()
        ce_loss = nn.CrossEntropyLoss()
        self.loss_func = {"gps": mse_loss, "grid": ce_loss, "roadnet": ce_loss}

    def forward(self, inputs, targets, tokens, valid_pos=None):
        loss = 0
        for token in tokens:
            loss_fn, input, target = self.loss_func[token], inputs[token], targets[token]
            if valid_pos is not None:
                input, target = input[valid_pos], target[valid_pos]
            loss += loss_fn(input, target)
        return loss
