import torch
import torch.nn as nn

class WeightedBCELoss(nn.Module):
    def __init__(self, weight_false = [1,1], weight_true = [2.5, 0.625]):
        super(WeightedBCELoss, self).__init__()
        self.weight_false = torch.tensor(weight_false, dtype=torch.float32)
        self.weight_true = torch.tensor(weight_true, dtype=torch.float32)

    def forward(self, output, label):
        loss_false = self.weighted_bce_loss(output[:, 0], label[:, 0], self.weight_false)
        loss_true = self.weighted_bce_loss(output[:, 1], label[:, 1], self.weight_true)
        loss = (loss_false + loss_true) / 2
        return loss

    def weighted_bce_loss(self, probs, targets, weight):
        loss = - (weight[1] * targets * torch.log(probs + 1e-6) + weight[0] * (1 - targets) * torch.log(1 - probs + 1e-6))
        return loss.mean()