import torch.nn as nn
import torch

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, weight = None, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.weight = weight
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(weight = self.weight, reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean() if self.reduction=='mean' else focal_loss.sum()