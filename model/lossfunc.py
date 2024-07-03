import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss 
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.reduction = reduction
        self.gamma = gamma
        self.nll_loss = nn.NLLLoss(weight=weight, reduction='none')

    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:

        # compute weighted cross entropy term: -alpha * log(pt)
        # (alpha is already part of self.nll_loss)
        log_p = F.log_softmax(x, dim=-1)
        ce = self.nll_loss(log_p, y)

        # get true class column from each row
        log_pt = log_p[torch.arange(len(x)), y]

        # compute focal term: (1 - pt)^gamma
        pt = log_pt.exp()
        focal_term = (1 - pt)**self.gamma

        # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
        loss:torch.Tensor = focal_term * ce

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss
    
    def __repr__(self):
        return f"Focal Loss"

        
def build_cls_criteria(cls_loss:str="ce", weight:torch.Tensor = None, gamma:float = 2) -> CrossEntropyLoss|FocalLoss:
    if cls_loss == "ce":
        # navie cross entropy
        return CrossEntropyLoss(weight = weight)
    elif cls_loss == "focal":
        return FocalLoss(weight=weight, gamma=gamma)