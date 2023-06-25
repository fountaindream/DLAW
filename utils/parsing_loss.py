import torch
import torch.nn as nn
from torch.nn import functional as F

class ParsingLoss(nn.Module):
    """
    Learn senmatic part features
    """
    def __init__(self, ignore_label=-1, weight=None):
        super(ParsingLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(weight=weight, 
                                             ignore_index=ignore_label)

    def forward(self, score, target):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.upsample(
                    input=score, size=(h, w), mode='bilinear',align_corners=True)
        loss = self.criterion(score, target)
        return loss