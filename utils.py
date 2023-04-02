import torch.nn.functional as F


def compute_supervised_loss(pred, mask, reduction=True):
    loss = F.cross_entropy(pred, mask.long(), ignore_index=-1)
    return loss
