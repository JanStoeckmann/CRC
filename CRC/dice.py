import torch.nn as nn
loss_function = nn.MSELoss()
from torch.autograd import Variable
def dice_loss(pred, target):
    smooth = 1.
    num = pred.size(0)
    m1 = pred.view(num, -1)  # Flatten
    m2 = target.view(num, -1)  # Flatten
    intersection = (m1 * m2).sum()

    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

def multiple_dice(pred_target):
    total = 0.
    for [pred,target] in pred_target:
        total += dice_loss(pred,target)
    dice = total/len(pred_target)
    return dice

def multiple_loss(pred_target):
    loss = 0.
    for [pred, target] in pred_target:
        loss += loss_function(pred, target)
    return loss