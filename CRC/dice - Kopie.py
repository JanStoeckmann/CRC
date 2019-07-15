import torch
def dice_loss(pred, target):
    #todo backgound wegschneiden
    pred = (pred > 0.5).type(torch.FloatTensor)
    num = pred.size(0)
    m1 = pred.view(num, -1)  # Flatten
    m2 = target.view(num, -1)  # Flatten
    if m2.sum() == 0:
        dice = "Klasse nicht vertreten"
    else:
        dice = (2. * (m1 * m2).sum()) / (m1.sum() + m2.sum())
        dice = dice.item()
    return dice