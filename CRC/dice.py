import torch
def dice_loss(predy, targety):
    inter = 0
    union = 0
    m2sum = 0
    for ele in range(len(predy)):
        for chan in range(1, len(predy[0])):
            pred = predy[ele][chan]
            target = targety[ele][chan]
            pred = (pred > 0.5).type(torch.FloatTensor)
            num = pred.size(0)
            m1 = pred.view(num, -1)  # Flatten
            m2 = target.view(num, -1)  # Flatten
            m2sum += m2.sum()
            inter += ((m1 * m2).sum()).item()
            union += (m1.sum() + m2.sum()).item()
    if m2sum == 0:
        dice = "Klasse nicht vertreten"
    else:
        dice = 2. * inter / union
    return dice

def dice_each(predy, targety, chan):
    inter = 0
    union = 0
    m2sum = 0
    for ele in range(len(predy)):
        pred = predy[ele][chan]
        target = targety[ele][chan]
        pred = (pred > 0.5).type(torch.FloatTensor)
        num = pred.size(0)
        m1 = pred.view(num, -1)  # Flatten
        m2 = target.view(num, -1)  # Flatten
        m2sum += m2.sum()
        inter += ((m1 * m2).sum()).item()
        union += (m1.sum() + m2.sum()).item()
    if m2sum == 0:
        dice = "Klasse nicht vertreten"
    else:
        dice = 2. * inter / union
    return dice

