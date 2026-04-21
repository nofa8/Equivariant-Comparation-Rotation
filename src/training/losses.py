import torch.nn as nn

def build_loss(name):
    if name == "ce":
        return nn.CrossEntropyLoss()
    elif name == "label_smoothing":
        return nn.CrossEntropyLoss(label_smoothing=0.1)