import torch.optim as optim

def build_optimizer(model, config):
    if config["optimizer"] == "sgd":
        opt = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    else:
        opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    return opt