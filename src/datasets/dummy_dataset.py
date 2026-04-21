import torch

class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, n=1000):
        self.x = torch.randn(n, 3, 224, 224)
        self.y = torch.randint(0, 10, (n,))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]