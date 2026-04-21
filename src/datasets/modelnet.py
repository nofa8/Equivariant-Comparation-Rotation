class ModelNetDataset(torch.utils.data.Dataset):
    def __init__(self, root, split, transform=None, angles=None):
        self.root = root
        self.transform = transform
        self.samples = self.load_split(split, angles)

    def load_split(self, split, angles):
        # enforce model-level split
        pass

    def __getitem__(self, idx):
        img, label = ...
        if self.transform:
            img = self.transform(img)
        return img, label