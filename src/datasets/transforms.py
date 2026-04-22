import torchvision.transforms as T


def build_transforms(img_size=224, augment=False):
    if augment:
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.RandomHorizontalFlip(),
            T.RandomRotation(30),
            T.ColorJitter(0.2, 0.2, 0.1),
            T.ToTensor(),
        ])
    else:
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
        ])