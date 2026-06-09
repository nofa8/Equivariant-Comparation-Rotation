import torchvision.transforms as T


def build_transforms(img_size=224, augment=False, normalize=False):
    transform_list = []
    
    transform_list.append(T.Resize((img_size, img_size)))
    
    if augment:
        transform_list.append(T.RandomHorizontalFlip())
        transform_list.append(T.RandomRotation(30))
        transform_list.append(T.ColorJitter(0.2, 0.2, 0.1))
        
    # This is where the image or numpy array is converted into a Pytorch Float Tensor
    # The values are scaled to be between 0 and 1
    transform_list.append(T.ToTensor())
    
    
    if normalize:
        # This are the values of the ImageNet dataset, used for pre-trained models
        transform_list.append(
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )
        
    return T.Compose(transform_list)    