import torch
import torchvision.transforms as T
import albumentations as A

mean = (0.485, 0.456, 0.406)  # RGB
std = (0.229, 0.224, 0.225)  # RGB


def aug_transform1(size):
    size = list(map(int, size.split(',')))
    transform = {
        'albu_train': A.Compose([
            A.Resize(size[0], size[1]),
            A.RandomResizedCrop(size[0], size[1]),
            A.Transpose(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(p=0.5),
            A.CoarseDropout(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.Normalize(mean, std),
        ]),
        'torch_train': T.Compose([
            T.ConvertImageDtype(torch.float),
        ]),
        'albu_val': A.Compose([
            A.Resize(size[0], size[1]),
            A.Normalize(mean, std),
        ]),
        'torch_val': T.Compose([
            T.ConvertImageDtype(torch.float),
        ])
    }
    return transform
