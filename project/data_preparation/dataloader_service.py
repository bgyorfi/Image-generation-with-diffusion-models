import torch
import torchvision.transforms as TF
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split

class DeviceDataLoader():
    def __init__(self, dataloader, device):
        self.dataloader = dataloader
        self.device = device

    def __iter__(self):
        for b in self.dataloader:
            yield [item.to(self.device) for item in b]

    def __len__(self):
        return len(self.dataloader)

class ClampTransform:
    def __init__(self, min_value=0.0, max_value=1.0):
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, img):
        return torch.clamp(img, min=self.min_value, max=self.max_value)


def get_dataset(dataset_name='Flowers', , augment=False):
    if augment:
        transforms = TF.Compose([
            TF.Resize((256, 256)),
            TF.RandomHorizontalFlip(),
            TF.RandomRotation(15),
            TF.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            TF.ToTensor(),
            TF.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    else:
        transforms = TF.Compose([
            TF.Resize((128, 128), interpolation=TF.InterpolationMode.BICUBIC, antialias=True),
            TF.ToTensor(),
            ClampTransform(min_value=0.0, max_value=1.0),
        ])
    
    if dataset_name == "Flowers":
        dataset = ImageFolder(root="flowers-recognition/flowers", transform=transforms)
    elif dataset_name == "CelebA":
        dataset = ImageFolder(root="celeba-dataset/img_align_celeba", transform=transforms)
    else:
        raise ValueError(f"Dataset {dataset_name} is not supported.")
    
    return dataset

def split_dataset(dataset, train_ratio=0.7, val_ratio=0.15):
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    
    return random_split(dataset, [train_size, val_size, test_size])

def get_dataloader(dataset_name='Flowers', 
                   batch_size=32, 
                   shuffle=True,
                   device="cpu"
                  ):
    dataset = get_dataset(dataset_name=dataset_name)
    
    train_set, val_set, test_set = split_dataset(dataset)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    train_loader = DeviceDataLoader(train_loader, device)
    val_loader = DeviceDataLoader(val_loader, device)
    test_loader = DeviceDataLoader(test_loader, device)
    
    return train_loader, val_loader, test_loader
