import torchvision.transforms as TF
import torchvision.datasets as datasets
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


def get_dataset(dataset_name='Flowers'):
    transforms = TF.Compose(
        [
            TF.ToTensor(),
            TF.Resize((32, 32), 
                      interpolation=TF.InterpolationMode.BICUBIC, 
                      antialias=True),
            TF.RandomHorizontalFlip(),
            TF.Lambda(lambda t: (t * 2) - 1) # Scale between [-1, 1] 
        ]
    )
    
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