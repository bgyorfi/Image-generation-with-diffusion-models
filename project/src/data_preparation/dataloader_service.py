import torch
import torchvision.transforms as TF
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split


class DeviceDataLoader:
    def __init__(self, dataloader, device):
        self.dataloader = dataloader
        self.device = device
        self.dataset = dataloader.dataset

    def __iter__(self):
        for b in self.dataloader:
            yield [item.to(self.device) for item in b]

    def __len__(self):
        return len(self.dataloader)
    
    def get_dataset(self):
        return self.dataset 


class ClampTransform:
    def __init__(self, min_value=0.0, max_value=1.0):
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, img):
        return torch.clamp(img, min=self.min_value, max=self.max_value)


def get_dataset(dataset_name="Flowers", image_size=64, augment=False, root="./datasets", baseline=False):
    if augment:
        transforms = TF.Compose(
            [
                TF.Resize((image_size, image_size)),
                TF.RandomHorizontalFlip(),
                TF.RandomRotation(15),
                TF.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                TF.ToTensor(),
                TF.Lambda(lambda x: x * 2.0 - 1.0),
            ]
        )
    else:
        transform_list = [
            TF.Resize(
                (image_size, image_size),
                interpolation=TF.InterpolationMode.BICUBIC,
                antialias=True,
            ),
            TF.ToTensor(),
            ClampTransform(min_value=0.0, max_value=1.0),
        ]
        if not baseline:
            transform_list.append(TF.Lambda(lambda x: x * 2.0 - 1.0))
        transforms = TF.Compose(transform_list)

    if dataset_name == "Flowers":
        dataset = ImageFolder(
            root=f"{root}flowers-recognition/flowers", transform=transforms
        )
    elif dataset_name == "CelebA":
        dataset = ImageFolder(
            root=f"{root}celeba-dataset/img_align_celeba", transform=transforms
        )
    else:
        raise ValueError(f"Dataset {dataset_name} is not supported.")

    return dataset


def split_dataset(dataset, train_ratio=0.7):
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    test_size = total_size - train_size

    return random_split(dataset, [train_size, test_size])


def get_dataloader(
    dataset_name="Flowers", batch_size=32, image_size=64, shuffle=True, augment=False, device="cpu", root="./../", baseline=False
):
    dataset = get_dataset(dataset_name=dataset_name, root=root, image_size=image_size, augment=augment, baseline=baseline)

    train_set, test_set = split_dataset(dataset)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    train_loader = DeviceDataLoader(train_loader, device)
    test_loader = DeviceDataLoader(test_loader, device)

    return train_loader, test_loader
