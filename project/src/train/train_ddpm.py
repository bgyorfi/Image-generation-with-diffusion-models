import torch
from torch.optim import Adam
from model.unet import Unet
from constants import IMAGE_SIZE, NUM_EPOCH, BATCH_SIZE

from model.sample import params, p_losses
from data_preparation import dataloader_service

def setup_device():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_built() else "cpu"
    )
    return device

device = setup_device()

def train_ddpm(loader):
    model = Unet(
        dim=IMAGE_SIZE,
        dim_mults=(1, 2, 4,)
    )
    model.to(device)

    optimizer = Adam(model.parameters(), lr=1e-3)

    epochs = NUM_EPOCH
    steps = len(loader) * epochs
    progress = 0

    for epoch in range(epochs):
        for step, batch in enumerate(loader):
            optimizer.zero_grad()

            batch, label = batch

            batch_size = batch.shape[0]
            batch = batch.to(device)

            # Algorithm 1 line 3: sample t uniformally for every example in the batch
            t = torch.randint(0, params.timesteps, (batch_size,), device=device).long()

            loss = p_losses(model, batch, t, loss_type="huber")

            loss.backward()
            optimizer.step()

            progress = progress + 1
            print(f"Progress: {progress}/{steps} ({progress / steps * 100:.2f}%, Loss: {loss})", end="\r")
    return model


def load_data():
    flowers_train_loader, flowers_test_loader = (
        dataloader_service.get_dataloader(
            dataset_name="Flowers",
            batch_size=BATCH_SIZE,
            image_size=IMAGE_SIZE,
            shuffle=True,
            device=device,
            root="/app/datasets/flowers102/",
        )
    )

    celeb_train_loader, celeb_test_loader = (
        dataloader_service.get_dataloader(
            dataset_name="CelebA",
            batch_size=BATCH_SIZE,
            image_size=IMAGE_SIZE,
            shuffle=True,
            device=device,
            root="/app/datasets/celeba-dataset/",
        )
    )

    return (
        flowers_train_loader,
        flowers_test_loader,
        celeb_train_loader,
        celeb_test_loader,
    )

def load_flowers_model(best=True):
    model = Unet(
        dim=IMAGE_SIZE,
        dim_mults=(1, 2, 4,)
    )
    model.to("cpu")
    if best:
        model.load_state_dict(torch.load('/app/models/flowers/flowers_model_best.pth', map_location="cpu", weights_only=False))
    else:
        model.load_state_dict(torch.load('/app/models/flowers/flowers_model.pth', map_location="cpu", weights_only=False))
    model.to(device)
    return model

def load_celebs_model(best=True):
    model = Unet(
        dim=IMAGE_SIZE,
        dim_mults=(1, 2, 4,)
    )
    model.to("cpu")
    if best:
        model.load_state_dict(torch.load('/app/models/celeba/celeba_model_best.pth', map_location="cpu", weights_only=False))
    else:
        model.load_state_dict(torch.load('/app/models/celeba/celeba_model.pth', map_location="cpu", weights_only=False))
    model.to(device)
    return model
