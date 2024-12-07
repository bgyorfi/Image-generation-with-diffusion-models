import torch
from torch.optim import Adam
from model.unet import Unet
from constants import IMAGE_SIZE, NUM_EPOCH, BATCH_SIZE

from model.sample import params, p_losses
from data_preparation import dataloader_service

def train_ddpm(loader, device):
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


def load_data(device):
    flowers_train_loader, flowers_test_loader = (
        dataloader_service.get_dataloader(
            dataset_name="Flowers",
            batch_size=BATCH_SIZE,
            image_size=IMAGE_SIZE,
            shuffle=True,
            device=device,
            root="/app/datasets/",
        )
    )

    celeb_train_loader, celeb_test_loader = (
        dataloader_service.get_dataloader(
            dataset_name="CelebA",
            batch_size=BATCH_SIZE,
            image_size=IMAGE_SIZE,
            shuffle=True,
            device=device,
            root="/app/datasets/",
        )
    )

    return (
        flowers_train_loader,
        flowers_test_loader,
        celeb_train_loader,
        celeb_test_loader,
    )

