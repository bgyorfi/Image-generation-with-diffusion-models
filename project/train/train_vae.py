import os
import random
import torch
from torch import nn
import matplotlib.pyplot as plt
from torchmetrics.image.fid import FrechetInceptionDistance
from data_preparation import dataloader_service
from model.vae_model import VAE
from constants import LATENT_DIM, BATCH_SIZE, NUM_EPOCH


def setup_device():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_built() else "cpu"
    )
    return device

def load_data(batch_size = BATCH_SIZE):
    flowers_train_loader, flowers_val_loader, flowers_test_loader = (
        dataloader_service.get_dataloader(
            dataset_name="Flowers", batch_size=batch_size, shuffle=True
        )
    )

    celeb_train_loader, celeb_val_loader, celeb_test_loader = (
        dataloader_service.get_dataloader(
            dataset_name="CelebA", batch_size=batch_size, shuffle=True
        )
    )

    return (flowers_train_loader, flowers_val_loader, flowers_test_loader, 
            celeb_train_loader, celeb_val_loader, celeb_test_loader)

def train(train_loader, device, num_epoch = NUM_EPOCH):
    model = VAE().to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epoch):
        for i, (x_batch, _) in enumerate(train_loader):
            x_batch_gpu = x_batch.to(device)
            x_rec, mu, lv = model(x_batch_gpu)
            kl = torch.sum(0.5 * (mu ** 2 + (torch.exp(lv)) - 1 - lv))
            kl_normalized = kl / x_batch_gpu.size(0)
            loss = torch.sum((x_batch_gpu - x_rec) ** 2) + kl_normalized
            opt.zero_grad()
            loss.backward()
            opt.step()

            if i % 10 == 0:
                print(f"Epoch [{epoch + 1}/{num_epoch}], Batch [{i}/{len(train_loader)}], Loss: {loss.item():.4f}")
    return model

def generate_images(model, device, output_dir='generated_images'):
    os.makedirs(output_dir, exist_ok=True)  
    with torch.no_grad():
        out = model.decode(torch.randn(5, LATENT_DIM).to(device))

    for i in range(5):
        img = out[i].cpu().permute(1, 2, 0).clamp(0, 1)
        save_image(img, os.path.join(output_dir, f'generated_image_{i}.png'))

def save_image(image, file_path):
    """ Save the image tensor as a PNG file. """
    plt.imsave(file_path, image.numpy())

def compare_images(model, test_loader, device, output_dir='reconstructed_images'):
    os.makedirs(output_dir, exist_ok=True) 
    images, _ = next(iter(test_loader))
    random_index = random.randint(0, images.size(0) - 1)
    image = images[random_index].unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        mu, log_var = model.encode(image)
        z = model.reparameterize(mu, log_var)
        reconstructed_image = model.decode(z)

    original_image = image.squeeze(0).cpu().permute(1, 2, 0).clamp(0, 1)
    reconstructed_image = reconstructed_image.squeeze(0).cpu().permute(1, 2, 0).clamp(0, 1)

    save_image(original_image, os.path.join(output_dir, 'original_image.png'))
    save_image(reconstructed_image, os.path.join(output_dir, 'reconstructed_image.png'))

def calculate_fid_score(model, test_loader, device="cpu"):
    fid_metric = FrechetInceptionDistance(feature=2048).to(device=device, dtype=torch.float32)
    model.to(device).eval()

    with torch.no_grad():
        for idx, (images, _) in enumerate(test_loader):
            if idx > 2:  # Limit to the first few batches for performance
                continue
            real_images = (images * 255).clamp(0, 255).to(torch.uint8).to(device)
            reconstructed_images = []

            for img in images:
                img = img.unsqueeze(0).to(device)
                mu, log_var = model.encode(img)
                z = model.reparameterize(mu, log_var)
                reconstructed_img = model.decode(z)
                reconstructed_images.append(reconstructed_img.squeeze(0))

            reconstructed_images = (
                torch.stack(reconstructed_images).mul(255).clamp(0, 255).to(torch.uint8)
            )

            fid_metric.update(real_images, real=True)
            fid_metric.update(reconstructed_images, real=False)

    fid_score = fid_metric.compute()
    return fid_score.item()