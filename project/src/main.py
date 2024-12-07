import argparse, os
from dataset_acquisition import init_dataset_service
from data_preparation import dataloader_service
import torch

from train.train_ddpm import train_ddpm, load_data

def setup_device():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_built() else "cpu"
    )
    return device

device = setup_device()

def test_data_preparation():
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")

        print("Loading data using dataloader_service...")
        train_loader, val_loader, test_loader = dataloader_service.get_dataloader(
            dataset_name="Flowers",
            batch_size=32,
            shuffle=True,
            device=device
        )

        print("Running data preparation test...")

        # Ellenőrizzük, hogy a DataLoader-ek nem üresek
        assert len(train_loader) > 0, "Train DataLoader is empty!"
        assert len(val_loader) > 0, "Validation DataLoader is empty!"
        assert len(test_loader) > 0, "Test DataLoader is empty!"
        
        print(f"Train DataLoader length: {len(train_loader)} batches")
        print(f"Validation DataLoader length: {len(val_loader)} batches")
        print(f"Test DataLoader length: {len(test_loader)} batches")

        # Megnézzük az első batch-t mindegyik DataLoader-ből
        train_batch = next(iter(train_loader))
        val_batch = next(iter(val_loader))
        test_batch = next(iter(test_loader))

        print("Fetched first batches from each DataLoader...")

        # Ellenőrizzük, hogy a batch-ek megfelelő formátumúak-e
        assert len(train_batch) == 2, "Train batch should contain inputs and labels!"
        assert len(val_batch) == 2, "Validation batch should contain inputs and labels!"
        assert len(test_batch) == 2, "Test batch should contain inputs and labels!"

        print("Batch format check passed for train, validation, and test loaders.")

        # Az input képek és címkék formátumának ellenőrzése
        images, labels = train_batch
        print(f"Train batch - Images shape: {images.shape}, Labels shape: {labels.shape}")

        assert images.shape[0] == 32, "Batch size should be 32!"
        assert images.shape[1:] == (3, 32, 32), "Images should be resized to (3, 32, 32)!"

        print("Batch size and image dimensions check passed.")

        # Képek statisztikák megjelenítése (pl. min, max értékek)
        print(f"Sample image stats - Min: {images.min().item()}, Max: {images.max().item()}, Mean: {images.mean().item()}")

        #A címkék a megfelelő tartományban vannak-e
        unique_labels = set(labels.tolist())
        print(f"Unique labels in train batch: {unique_labels}")

        print("Data preparation test passed successfully!")
    
    except Exception as e:
        print(f"Data preparation test failed: {e}")

def train_flowers():
    flowers_train_loader, flowers_test_loader, celeb_train_loader, celeb_test_loader = load_data(device)
    print("Training DDPM model on Flowers dataset...")
    model = train_ddpm(flowers_train_loader, device)
    torch.save(model.state_dict(), 'model_flowers.pth')
    print("Training completed.")


def train_models(flowers_only=False):
    flowers_train_loader, flowers_test_loader, celeb_train_loader, celeb_test_loader = load_data(device)
    if (flowers_only):
        print("Training on flowers dataset only")
        model = train_ddpm(flowers_train_loader, device)
        print("Training completed.")
        return model
    else:
        print("Training on both datasets")
    return

def load_models():
    print("loading modes")
    return 
    print("Loading trained models from disk...")
    device = train_vae.setup_device()
    model_flowers = vae_model.VAE()
    # model_celeb = vae_model.VAE()
    model_flowers.load_state_dict(torch.load('model_flowers.pth'))
    # model_celeb.load_state_dict(torch.load('model_celeb.pth'))
    print("Models loaded successfully.")
    return model_flowers, device
#  return model_flowers, model_celeb, device

def generate_images(model_name):
    print(f"Generating images from {model_name}...")
    return
    print("Generating images...")

    if args.generate_flower:
        train_vae.generate_images(model_flowers, device, output_dir='generated_flower_images')
        print("Flower images generated successfully.")
    # elif args.generate_celeb:
    #     train_vae.generate_images(model_celeb, device, output_dir='generated_celeb_images')
    #     print("CelebA images generated successfully.")


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description='Train, evaluate, and generate images using DDPM models.')
        parser.add_argument('--train-flowers', action='store_true', help='Train DDPM model on flowers dataset')
        parser.add_argument('--flowers-celebs', action='store_true', help='Train DDPM model on CelebA dataset')
        parser.add_argument('--eval', action='store_true', help='Evaluate DDPM models')
        parser.add_argument('--generate-flowers', action='store_true', help='Generate flower images')
        parser.add_argument('--generate-celebs', action='store_true', help='Generate CelebA images')

        args = parser.parse_args()

        init_dataset_service.download_datasets()

        if args.train:
            train_models(flowers_only=args.flowers_only)
        elif args.eval:
            print("Evaluating models...")
        elif args.generate_flowers:
            generate_images("flowers")
        elif args.generate_celebs:
            generate_images("CelebA")

        while True:
            pass
        
    except KeyboardInterrupt:
        print("Exit...")
