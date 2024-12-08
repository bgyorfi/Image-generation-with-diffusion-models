import argparse, os
from dataset_acquisition import init_dataset_service
from data_preparation import dataloader_service
import torch

from train.train_ddpm import train_ddpm, load_data, load_flowers, load_celebs
from model.generate import generate_images_from_model

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
    torch.save(model.state_dict(), '/app/models/flowers/flowers_model.pth')
    print("Training completed.")

def train_celebs():
    flowers_train_loader, flowers_test_loader, celeb_train_loader, celeb_test_loader = load_data(device)
    print("Training DDPM model on CelebA dataset...")
    model = train_ddpm(celeb_train_loader, device)
    torch.save(model.state_dict(), '/app/models/celeba/celeba_model.pth')
    print("Training completed.")

def generate_images(dataset, latest=False):
    print(f"Generating images for {dataset} dataset...")
    if dataset == "flowers":
        model = load_flowers(device, best=not latest)
    elif dataset == "celeba":
        model = load_celebs(device, best=not latest)
    else:
        print("Invalid dataset name.")
        return

    generate_images_from_model(model, number_of_images=4, dataset=dataset)
    print("Image generation completed.")

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description='Train, evaluate, and generate images using DDPM models.')
        parser.add_argument('--train-flowers', action='store_true', help='Train DDPM model on flowers dataset')
        parser.add_argument('--train-celebs', action='store_true', help='Train DDPM model on CelebA dataset')
        parser.add_argument('--eval', action='store_true', help='Evaluate DDPM models')
        parser.add_argument('--latest', action='store_true', help='Works with --eval or --generate. Use latest trained model instead of best.')
        parser.add_argument('--generate-flowers', action='store_true', help='Generate flower images')
        parser.add_argument('--generate-celebs', action='store_true', help='Generate CelebA images')

        args = parser.parse_args()

        init_dataset_service.download_datasets()

        if args.train_flowers:
            train_flowers()
        elif args.train_celebs:
            train_celebs()
        elif args.eval and args.latest:
            print("Evaluating latest models...")
        elif args.eval:
            print("Evaluating best models...")
        elif args.generate_flowers:
            generate_images("flowers", latest=args.latest)
        elif args.generate_celebs:
            generate_images("celeba", latest=args.latest)

        while True:
            pass
        
    except KeyboardInterrupt:
        print("Exit...")
