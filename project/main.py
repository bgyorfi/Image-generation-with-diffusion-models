from dataset_acquisition import init_dataset_service
from data_preparation import dataloader_service
import torch

init_dataset_service.download_datasets()

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

if __name__ == "__main__":
    try:
        test_data_preparation()
        while True:
            pass
    except KeyboardInterrupt:
        print("Kilépés...")