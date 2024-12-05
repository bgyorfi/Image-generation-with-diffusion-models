import os
import opendatasets as od

def download_datasets():
    flowers_dir = '/app/datasets/flowers102'
    celebA_dir = '/app/datasets/celeba-dataset'

    if not os.path.exists(flowers_dir) or not os.path.isdir(flowers_dir) or len(os.listdir(flowers_dir)) == 0:
        print("Flowers dataset not found, download in progress...")
        flowers_url = "https://www.kaggle.com/datasets/alxmamaev/flowers-recognition"
        od.download(flowers_url, data_dir=flowers_dir)
        print("Flowers dataset downloaded successfully.")
    else:
        print("Flowers dataset already downloaded.")

    if not os.path.exists(celebA_dir) or not os.path.isdir(celebA_dir) or len(os.listdir(celebA_dir)) == 0:
        print("CelebA dataset not found, download in progress...")
        celebA_url = "https://www.kaggle.com/datasets/jessicali9530/celeba-dataset"
        od.download(celebA_url, data_dir=celebA_dir)
        print("CelebA dataset downloaded successfully.")
    else:
        print("CelebA dataset already downloaded.")
