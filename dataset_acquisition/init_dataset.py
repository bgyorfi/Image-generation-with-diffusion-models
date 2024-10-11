
from tqdm import tqdm
import pandas as pd
import opendatasets as od
import os

def download_datasets():
    
    flowers_dir = 'datasets/flowers-recognition'
    celebA_dir = 'datasets/celeba-dataset'

    if not os.path.exists(flowers_dir) or not os.listdir(flowers_dir):
        print("Flowers dataset nem található, letöltés folyamatban...")
        flowers_url = "https://www.kaggle.com/datasets/alxmamaev/flowers-recognition"
        od.download(flowers_url)
    else:
        print("Flowers dataset már létezik, nincs szükség letöltésre.")

    if not os.path.exists(celebA_dir) or not os.listdir(celebA_dir):
        print("CelebA dataset nem található, letöltés folyamatban...")
        celebA_url = "https://www.kaggle.com/datasets/jessicali9530/celeba-dataset"
        od.download(celebA_url)
    else:
        print("CelebA dataset már létezik, nincs szükség letöltésre.")
