import torch
import numpy as np
import pandas as pd

from model.sample import sample, reverse_transform, sample_with_steps
from train.train_ddpm import load_flowers_model, load_celebs_model
from constants import IMAGE_SIZE

def get_img_from_sample(sample):
    return reverse_transform(torch.from_numpy(sample).squeeze())

def generate_images_from_model(model, number_of_images=4, dataset="flowers"):
    samples_new = sample(model, image_size=IMAGE_SIZE, batch_size=number_of_images, channels=3)
    indexes = range(0, number_of_images)
    for idx in indexes:
        image = get_img_from_sample(samples_new[-1][idx])
        image_path = f"/app/images/{dataset}/{pd.Timestamp.now().strftime('%Y-%m-%d-%H-%M')}_{idx}.png"
        image.save(image_path)

def generate_images(dataset, latest=False):
    print(f"Generating images for {dataset} dataset...")
    if dataset == "flowers":
        model = load_flowers_model(best=not latest)
    elif dataset == "celeba":
        model = load_celebs_model(best=not latest)
    else:
        print("Invalid dataset name.")
        return

    generate_images_from_model(model, number_of_images=4, dataset=dataset)
    print("Image generation completed.")

def generate_images_with_steps(dataset):
    print(f"Generating images for {dataset} dataset...")
    if dataset == "flowers":
        model = load_flowers_model(best=True)
    elif dataset == "celeba":
        model = load_celebs_model(best=True)
    else:
        print("Invalid dataset name.")
        return

    for current_image, final_image in sample_with_steps(model, IMAGE_SIZE, batch_size=1, channels=3):
        yield current_image, final_image

    image_path = f"/app/images/{dataset}/{pd.Timestamp.now().strftime('%Y-%m-%d-%H-%M')}.png"
    final_image.save(image_path)
