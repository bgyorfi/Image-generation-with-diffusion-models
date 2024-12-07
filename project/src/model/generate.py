import torch
import numpy as np
import pandas as pd
from torchvision import transforms as TF

from model.sample import sample
from constants import IMAGE_SIZE


reverse_transform = TF.Compose([
    TF.Lambda(lambda t: (t + 1) / 2),
    TF.Lambda(lambda t: t.permute(1, 2, 0)),
    TF.Lambda(lambda t: t * 255.),
    TF.Lambda(lambda t: t.numpy().astype(np.uint8)),
    TF.ToPILImage()
])

def get_img_from_sample(sample):
    return reverse_transform(torch.from_numpy(sample).squeeze())

def generate_images_from_model(model, number_of_images=4, dataset="flowers"):
    samples_new = sample(model, image_size=IMAGE_SIZE, batch_size=number_of_images, channels=3)
    indexes = range(0, number_of_images)
    for idx in indexes:
        for step in [50, 100, 200, 300]:
            image = get_img_from_sample(samples_new[step][idx])
            image_path = f"/app/images/{dataset}/{pd.Timestamp.now().strftime('%Y-%m-%d-%H-%M')}_{idx}_{step}.png"
            image.save(image_path)


