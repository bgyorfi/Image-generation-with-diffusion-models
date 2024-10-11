# Image generation with diffusion models

Created by: Cristiano McDonaldo

## Contributing

- Füstös Gergely
- Györfi Bence


## Project Overview


This project implements unconditional diffusion models, specifically DDPM (Denoising Diffusion Probabilistic Model) and DDIM (Denoising Diffusion Implicit Model), to generate realistic images. The models are evaluated on two datasets: CelebA (a large-scale celebrity face dataset) and Flowers102 (a dataset containing images of 102 flower categories).

## Data Acquisition

The datasets used in this project are:

- CelebA Dataset: A large-scale face attributes dataset with celebrity images. Downloaded from CelebA dataset page.

- Flowers102 Dataset: A dataset containing 102 flower categories. Downloaded from Flowers102 dataset page.

### Main Steps to reach datasets

    1 Login into your Kaggle account
    2 Get into your account settings page
    3 Click on Create a new API token
    4 This will prompt you to download the .json file into your system. Save the file, and copy it to the main directory. (Exactly where Dockerfile is located)

# How to Start the Application

First, build the Docker Container

```
docker build -t ddpm-project .
```

Second, run it.
```
docker run -it ddpm-project
```