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

Open a terminal in the project root directory. First, build the Docker Image:

```
docker build -t ddpm-project .
```

Then, run the docker container with docker compose:
```
docker compose up
```

If you want to delete the container you can do it with:
```
docker compose down
```

## File Functions

- **`data_preparation/`**: This directory contains scripts related to data preparation and loading.
  - **`dataloader_service.py`**: Responsible for creating DataLoader instances for training, validation, and testing datasets. It handles batching and shuffling of data.
  - **`init_dataset_service.py`**: Initializes the dataset service, manages dataset access and setup.
- **`main.py`**: The main entry point for running the data preparation tests and loading the datasets.
- **`docker-compose.yml`**: Makes it easier to run the container as an application.
- **`Dockerfile`**: Contains the instructions to build a Docker container for the application. It sets up the environment, installs dependencies, and configures the working directory.
- **`.gitignore`**: Lists files and directories to be ignored by Git, ensuring that unnecessary files are not included in the version control.
- **`kaggle.json`**: Contains API credentials for accessing Kaggle datasets.
- **`requirements.txt`**: Lists Python package dependencies required to run the project.
- **`README.md`**: This file, which provides an overview and documentation for the project.

## Related Works

"An In-Depth Guide to Denoising Diffusion Probabilistic Models DDPM – Theory to Implementation" - by Vaibhav Singh (2023) 

"Denoising Diffusion Implicit Models" by András Béres (2022)

"The Annotated Diffusion Model" - by Niels Rogge et al. (2022)

- **GitHub Repositories**:
  - (https://github.com/pytorch/pytorch) - LearnOpenCV
