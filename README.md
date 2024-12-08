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

Open a terminal in the project root directory. To start the application, follow these steps:

### 1. Build the Docker Image

Before running the application, you need to build the Docker image:

```
docker-compose build
```

### 2. Grant Execute Permission to Scripts

If you are using the run and stop scripts, ensure they are executable by running:

```
chmod +x run.sh
chmod +x stop.sh
```

This command gives the necessary permissions for the scripts to run.

### 3. Run the Application

To start the application, use the run.sh script. Please add one of the following arguments.

Train the models:

```
./run.sh --train-flowers
./run.sh --train-celebs
```

Evaluate the models:

```
./run.sh --eval                        # Evaluate best model
./run.sh --eval --latest               # Evaluate latest model
```

Generate flower images:

```
./run.sh --generate-flowers            # Use best model
./run.sh --generate-flowers --latest   # Use latest model instead of best
./run.sh --generate-celebs             # Use best model
./run.sh --generate-celebs --latest    # Use latest model instead of best
```

### 4. Stop the Application

To stop the application, use the stop.sh script:

```
./stop.sh
```

This will stop the running Docker container.

### 5. Remove the Container (Optional)

If you need to completely stop and remove the container, you can use:

```
docker-compose down
```

This will remove the container and associated resources.

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

"[The Annotated Diffusion Model](https://huggingface.co/blog/annotated-diffusion)" - by Niels Rogge et al. (2022)


- **GitHub Repositories**:
  - https://github.com/pytorch/pytorch - LearnOpenCV

