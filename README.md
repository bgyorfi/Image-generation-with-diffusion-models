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

To start the application, use the run.sh script.

Run the gradio interface at [localhost:7860](http://localhost:7860):

```
./run.sh
```

Train the models:

```
./run.sh --train-flowers
./run.sh --train-celebs
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

# File Functions

### `docs/`
Tartalmazza a projekt dokumentációját, képeit és generált eredményeit.
- **`images/`**: A projekt által használt és generált képek mappája.
  - **`celeba/`**: A CelebA adathalmazhoz kapcsolódó mintaképek.
  - **`flowers/`**: A virágok adathalmazhoz kapcsolódó mintaképek.
  - **`results/`**: Nagyobb méretű képhalmazok
    - **`generated_celeb_vae_images/`**: CelebA adathalmazból generált képek VAE segítségével.
    - **`generated_ddpm_celeb_images/`**: CelebA adathalmazból generált képek DDPM segítségével.
    - **`generated_flower_ddpm_images/`**: Virágok adathalmazból generált képek DDPM segítségével.
    - **`generated_flower_vae_images/`**: Virágok adathalmazból generált képek VAE segítségével.

### `project/`
A projekt gyökérkönyvtára, amely az adathalmazokat, modelleket, kódfájlokat és notebookokat tartalmazza.
- **`models/`**: A mentett modellek és modellezéshez kapcsolódó osztályok.
  - **`celeba/`**: CelebA modellek. (*_best elnevezésű a DDPM model)
  - **`flowers/`**: Virág modellek.  (*_best elnevezésű a DDPM model)
- **`notebooks/`**: Jupyter notebookok a projekt vizualizációjához, tanításához, elemzéséhez, kiértékeléséhez.
- **`src/`**: Forráskód a modell és adathalmaz kezeléséhez.
  - **`dataset_acquisition/`**: Az adathalmaz letöltéséhez és kezeléséhez szükséges kódok.
  - **`data_preparation/`**: Az adatok előkészítését végző kódok.
  - **`model/`**: A modellek definiálásához és kezeléséhez szükséges kódok, főként a DDPM modelhez szükséges segédosztályok.
  - **`train/`**: A modellek tanításához tartozó kódok.
- **`train/`**: Tanításhoz szükséges konfigurációs fájlok.
  - **`interface.py`**: Egy Gradio alapú vizuális interfész, amely lehetővé teszi a CelebA és virág képek generálását lépésenkénti előrehaladással, valamint a generált képek galériában történő megjelenítését. Az interfész valós időben frissíti az eredményeket a felhasználói interakciók alapján.
  - **`main.py`**: A fő futtató script, amely az alkalmazás különböző funkcióit kezeli, mint például adatok előkészítése, DDPM modellek tanítása (virágok és CelebA adathalmazokon), értékelés és kép generálás. Gradio interfészen keresztül lehetőség van a generált képek megtekintésére, vagy parancssori argumentumokkal modellek tanítására és értékelésére.


### További file-ok a gyökérkönyvtárban
- **`.gitignore`**: Azokat a fájlokat listázza, amelyeket a Git figyelmen kívül hagy.
- **`docker-compose.yml`**: A Docker konténer indításának konfigurációját tartalmazza.
- **`Dockerfile`**: A Docker konténer környezetének definiálásához szükséges fájl.
- **`kaggle.json`**: API kulcs a Kaggle adathalmazok letöltéséhez.
- **`README.md`**: A projekt áttekintése és dokumentációja.
- **`requirements.txt`**: A Python függőségeket tartalmazó fájl.
- **`run.sh`**: Script az alkalmazás elindításához.
- **`stop.sh`**: Script az alkalmazás leállításához.
- **`Summary_Of_Semester_Work.pdf`**: A szemeszteri munka összefoglaló dokumentuma.

## Related Works

### Research Articles and Blogs
1. Vaibhav Singh. *An In-Depth Guide to Denoising Diffusion Probabilistic Models DDPM – Theory to Implementation* (2023).  
   [LearnOpenCV](https://learnopencv.com/denoising-diffusion-probabilistic-models/)

2. Niels Rogge et al. *The Annotated Diffusion Model*.  
   [Hugging Face Blog](https://huggingface.co/blog/annotated-diffusion)

---

### GitHub Repositories
1. Hugging Face. *Diffusers Library*.  
   [https://github.com/huggingface/diffusers](https://github.com/huggingface/diffusers)

2. Mallick, S. *Guide to Training DDPMs from Scratch: Generating Flowers Using DDPMs*.  
   [LearnOpenCV GitHub Repository](https://github.com/spmallick/learnopencv/blob/master/Guide-to-training-DDPMs-from-Scratch/Generating_flowers_using_DDPMs.ipynb)

3. LearnOpenCV. *Denoising Diffusion Probabilistic Models*.  
   [https://learnopencv.com/denoising-diffusion-probabilistic-models/#What-Are-Diffusion-Probabilistic-Models](https://learnopencv.com/denoising-diffusion-probabilistic-models/#What-Are-Diffusion-Probabilistic-Models)

4. Keras Documentation. *Denoising Diffusion Implicit Models (DDIM)*.  
   [https://keras.io/examples/generative/ddim/](https://keras.io/examples/generative/ddim/)