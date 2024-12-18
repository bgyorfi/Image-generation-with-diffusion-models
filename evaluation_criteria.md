# Evaluation criteria

## Overview

In this project, our task is to implement a complex generative AI using the DDPM (Denoising Diffusion Probabilistic Model) method for image generation. Our main goal is to outperform the baseline model, so we need reliable metrics to evaluate both models’ performance. Evaluating generated images can be challenging, so we’ve gathered several approaches to do this effectively. Below, we outline each of these methods in detail:

## 1. Fréchet Inception Distance (FID)

The Fréchet Inception Distance (FID) evaluates the quality of images generated by models by comparing their distribution to real images. It captures both image quality and diversity, providing a reliable assessment for generative models.

### Mathematical Representation

Let:

- $\mu_r$ and $\Sigma_r$ be the mean and covariance matrix of real images.
- $\mu_g$ and $\Sigma_g$ be the mean and covariance matrix of generated images.

The FID score is calculated as:

$$\text{FID} = \|\mu_r - \mu_g\|^2_2 + \text{Tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2})$$

Where:

- $\|\mu_r - \mu_g\|^2_2$: Squared Euclidean distance between the means.
- $\text{Tr}$: Trace of a matrix.
- $(\Sigma_r \Sigma_g)^{1/2}$: Matrix square root of the product of covariances.

### Interpretation

- **Lower FID**: Indicates better similarity to real images (higher quality and diversity).
- **Higher FID**: Suggests lower quality or diversity.

### Conclusion

FID is a standard metric for generative models, balancing quality and diversity by comparing the distribution of generated images to real ones.

## 2. Inception Score (IS)

The Inception Score (IS) is a metric used to evaluate the quality and diversity of images generated by generative models. It measures how well a model's generated images resemble real-world images based on the confidence of a pretrained classifier.

### Mathematical Representation

Given a generated image $x$, let $p(y|x)$ be the conditional probability distribution of image class labels $y$ given by a pretrained Inception network. The Inception Score is calculated as:

$$\text{IS} = \exp ( E_x [ D_{\text{KL}}(p(y|x) \| p(y)) ])$$

Where:

- $D_{\text{KL}}$ is the Kullback–Leibler divergence.
- $p(y)$ is the marginal distribution, $p(y) = \frac{1}{N} \sum_{i=1}^N p(y|x_i)$, averaged over all generated images.

### Interpretation

- **Higher IS**: Indicates higher quality and diversity of generated images. The model produces images that are easily classified into distinct classes with high confidence.
- **Lower IS**: Implies less realistic or less diverse images.

### Conclusion

The Inception Score is an effective metric for evaluating generative models as it balances both image quality and diversity. It measures **_classifiability_** by assessing how confidently generated images can be assigned to specific classes and captures **_diversity_** by ensuring that the generated images vary across different categories rather than being repetitive or assigned to the same class. This makes it a valuable tool for understanding both the clarity and the variety of outputs from generative models.

## 3. Visual Turing Test

The Visual Turing Test, is a subjective evaluation method used to assess the quality of images generated by generative models. It involves human participants reviewing and rating or distinguishing generated images from real ones to provide insight into how realistic and convincing the generated content is.

### How It Works

1. **Setup**: Participants are shown a mix of real and generated images without being told which is which.
2. **Tasks**:
   - **Classification**: Participants are asked to identify whether each image is real or generated.
   - **Rating**: Participants may also rate the quality or realism of the images on a predefined scale (e.g., from 1 to 5).
3. **Data Collection**: Responses are collected and analyzed to determine the model's success in producing realistic and indistinguishable images.

### Properties

- **Subjectivity**: Results are based on human perception, making them valuable for assessing how real or convincing the generated images appear.
- **Flexibility**: The study design can be adapted for various evaluation needs, including testing specific attributes of generated images.
- **Perceptual Insight**: Provides feedback that aligns with human judgment, which is especially useful when quantitative metrics (e.g., IS or FID) may not capture subtle perceptual details.

### Advantages

- **Human-Centric**: Captures the subjective perception of image quality, which may not align with purely quantitative metrics.
- **Qualitative Feedback**: Offers detailed insights into specific strengths and weaknesses of the generative model.

### Limitations

- **Time-Consuming**: Setting up and conducting user studies can be time-intensive, especially for large sample sizes.
- **Subjectivity**: Results may vary based on the participants' backgrounds, expertise, and biases.
- **Scalability**: Harder to scale compared to automated evaluation methods.

### Conclusion

The Visual Turing Test or is a powerful method for evaluating generative models when human perception is a priority. While it may be more resource-intensive than automated metrics, it provides essential qualitative feedback that helps align generated outputs with human expectations.
