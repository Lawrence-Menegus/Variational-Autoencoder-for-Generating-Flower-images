# Variational-Autoencoder-(VAE)-for-Flower-Image-Generation
<p>The provided program implements a Variational Autoencoder (VAE) for generating and reconstructing flower images. The program utilizes TensorFlow and Keras to preprocess data, build encoder-decoder architectures, and train the model for latent space exploration and image generation.</p>

## Key Features
#### Data Preparation:
Supports dataset extraction and loading on both Google Colab and local machines.
Preprocesses images by resizing and normalizing to the range [0, 1].
#### Model Architecture:
Encoder: Uses convolutional layers to compress input images into latent space, outputting mean and log variance vectors.
Decoder: Uses transposed convolutional layers to reconstruct images from latent space.
VAE: Integrates encoder, decoder, and sampling function for end-to-end training.
#### Training and Evaluation:
Custom loss includes:
Reconstruction Loss: Scaled Mean Squared Error (MSE) for pixel-wise reconstruction.
KL Divergence: Regularizes the latent space for Gaussian distribution.
Tracks loss metrics (total loss, reconstruction loss, and KL divergence) during training.
#### Visualization:
Displays original images, reconstructed outputs, and new images generated from latent space sampling.
### Install the Package
pip install tensorflow matplotlib
<p>Ensure TensorFlow and Matplotlib are installed before running the program. For TensorFlow installation instructions, visit [TensorFlow Installation](https://www.tensorflow.org/install).</p>

## Running the Program
#### Dataset Preparation:
Extract the flower dataset from the zip file to the appropriate directory.
Structure the dataset into training and validation directories.
#### On Google Colab:
Mount Google Drive and set paths to the dataset accordingly.
Use the provided script to unzip and organize the dataset.
#### On Local Machine:
Ensure the dataset path is correctly set in the script for training and validation directories.
Use the provided script to unzip the dataset.
#### Execution:
Preprocess the dataset using image_dataset_from_directory.
Build and compile the VAE model.
Train the model with the training dataset.
Visualize reconstructed and generated images.

## Contributor
<p>Lawrence Menegus</p>
