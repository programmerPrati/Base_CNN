# Base CNN: Deep Learning Architecture from Scratch

This repository implements a  **Convolutional Neural Network (CNN)** architecture built entirely from scratch in Python. By avoiding high-level abstractions, this project serves as a deep dive into the fundamental mathematics of feature extraction and gradient-based optimization—skills.

## Core Architecture
The network is designed as a sequential pipeline of custom-built layers:

* **Convolutional Layer:** Performs spatial feature extraction by convolving learnable kernels over input volumes, supporting custom strides and padding.
* **Pooling Layer (Max-Pooling):** Reduces spatial dimensions to provide translational invariance and lower computational complexity.
* **Fully Connected (Dense) Layer:** Flattens multidimensional feature maps into 1D vectors for final classification.
* **Activation Functions:** Implementation of non-linearities, specifically **ReLU**, to enable the learning of complex non-linear patterns.



## Dropout
To improve generalization and combat overfitting, I implemented a **Dropout** regularization layer.

### How it Works:
* **Random Masking:** During the training phase, neurons are randomly "dropped" with a probability $p$. This prevents the model from relying on specific "co-adaptations" of neurons.
* **Ensemble Effect:** Dropout effectively trains an ensemble of sub-networks, forcing the model to learn more robust and redundant features.
* **Inference Scaling:** During evaluation, all neurons are active, but their outputs are scaled by $(1-p)$ to ensure the total activation remains consistent with the training phase.


## Usage
1. Open the files in Google Colab.
2. Install dependencies.
3. Execute the cells to see the training loop and loss convergence visualizations.
