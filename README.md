# Transformers and Vision Transformers

This project is a flexible and educational framework for experimenting with both standard Transformer and Vision Transformer (ViT) architectures. It allows users to build, train, and evaluate models with customizable parameters, making it an ideal environment for learning and research.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [File Structure](#file-structure)
- [Getting Started](#getting-started)
- [Running the Project](#running-the-project)
- [Configuration](#configuration)
- [Editable Parameters](#editable-parameters)

---

## Project Overview

Originally designed for Natural Language Processing, the Transformer architecture has proven to be a powerful tool in various domains. This project provides a hands-on implementation of Transformers, with a specific focus on Vision Transformers for image classification tasks on the MNIST dataset. You can easily switch between different models and datasets as more are implemented.

---

## Features

- **Customizable Models**: Both standard Transformer and ViT architectures can be modified through the `config.ini` file.
- **Pluggable Architectures**: The project is designed to easily incorporate new models and datasets.
- **Easy Experimentation**: Modify hyperparameters and architectural details to see their impact on performance.
- **Standard ViT Structure**:
    - Image patching and embedding
    - Transformer encoder blocks (multi-head attention, normalization, skip connections, MLP)
    - MLP head for classification
- **Comprehensive Evaluation**: Generates detailed reports on model performance, including per-class accuracy.

---

## File Structure

```
Tranformers-ViT/
│
├── config.ini              # Configuration file for model and training parameters
├── main.py                 # Main script to run experiments
├── python/
│   ├── config_utils.py     # Utilities for managing configurations
│   ├── dataset_utils.py    # Utilities for handling datasets (e.g., MNIST)
│   └── VITs.py             # Implementation of the Vision Transformer model
│
├── data/                   # Directory for datasets
├── reports/                # Directory for evaluation reports
├── requirements.txt        # Pip requirements for standard setup
└── requirements_intel.txt  # Pip requirements for Intel GPU setup
```

---

## Getting Started

### Prerequisites

- Git
- A Python package manager like Conda or UV.

### 1. Clone the Repository

```bash
git clone <repository-url>
cd Tranformers-ViT
```

### 2. Set Up the Environment

You can use either Conda or UV to set up the project environment.

#### Conda Setup

```bash
# Create a new conda environment
conda create --name vit-env python=3.11 -y

# Activate the environment
conda activate vit-env

# Install the required packages
pip install -r requirements.txt

# For Intel GPUs, use the following command instead
pip install -r requirements_intel.txt
```

```bash
# Initialize the UV Project
uv init

# Add Dependencies with pip install, depending on GPU-Type
# CUDA:
uv pip install -r requirements.txt

# Intel:
uv pip install -r requirements_intel.txt
```

---

## Running the Project

The `main.py` script is the entry point for running experiments. You can override the default configurations from `config.ini` by passing command-line arguments.

```bash
uv run main.py
```

### Command-Line Arguments

You can override any of the parameters in `config.ini` using command-line arguments.

-   `--config_path`: Path to the configuration file (default: `config.ini`).
-   `--model_name`: The model to use (e.g., `VisionTransformer`).
-   `--dataset_name`: The dataset to use (e.g., `MNIST`).
-   `--batch_size`: The batch size for training.
-   `--num_epochs`: The number of training epochs.
-   `--learning_rate`: The learning rate for the optimizer.
-   `--patch_size`: The size of image patches.
-   `--img_size`: The input image size.
-   `--in_channels`: The number of input channels.
-   `--num_layers`: The number of Transformer encoder layers.
-   `--embed_dim`: The embedding dimension.
-   `--num_heads`: The number of attention heads.

**Example:**

```bash
python main.py --num_epochs 15 --learning_rate 0.0005 --embed_dim 256
```

---

## Configuration

The `config.ini` file allows you to set the default parameters for your experiments.

```ini
[General]
model_name = VisionTransformer
dataset_name = MNIST
batch_size = 32

[Training]
num_epochs = 10
learning_rate = 0.001

[ImagePatching]
patch_size = 7
img_size = 28
in_channels = 1

[TransformerEncoder]
num_layers = 6
embed_dim = 192
num_heads = 3
```

---

## Editable Parameters

### Image Patching

-   `patch_size`: Size of each square patch (e.g., 7 for 7x7 patches).
-   `img_size`: Input image size (e.g., 28 for MNIST).
-   `in_channels`: Number of image channels (1 for grayscale, 3 for RGB).

### Transformer Encoder

-   `num_layers`: Number of transformer encoder layers (blocks).
-   `embed_dim`: Dimensionality of patch embeddings and hidden representations.
-   `num_heads`: Number of attention heads in multi-head self-attention.
