# Vision Transformers (ViT) for MNIST

This project implements a customizable Vision Transformer (ViT) model for image classification, with a focus on the MNIST handwritten digit dataset. It is designed for educational and experimental purposes, allowing users to easily modify model parameters and observe their effects.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Editable Parameters](#editable-parameters)
- [Configuration](#configuration)
- [Getting Started](#getting-started)

---

## Project Overview

Vision Transformers (ViTs) bring the power of Transformer architectures—originally developed for natural language processing—to computer vision tasks. This project provides a hands-on framework for building, training, and experimenting with ViTs on the MNIST dataset.

---

## Features

- **Customizable ViT Model:**  
  All key architectural parameters are editable via a configuration file (`config.ini`).

- **Standard ViT Structure:**  
  - Image patching and embedding
  - Transformer encoder blocks (multi-head attention, normalization, skip connections, MLP)
  - MLP head for classification

- **Easy Experimentation:**  
  Modify parameters and instantly see their impact on model performance.

---

## Editable Parameters

You can control the following parameters in `config.ini`:

### Image Patching

- `patch_size`: Size of each square patch (e.g., 7 for 7x7 patches)
- `img_size`: Input image size (e.g., 28 for MNIST)
- `in_channels`: Number of image channels (1 for grayscale, 3 for RGB)

### Transformer Encoder

- `num_layers`: Number of transformer encoder layers (blocks)
- `embed_dim`: Dimensionality of patch embeddings and hidden representations
- `num_heads`: Number of attention heads in multi-head self-attention

---

## Configuration

Edit the `config.ini` file to set your desired model parameters. Example:

```ini
[General]
model_name = VisionTransformer
dataset_name = MNIST

[ImagePatching]
patch_size = 7
img_size = 28
in_channels = 1

[TransformerEncoder]
num_layers = 6
embed_dim = 64
num_heads = 8
```