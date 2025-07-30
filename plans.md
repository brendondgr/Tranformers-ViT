# Project Plans for Vision Transformers

## 1. Project Overview

The primary goal of this project is to practice and implement Transformer architectures, with a specific focus on Vision Transformers (ViTs). The project is designed to provide hands-on experience with ViTs by building a customizable model for image classification tasks. The initial target dataset is the MNIST handwriting dataset, which consists of grayscale images of handwritten digits.

---

## 2. Purpose and Use-Case

Vision Transformers have shown significant promise in computer vision tasks by leveraging the self-attention mechanism of Transformers, originally developed for natural language processing. This project aims to:

- Deepen understanding of Transformer-based models in the context of vision.
- Provide a flexible framework for experimenting with ViT architectures.
- Enable users to easily modify model parameters and observe their effects on performance.
- Serve as a foundation for further research or application to other image datasets.

The use-case is educational and experimental: users can adjust model parameters via a configuration file and run experiments to see how these changes impact results on the MNIST dataset.

---

## 3. Project Goals

1. **Create a Vision Transformer with Editable Parameters:**  
   The model, referred to as **VITStandard**, will allow users to specify key architectural parameters through a configuration file (`config.ini`). This enables easy experimentation and reproducibility.

2. **Implement a Standard ViT Structure:**  
   The VITStandard model will include the following components:
   - **Image Patching**
   - **Patch Embedding**
   - **Transformer Encoder**
     - Normalization
     - Multi-head Attention
     - Skip Connection
     - Normalization
     - MLP/Feed Forward Network (FFN)
     - Skip Connection
   - **MLP Head**
   - **Output/Results**

---

## 4. Editable Parameters

The following parameters can be set in `config.ini` to customize the VITStandard model:

### 4.1 Image Patching Parameters

- **patch_size**:  
  The size (in pixels) of each square patch extracted from the input image. For example, a `patch_size` of 7 on a 28x28 image will divide the image into 16 patches of size 7x7. This controls the granularity of the input representation.

- **img_size**:  
  The size of the input images (height and width). For MNIST, this is typically 28. This parameter ensures the model is compatible with the dataset.

- **in_channels**:  
  The number of channels in the input images. For grayscale images like MNIST, this should be set to 1. For RGB images, it would be 3.

### 4.2 Transformer Encoder Parameters

- **num_layers**:  
  The number of transformer encoder layers (also called blocks). Increasing this value allows the model to learn more complex representations, but may increase training time and risk overfitting.

- **embed_dim**:  
  The dimensionality of the patch embeddings and the hidden representations within the transformer. Larger values can capture more information but require more computation.

- **num_heads**:  
  The number of attention heads in the multi-head self-attention mechanism. More heads allow the model to focus on different parts of the input simultaneously.

---

## 5. Configuration File (`config.ini`)

All editable parameters should be specified in the `config.ini` file as follows:

```ini
[General]
model_name = VisionTransformer
dataset_name = MNIST

[ImagePatching]
patch_size = <value>
img_size = <value>
in_channels = <value>

[TransformerEncoder]
num_layers = <value>
embed_dim = <value>
num_heads = <value>
```

Replace `<value>` with the desired setting for each parameter.

---

## 6. How to Use

1. **Edit `config.ini`:**  
   Set the desired values for each parameter in the configuration file.

2. **Run the Python Script:**  
   Execute the main script (e.g., `python main.py`). The script will read the configuration and build the VITStandard model accordingly.

3. **Experiment:**  
   Modify parameters in `config.ini` to experiment with different model configurations and observe their effects on performance.

---

## 7. Conclusion

This documentation provides a clear guide for understanding, configuring, and experimenting with the Vision Transformer model in this project. By adjusting the parameters in `config.ini`, users can explore the impact of architectural choices on the MNIST classification task, deepening their understanding of Vision Transformers.