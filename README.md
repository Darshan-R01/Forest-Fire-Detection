# Forest Fire Detection Using Deep Learning

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-DL-red?logo=keras)
![Google Colab](https://img.shields.io/badge/Google%20Colab-Supported-yellow?logo=googlecolab)

A custom Convolutional Neural Network (CNN) built with TensorFlow and Keras to perform real-time binary image classification (**Fire** vs. **No-Fire**) across static datasets and live web feeds.

# Project Overview

Early detection of wildfires is critical for mitigating environmental and economic damage. This project delivers an automated computer vision pipeline designed to process aerial, satellite, or ground-level images with high precision and low operational cost. 

* **Binary Classification:** Accurately differentiates between `fire` and `nofire` scenes.
* **Flexible Input Sources:** Handles local dataset samples (e.g., *the-wildfire-dataset*) as well as dynamic web URLs.
* **Lightweight & Open Source:** Built entirely using open-source Python frameworks accessible via cloud hardware like Google Colab.


# Tech Stack & Tools

| Category | Tools & Libraries |
| :--- | :--- |
| **Language** | Python |
| **Frameworks** | TensorFlow, Keras |
| **Data & Image Processing** | KaggleHub, PIL (Pillow) |
| **Visualization** | Matplotlib |
| **Environment & Version Control** | Google Colab, GitHub |

#  Model Architecture
The model utilizes a custom `Sequential` CNN architecture with a total of 490,689 parameters (489,729 trainable). 

**Layer Breakdown:**

1. **Convolutional Blocks:** Four `Conv2D` layers progressively scaling in filter size (32 ➔ 64 ➔ 128 ➔ 256).
2. **Normalization & Pooling:** Each convolutional layer is immediately followed by `BatchNormalization`, an `Activation` function, and `MaxPooling2D` to extract dominant features efficiently.
3. **Global Average Pooling:** A `GlobalAveragePooling2D` layer replaces traditional flattening to significantly reduce the parameter count and mitigate overfitting risks.
4. **Dense & Dropout Layers:** A fully connected `Dense` layer (256 units) followed by a `Dropout` layer.
5. **Output Layer:** A final single-unit `Dense` layer for binary classification output.

# Methodology & Training Performance

* **Data Augmentation:** Applied dynamic rotations, zooms, and flips to enrich feature diversity and minimize overfitting.
* **Training Setup:** The model was trained over 15 epochs. 
* **Model Convergence:** 
  * The **Model Accuracy** graph demonstrates the training accuracy stabilizing between 0.7 and 0.8, while validation accuracy climbs significantly around epoch 5 before settling near 0.7.
  * The **Model Loss** graph shows validation loss dropping rapidly in early epochs and following the downward trend of the training loss toward the final epochs.

#  Key Features & Prediction Pipeline

* **End-to-End Automation:** Complete pipeline handling image fetching, preprocessing, inference, and visualization.
* **Real-time Visual Feedback:** Generates annotated image outputs displaying class labels (`Predicted: nofire` or `Predicted: fire`) alongside the processed visual feed.
* **Batch Processing:** Supports evaluating multi-image batches for ongoing surveillance using custom functions like `predict_fire()`.

