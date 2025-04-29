# Multiclass-Fish-Image-Classifications
Multiclass Fish Image Classifications
# Fish Image Classification with Transfer Learning and Streamlit App

This project demonstrates how to classify fish images into multiple categories using Convolutional Neural Networks (CNN) and Transfer Learning with pre-trained models. It utilizes popular models like VGG16, ResNet50, MobileNet, InceptionV3, and EfficientNetB0. Additionally, a **Streamlit app** is developed to allow users to interact with the model, upload fish images, and get predictions.

## Table of Contents

- [Project Overview](#project-overview)
- [Installation Instructions](#installation-instructions)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Models Used](#models-used)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Streamlit Fish Classification App](#streamlit-fish-classification-app)
- [Usage](#usage)
- [License](#license)
- [Contributors](#contributors)

## Project Overview

This project uses **Convolutional Neural Networks (CNNs)** and **Transfer Learning** techniques for classifying fish species into 11 different categories. The project demonstrates both training a custom CNN model and fine-tuning pre-trained models for image classification tasks. It includes the development of a **Streamlit app** to interact with the model for real-time predictions.

### Key Techniques Used:
- **Convolutional Neural Networks (CNNs)**
- **Transfer Learning**
- **Data Augmentation**
- **Model Evaluation with Metrics:**
    - Accuracy
    - Precision
    - Recall
    - F1-score
    - Confusion Matrix
Prepare the dataset: Ensure your dataset is organized into train, val, and test directories with subfolders for each class of fish.

Dataset
The dataset consists of labeled images of fish species divided into three main categories:

Train: For training the model.

Validation: For tuning hyperparameters and selecting the best model.

Test: For evaluating the final model's performance.

Example directory structure:

plaintext
Copy
Edit
Dataset/
├── train/
│   ├── class1/
│   ├── class2/
│   ├── ...
├── val/
│   ├── class1/
│   ├── class2/
│   ├── ...
├── test/
│   ├── class1/
│   ├── class2/
│   ├── ...
Preprocessing
The images are preprocessed using the following steps:

Resizing: All images are resized to 224x224 pixels.

Data Augmentation: Includes random horizontal flips, rotations, and affine transformations for better generalization.

Normalization: Images are normalized using mean and standard deviation values for the RGB channels.

Models Used
The project includes the following models:

Custom CNN: A simple CNN architecture with convolutional layers, dropout, and fully connected layers.

VGG16: Fine-tuned for the fish image classification task.

ResNet50: A residual network that is also fine-tuned.

MobileNet: A lightweight model that is fine-tuned.

InceptionV3: A model known for multi-level feature extraction.

EfficientNetB0: Efficiently balances depth, width, and resolution.

Model Fine-Tuning:
The final layer of each model is modified to output predictions for the 11 classes of fish.

Example of fine-tuning VGG16:

python
Copy
Edit
vgg16.classifier[6] = nn.Linear(num_features, 11)
Training
The models are trained using PyTorch for 20 epochs:

Data Loading: The dataset is loaded using DataLoader.

Forward Pass: Images are passed through the model.

Backward Pass: Gradients are computed using backpropagation.

Optimization: Weights are updated using the Adam optimizer.

Epochs: The model is trained for 20 epochs to achieve good performance.

Evaluation
Model performance is evaluated using the following metrics:

Accuracy

Precision

Recall

F1 Score

Confusion Matrix
