# Fish Image Classification with CNN & Pretrained Models

This repository contains code for classifying fish images into multiple categories using a custom Convolutional Neural Network (CNN) and several pretrained models like VGG16, ResNet50, MobileNet, InceptionV3, and EfficientNetB0. The project includes preprocessing, training, evaluation, deployment, and performance comparison.

## Table of Contents
- [Project Overview](#project-overview)
- [Installation](#installation)
- [Dataset](#dataset)
- [Directory Structure](#directory-structure)
- [Preprocessing](#preprocessing)
- [Model](#model)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Usage](#usage)
- [License](#license)
- [Class Names](#class-names)

## Project Overview
The objective of this project is to classify fish species using deep learning models. It compares the performance of a basic CNN and several pretrained models to identify which architecture performs best for fish image classification.

- **Dataset**: Images labeled into 11 fish categories.
- **Techniques**: Custom CNN, data augmentation, pretrained model fine-tuning (VGG16, ResNet50, MobileNet, InceptionV3, EfficientNetB0).
- **Evaluation**: Metrics include accuracy, precision, recall, F1-score, and confusion matrix.

## Installation
```bash
pip install torch torchvision matplotlib numpy pandas seaborn scikit-learn tqdm
pip install streamlit pyngrok streamlit-folium nbconvert
npm install localtunnel
```

## Dataset
The dataset contains labeled fish images categorized into:
- `train/`: Training data
- `val/`: Validation data
- `test/`: Testing data

### Directory Structure
```
Dataset/
├── train/
│   ├── class1/
│   ├── class2/
├── val/
│   ├── class1/
│   ├── class2/
├── test/
│   ├── class1/
│   ├── class2/
```

## Preprocessing
- Resize images to 224x224
- Random horizontal flips
- Rotation between -15° to 15°
- Random affine transforms
- Normalize using ImageNet mean and std

## Model
### CNN Model
- 2 convolutional layers + ReLU + MaxPooling
- Dropout layers
- Fully connected layers for classification

### Pretrained Models
- VGG16, ResNet50, MobileNet, InceptionV3, EfficientNetB0
- Custom final layers adapted to 11 classes
```python
vgg16.classifier[6] = nn.Linear(num_features, 11)
```

## Loss and Optimizer
- **Loss**: CrossEntropyLoss
- **Optimizer**: Adam (lr=0.001)

## Training
- Trained for 20 epochs
- DataLoader for batching
- Transformations during train/validation
- Backpropagation and weight updates

## Evaluation
Metrics used:
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

## Results
### Model Performance Comparison
| Model         | Accuracy | Precision | Recall | F1 Score |
|---------------|----------|-----------|--------|----------|
| VGG16         | 85.5%    | 84.0%     | 85.0%  | 84.5%    |
| ResNet50      | 87.2%    | 86.5%     | 87.0%  | 86.7%    |
| MobileNet     | 89.0%    | 88.0%     | 89.2%  | 88.6%    |
| InceptionV3   | 90.3%    | 89.5%     | 90.0%  | 89.7%    |
| EfficientNetB0| 91.1%    | 90.5%     | 91.2%  | 90.9%    |

## Usage
Run the Streamlit app from Colab or local:
```python
!streamlit run /content/app.py &>/content/logs.txt & npx localtunnel --port 8501
```

## Class Names
```python
class_names = [
    'animal fish', 'animal fish bass', 'fish sea_food black_sea_sprat',
    'fish sea_food gilt_head_bream', 'fish sea_food hourse_mackerel',
    'fish sea_food red_mullet', 'fish sea_food red_sea_bream',
    'fish sea_food sea_bass', 'fish sea_food shrimp',
    'fish sea_food striped_red_mullet', 'fish sea_food trout'
]
```

## License
This project is open source and available under the [MIT License](LICENSE).

