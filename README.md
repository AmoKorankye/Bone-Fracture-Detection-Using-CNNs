# Bone Fracture Detection Using Deep Learning

This project implements a convolutional neural network (CNN) to detect bone fractures in X-ray images. The model classifies X-ray images into two categories: fractured and not fractured bones.

## Overview

The project uses a custom CNN architecture implemented in TensorFlow/Keras to classify bone X-ray images. It includes:

- Data loading and preprocessing
- Data augmentation
- Model architecture design and training 
- Model evaluation and visualization
- Testing and performance metrics

## Requirements

- Python 3.x
- TensorFlow 
- Keras
- OpenCV
- NumPy
- Pandas 
- Matplotlib
- Seaborn
- scikit-learn
- kagglehub

## Dataset

The dataset is sourced from Kaggle's "Bone Fracture" dataset containing X-ray images split into:
- Training set
- Validation set  
- Test set

Each image is labeled as either fractured or not fractured.

## Model Architecture

The CNN model consists of:
- Multiple convolutional layers with ReLU activation
- Batch normalization layers
- Max pooling layers
- Dropout layers for regularization
- Dense layers
- Final sigmoid activation for binary classification

## Performance Metrics

The model is evaluated using:
- Accuracy
- Loss
- Sensitivity
- Specificity
- AUC-ROC
- Confusion Matrix
- Classification Report

## Usage

1. Install required dependencies
2. Download the dataset using kagglehub
3. Run the Jupyter notebook to:
   - Load and preprocess data
   - Train the model
   - Evaluate performance 
   - Make predictions

## Files

- `Bone_Fracture.ipynb`: Main Jupyter notebook containing the implementation

## Results

The model achieves:
- High accuracy in classifying bone fractures
- Good sensitivity and specificity metrics
- Reliable performance on test set predictions

## Author

Kwaku Amo-Korankye
10211100331
Academic City